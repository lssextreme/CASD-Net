# -*- coding: utf-8 -*-
"""
CASD-Net: Context-Aware Statistical Proxy with Dynamic Modality Arbitration.

This module implements the network described in the CASD-Net paper
("Context-Aware Statistical Proxy ..."). The names below map one-to-one to the
methodology sections of the paper:

    CASDNet   : the full dual-stream segmentation pipeline (frozen SAM ViT-B
                encoder + LoRA, a four-level feature pyramid, and a lightweight
                UNetFormer-style decoder).            [Sec. "Overall Architecture"]
    CASP      : Context-Aware Statistical Proxy. A per-level, modality-specific
                spatial aggregator that maps the mean / standard-deviation /
                maximum statistics of each stream to a channel condition S^s.
                                                     [Sec. "Context-Aware Statistical Proxy"]
    DMA       : Dynamic Modality Arbitration. Combines the ordered local feature
                pair with S^s and produces a dense complementary gate A^s.
                                                     [Sec. "Dynamic Modality Arbitration"]
    MOCLoss   : Modality Orthogonal Constraint Loss. A corresponding-channel
                decorrelation term on the deepest optical/elevation features.
                                                     [Sec. "Modality Orthogonal Constraint Loss"]

The encoder is a *shared, frozen* SAM ViT-B. LoRA adapters (parameters whose
name contains ``lora_``) are the only trainable encoder parameters; every other
encoder parameter is kept frozen to realise the parameter-efficient setting
reported in the paper (~6.5M trainable parameters).
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

from MedSAM.models.sam import sam_model_registry

# Minimal argument bundle expected by the vendored SAM ViT-B backbone. The
# interactive ``MedSAM.cfg.parse_args()`` is deliberately *not* used here so
# that ``train.py`` / ``predict.py`` can expose their own command-line interface
# without an argparse clash. See MedSAM/models/sam/build_sam.py and
# MedSAM/models/sam/modeling/image_encoder.py for where each field is read.
SAM_DEFAULT_ARGS = SimpleNamespace(
    mod='sam_lora',        # use LoRA adapters inside every encoder block
    mid_dim=None,          # None -> default LoRA rank (4)
    image_size=1024,       # positional-embedding grid (interpolated to input)
    multimask_output=1,
)


# =============================================================================
# Loss
# =============================================================================
class MOCLoss(nn.Module):
    r"""Modality Orthogonal Constraint Loss (MOC-Loss).

    Matches Eq. (``L_MOC``) of the paper::

        L_MOC = (1 / C) * sum_{i=1}^{C} | < v_O^(i), v_E^(i) > |

    where ``v_O^(i)`` and ``v_E^(i)`` are the *corresponding* (same-index)
    channels of the spatially flattened optical and elevation features after
    spatial L2-normalisation. Only the diagonal of the cross-correlation matrix
    is penalised, so off-diagonal (non-corresponding) channel pairs are free to
    differ. The absolute value is used because a sign-reversed pattern encodes
    the same one-dimensional spatial structure and therefore still constitutes
    corresponding-channel redundancy.
    """

    def forward(self, f_optical, f_elevation):
        """
        Args:
            f_optical   : [B, C, H, W] optical (RGB) feature map.
            f_elevation : [B, C, H, W] elevation (DSM) feature map.

        Returns:
            A scalar loss in [0, 1].
        """
        B, C, H, W = f_optical.shape

        # Flatten the spatial dimensions: [B, C, H, W] -> [B, C, N], N = H*W.
        v_o = f_optical.view(B, C, -1)
        v_e = f_elevation.view(B, C, -1)

        # L2-normalise each channel along the spatial axis (the cosine
        # similarity pre-requisite).
        v_o = F.normalize(v_o, p=2, dim=2)
        v_e = F.normalize(v_e, p=2, dim=2)

        # Element-wise dot product keeps only the *corresponding* channels
        # (the diagonal of the C x C correlation matrix) -> [B, C].
        corr = (v_o * v_e).sum(dim=2)

        return corr.abs().mean()


# =============================================================================
# Lightweight building blocks (shared by the decoder, per the UNetFormer design
# referenced in the paper).
# =============================================================================
class Norm2d(nn.Module):
    """LayerNorm adapted to [B, C, H, W] tensors."""

    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6(),
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6(),
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class Mlp(nn.Module):
    """1x1-convolution MLP used inside the decoder transformer blocks."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    """Window attention combined with a local (depth-wise) convolution path.

    This is the global-local attention head of the UNetFormer-style decoder.
    """

    def __init__(self, dim=256, num_heads=16, qkv_bias=False, window_size=8,
                 relative_pos_embedding=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,
                                   padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1,
                                   padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding
        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        return F.pad(x, pad=(0, 1, 0, 1), mode='reflect')

    def forward(self, x):
        B, C, H, W = x.shape
        local = self.local2(x) + self.local1(x)
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(
            qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
            h=self.num_heads, d=C // self.num_heads, hh=Hp // self.ws,
            ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(
            attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
            h=self.num_heads, d=C // self.num_heads, hh=Hp // self.ws,
            ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]
        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]
        return out


class Block(nn.Module):
    """A decoder transformer block (attention + MLP with skip connections)."""

    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.ReLU6,
                 norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                         window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    """Learnable weighted fusion of a decoder feature and a skip feature."""

    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    """Finest-scale refinement head with spatial (PA) and channel (CA) attention."""

    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1,
                      groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(decode_channels, decode_channels // 16, kernel_size=1),
            nn.ReLU6(),
            Conv(decode_channels // 16, decode_channels, kernel_size=1),
            nn.Sigmoid())
        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        x = self.pa(x) * x + self.ca(x) * x
        x = self.proj(x) + shortcut
        return self.act(x)


class Decoder(nn.Module):
    """Lightweight UNetFormer-style segmentation decoder.

    Consumes the four fused pyramid features (res1..res4) and produces the
    pixel-wise class logits at the input resolution.
    """

    def __init__(self, encoder_channels=(64, 128, 256, 512), decode_channels=64,
                 dropout=0.1, window_size=8, num_classes=6):
        super(Decoder, self).__init__()
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)
        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)
        x = self.p2(x, res2)
        x = self.b2(x)
        x = self.p1(x, res1)
        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# =============================================================================
# CASP: Context-Aware Statistical Proxy  (core contribution 1)
# =============================================================================
class CASP(nn.Module):
    r"""Context-Aware Statistical Proxy.

    For each modality, CASP computes *three* channel-wise scene statistics over
    all spatial positions of the level-s feature map (Eqs. ``mu``, ``sigma``,
    ``gamma`` in the paper)::

        mu_c    = (1 / HW) * sum_{i,j} F_{c,i,j}                  (baseline activation)
        sigma_c = sqrt( (1 / HW) * sum_{i,j} (F_{c,i,j} - mu_c)^2 + eps )   (dispersion)
        gamma_c = max_{i,j} F_{c,i,j}                             (sparse extreme)

    The per-modality descriptors ``d_m = [mu, sigma, gamma]`` are concatenated
    in modality order and projected to a *channel condition* ``S^s`` through a
    two-layer MLP with a hidden width of 128 and a ReLU activation::

        S^s = W_2^s * delta( W_1^s * [d_O^s, d_E^s] )

    Note that ``S^s`` is a per-channel vector (C dims), *not* a single scalar:
    it conditions DMA channel by channel. Each pyramid level uses its own CASP
    instance (``level-specific`` MLPs, as stated in the implementation details).
    """

    def __init__(self, channels, hidden_dim=128, eps=1e-8):
        super(CASP, self).__init__()
        self.eps = eps
        # 3 statistics x 2 modalities -> 6 * C descriptor dimension.
        mlp_in = channels * 6
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
        )

    def _channel_statistics(self, F):
        """Return (mean, std, max) of F, each of shape [B, C]."""
        B, C = F.shape[:2]
        mu = F.mean(dim=(2, 3))
        sigma = torch.sqrt(F.var(dim=(2, 3), unbiased=False) + self.eps)
        gamma = F.view(B, C, -1).max(dim=2).values
        return mu, sigma, gamma

    def forward(self, f_optical, f_elevation):
        """
        Args:
            f_optical   : [B, C, H, W] optical feature map at level s.
            f_elevation : [B, C, H, W] elevation feature map at level s.

        Returns:
            S : [B, C] channel condition (one scalar per channel).
        """
        mu_o, sigma_o, gamma_o = self._channel_statistics(f_optical)
        mu_e, sigma_e, gamma_e = self._channel_statistics(f_elevation)

        # Ordered channel concatenation, preserving modality identity.
        d = torch.cat([mu_o, sigma_o, gamma_o, mu_e, sigma_e, gamma_e], dim=1)
        return self.mlp(d)  # [B, C]


# =============================================================================
# DMA: Dynamic Modality Arbitration  (core contribution 2)
# =============================================================================
class DMA(nn.Module):
    r"""Dynamic Modality Arbitration.

    DMA turns the global channel condition ``S^s`` into a dense, spatially
    localised gate. A lightweight convolution maps the ordered 2C-channel local
    concatenation to C channels; the resulting logit is then modulated channel
    by channel with ``S^s`` before a sigmoid bounds it to (0, 1)::

        Z_{c,x,y} = ( W_c^s [F_O^s, F_E^s] )_{c,x,y} * S_c^s
        A_{c,x,y} = 1 / (1 + exp(-Z_{c,x,y}))
        F_fused   = A (x) F_O + (1 - A) (x) F_E

    The complementary weighting permits optical-dominant, elevation-dominant,
    and intermediate mixtures at every pixel and channel.
    """

    def __init__(self, channels):
        super(DMA, self).__init__()
        # Local multi-modal correlation conv: 2C -> C at the same resolution.
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, f_optical, f_elevation, S):
        """
        Args:
            f_optical   : [B, C, H, W] optical feature map.
            f_elevation : [B, C, H, W] elevation feature map.
            S           : [B, C] channel condition from CASP.

        Returns:
            f_fused : [B, C, H, W] arbitration-weighted fused feature.
        """
        B, C, H, W = f_optical.shape

        # Local path: 2C -> C logits that keep track of which stream produced
        # each response.
        z = self.conv(torch.cat([f_optical, f_elevation], dim=1))  # [B, C, H, W]

        # Channel-wise modulation by the CASP condition (broadcast over space).
        z = z * S.view(B, C, 1, 1)

        A = torch.sigmoid(z)                                       # dense gate in (0,1)
        return A * f_optical + (1 - A) * f_elevation


# =============================================================================
# CASDNet: the full pipeline
# =============================================================================
class CASDNet(nn.Module):
    """CASD-Net dual-stream multimodal segmentation pipeline.

    Architecture (matching the paper's "Overall Architecture" figure):

        1. A *shared frozen* SAM ViT-B encoder encodes the optical and elevation
           streams in parallel; only the LoRA adapters are trainable.
        2. A four-level feature pyramid produces ordered feature pairs at
           [64x64, 32x32, 16x16, 8x8].
        3. At each level, CASP summarises the pair into a channel condition S^s.
        4. At each level, DMA fuses the pair under the guidance of S^s.
        5. The fused pyramid is decoded into per-pixel logits; during training
           the deepest optical/elevation features are additionally returned for
           MOC-Loss.
    """

    def __init__(self, num_classes=6, decode_channels=64, dropout=0.1,
                 window_size=8, sam_checkpoint='weights/sam_vit_b_01ec64.pth',
                 args=None):
        super().__init__()

        if args is None:
            args = SAM_DEFAULT_ARGS
        # Build the SAM ViT-B backbone (LoRA mode). ``sam_checkpoint`` must be
        # the SAM ViT-B weights (sam_vit_b_01ec64.pth); see README for download.
        self.sam = sam_model_registry["vit_b"](args, checkpoint=sam_checkpoint)
        self.image_encoder = self.sam.image_encoder

        # The single-channel DSM is mapped to three channels with a 3x3 conv
        # before the shared patch embedding (per the implementation details).
        self.dsm_embed = nn.Conv2d(1, 3, kernel_size=3, padding=1)

        encoder_channels = (256, 256, 256, 256)

        # --- Feature pyramid (optical stream) ---
        self.fpn1x = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            Norm2d(256), nn.GELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn2x = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2))
        self.fpn3x = nn.Identity()
        self.fpn4x = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Feature pyramid (elevation stream) ---
        self.fpn1y = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn2y = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2))
        self.fpn3y = nn.Identity()
        self.fpn4y = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Four level-specific CASP modules (one per pyramid level) ---
        self.casp1 = CASP(channels=256)
        self.casp2 = CASP(channels=256)
        self.casp3 = CASP(channels=256)
        self.casp4 = CASP(channels=256)

        # --- Four level-specific DMA modules (one per pyramid level) ---
        self.dma1 = DMA(channels=encoder_channels[0])
        self.dma2 = DMA(channels=encoder_channels[1])
        self.dma3 = DMA(channels=encoder_channels[2])
        self.dma4 = DMA(channels=encoder_channels[3])

        # --- Freeze the encoder; keep only LoRA adapters trainable ---
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = ('lora_' in name)

        self.decoder = Decoder(encoder_channels, decode_channels, dropout,
                               window_size, num_classes)

    def forward(self, x, y):
        """
        Args:
            x : [B, 3, H, W] optical (RGB) image, normalised to [0, 1].
            y : [B, H, W] or [B, 1, H, W] elevation (DSM) map.

        Returns:
            seg_logits : [B, num_classes, H, W] segmentation logits.
            f_optical  : [B, 256, h, w] deepest optical features (for MOC-Loss).
            f_elevation: [B, 256, h, w] deepest elevation features (for MOC-Loss).
        """
        h, w = x.size()[-2:]

        # Normalise the elevation tensor to [B, 1, H, W], then embed to 3 ch.
        if y.dim() == 3:
            y = y.unsqueeze(1)
        if y.shape[1] == 1:
            y = self.dsm_embed(y)

        # 1. Shared frozen SAM encoder (dual-stream).
        deepx, deepy = self.image_encoder(x, y)          # [B, 256, 16, 16]

        # 2. Four-level feature pyramid.
        res1x, res2x = self.fpn1x(deepx), self.fpn2x(deepx)
        res3x, res4x = self.fpn3x(deepx), self.fpn4x(deepx)
        res1y, res2y = self.fpn1y(deepy), self.fpn2y(deepy)
        res3y, res4y = self.fpn3y(deepy), self.fpn4y(deepy)

        # 3. Per-level CASP: map each pair to a channel condition S^s.
        S1 = self.casp1(res1x, res1y)
        S2 = self.casp2(res2x, res2y)
        S3 = self.casp3(res3x, res3y)
        S4 = self.casp4(res4x, res4y)

        # 4. Per-level DMA: fuse each pair under the guidance of S^s.
        res1 = self.dma1(res1x, res1y, S1)
        res2 = self.dma2(res2x, res2y, S2)
        res3 = self.dma3(res3x, res3y, S3)
        res4 = self.dma4(res4x, res4y, S4)

        # 5. Decode into logits.
        seg_logits = self.decoder(res1, res2, res3, res4, h, w)

        # Return the deepest features for MOC-Loss (ignored at inference time).
        return seg_logits, deepx, deepy
