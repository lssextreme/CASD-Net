# -*- coding: utf-8 -*-
"""
Data loading and evaluation utilities for CASD-Net.

Every function in this module is dataset-agnostic: it receives the relevant
:class:`~config.DatasetConfig` (or a palette / label list) as an explicit
argument, so no absolute path or dataset name is hard-coded here.
"""

import itertools
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from sklearn.metrics import confusion_matrix


# =============================================================================
# Colour <-> label conversion
# =============================================================================
def convert_to_color(arr_2d, palette):
    """Map an integer label map ``[H, W]`` to an RGB image ``[H, W, 3]``."""
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for label, color in palette.items():
        arr_3d[arr_2d == label] = color
    return arr_3d


def convert_from_color(arr_3d, palette):
    """Map an RGB label image ``[H, W, 3]`` back to an integer label map."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for label, color in palette.items():
        match = np.all(arr_3d == np.asarray(color).reshape(1, 1, 3), axis=2)
        arr_2d[match] = label
    return arr_2d


# =============================================================================
# Tile readers (all return numpy arrays on CPU)
# =============================================================================
def read_image(path):
    """Read a (possibly 4-channel) ortho tile and return ``[3, H, W]`` in [0, 1].

    Potsdam stores IR-R-G-B imagery, so only the first three channels (R-G-B)
    are kept. Vaihingen and Hunan tiles are already 3-channel and are passed
    through unchanged.
    """
    arr = np.asarray(io.imread(path), dtype='float32')
    arr = arr[..., :3]                      # R-G-B (drop IR for Potsdam)
    return (arr / 255.0).transpose(2, 0, 1)


def read_dsm(path, normalization='local'):
    """Read a DSM and optionally normalise it.

    Args:
        normalization:
            'local' -> per-tile min-max scaling to [0, 1] (ISPRS).
            'none'  -> no scaling; the tile is assumed to be pre-normalised
                       (Hunan, whose DSM is already mapped to a global range).
    """
    dsm = np.asarray(io.imread(path), dtype='float32')
    if normalization == 'local':
        dmin, dmax = dsm.min(), dsm.max()
        dsm = (dsm - dmin) / (dmax - dmin + 1e-8)
    elif normalization == 'none':
        pass
    else:
        raise ValueError(f'unknown dsm_normalization: {normalization!r}')
    return dsm


def read_label(path, label_format='color', palette=None):
    """Read a label tile.

    Args:
        label_format: 'color' (ISPRS RGB labels, decoded via ``palette``) or
                      'index' (Hunan single-channel integer labels).
    """
    if label_format == 'color':
        assert palette is not None
        return np.asarray(
            convert_from_color(io.imread(path), palette), dtype='int64')
    elif label_format == 'index':
        return np.asarray(io.imread(path), dtype='int64')
    raise ValueError(f'unknown label_format: {label_format!r}')


# =============================================================================
# Dataset
# =============================================================================
class ISPRS_dataset(torch.utils.data.Dataset):
    """Random-patch (or full-tile) dataset built from a :class:`DatasetConfig`.

    For the ISPRS datasets (``label_format == 'color'``) a random
    ``window_size`` crop is sampled on every access; for Hunan
    (``label_format == 'index'``) the pre-tiled image is returned as a whole,
    which matches the tile-oriented pre-processing used in the paper.
    """

    def __init__(self, cfg, ids, cache=False, augmentation=True):
        super().__init__()
        self.cfg = cfg
        self.augmentation = augmentation
        self.cache = cache
        self.full_tile = (cfg.label_format == 'index')   # Hunan -> no cropping

        self.data_files = [cfg.data_template.format(id=id_) for id_ in ids]
        self.dsm_files = [cfg.dsm_template.format(id=id_) for id_ in ids]
        self.label_files = [cfg.label_template.format(id=id_) for id_ in ids]

        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError(f'{f} is not a file ! (check ``root`` in config.py)')

        self.data_cache_, self.dsm_cache_, self.label_cache_ = {}, {}, {}

    def __len__(self):
        # Fixed sampling budget per epoch (patch sampling), independent of the
        # DataLoader batch size: ``steps_per_epoch`` batches are produced.
        return self.cfg.steps_per_epoch

    @staticmethod
    def _augment(*arrays, flip=True, mirror=True):
        will_flip = flip and random.random() < 0.5
        will_mirror = mirror and random.random() < 0.5
        out = []
        for arr in arrays:
            if will_flip:
                arr = arr[::-1, :] if arr.ndim == 2 else arr[:, ::-1, :]
            if will_mirror:
                arr = arr[:, ::-1] if arr.ndim == 2 else arr[:, :, ::-1]
            out.append(np.copy(arr))
        return tuple(out)

    def __getitem__(self, idx):
        random_idx = random.randint(0, len(self.data_files) - 1)

        # --- image ---
        if random_idx in self.data_cache_:
            data = self.data_cache_[random_idx]
        else:
            data = read_image(self.data_files[random_idx])
            if self.cache:
                self.data_cache_[random_idx] = data

        # --- dsm ---
        if random_idx in self.dsm_cache_:
            dsm = self.dsm_cache_[random_idx]
        else:
            dsm = read_dsm(self.dsm_files[random_idx], self.cfg.dsm_normalization)
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        # --- label ---
        if random_idx in self.label_cache_:
            label = self.label_cache_[random_idx]
        else:
            label = read_label(self.label_files[random_idx], self.cfg.label_format,
                               self.cfg.palette)
            if self.cache:
                self.label_cache_[random_idx] = label

        if self.full_tile:
            data_p, dsm_p, label_p = data, dsm, label
        else:
            x1, x2, y1, y2 = get_random_pos(data, self.cfg.window_size)
            data_p = data[:, x1:x2, y1:y2]
            dsm_p = dsm[x1:x2, y1:y2]
            label_p = label[x1:x2, y1:y2]

        if self.augmentation:
            data_p, dsm_p, label_p = self._augment(data_p, dsm_p, label_p)

        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))


def get_random_pos(img, window_shape):
    """Sample the top-left corner of a ``window_shape`` crop inside ``img``."""
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    y1 = random.randint(0, H - h - 1)
    return x1, x1 + w, y1, y1 + h


# =============================================================================
# Losses
# =============================================================================
class CrossEntropy2d_ignore(nn.Module):
    """2D cross-entropy that ignores pixels labelled ``ignore_label`` (255)."""

    def __init__(self, reduction='mean', ignore_label=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        n, c, h, w = predict.size()
        target_mask = (target >= 0) & (target != self.ignore_label)
        target = target[target_mask]
        if target.numel() == 0:
            return torch.zeros(1, device=predict.device, requires_grad=True)

        predict = predict.permute(0, 2, 3, 1).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        return F.cross_entropy(predict, target, weight=weight,
                               reduction=self.reduction)


def seg_loss(pred, label, weight=None, ignore_label=255):
    """Convenience wrapper: semantic segmentation cross-entropy loss."""
    return CrossEntropy2d_ignore(ignore_label=ignore_label)(pred, label, weight)


# =============================================================================
# Sliding-window inference helpers
# =============================================================================
def sliding_window(top, step=10, window_size=(20, 20)):
    """Yield ``(x, y, w, h)`` windows covering ``top`` with the given stride.

    The last window on each axis is clamped so the image is fully covered.
    """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """Count how many windows :func:`sliding_window` would produce."""
    return sum(1 for _ in sliding_window(top, step, window_size))


def grouper(n, iterable):
    """Group an iterator into chunks of at most ``n`` items."""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# =============================================================================
# Evaluation metrics
# =============================================================================
def metrics(predictions, gts, labels):
    """Compute OA / per-class IoU & F1 / mIoU / kappa from flattened arrays.

    Args:
        predictions: 1-D int array of predicted labels.
        gts:          1-D int array of ground-truth labels.
        labels:       list of class names (also fixes the class ordering).

    Returns:
        dict with keys ``oa``, ``mIoU``, ``iou``, ``f1``, ``mF1``, ``kappa``,
        ``cm`` (confusion matrix).
    """
    n_classes = len(labels)
    cm = confusion_matrix(gts, predictions, labels=range(n_classes))

    total = cm.sum()
    oa = np.diag(cm).sum() / float(total)

    intersection = np.diag(cm)
    gt_sum = cm.sum(axis=1)
    pred_sum = cm.sum(axis=0)
    union = gt_sum + pred_sum - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float),
                    where=union != 0)
    f1 = np.divide(2 * intersection, gt_sum + pred_sum,
                   out=np.zeros_like(intersection, dtype=float),
                   where=(gt_sum + pred_sum) != 0)

    mIoU = np.nanmean(iou)
    mF1 = np.nanmean(f1)

    pa = np.trace(cm) / float(total)
    pe = np.sum(gt_sum * pred_sum) / float(total * total)
    kappa = (pa - pe) / (1 - pe + 1e-8)

    return {'oa': oa, 'mIoU': mIoU, 'iou': iou, 'f1': f1, 'mF1': mF1,
            'kappa': kappa, 'cm': cm}


def print_metrics(res, labels):
    """Pretty-print a :func:`metrics` result dict."""
    n = len(labels)
    print('-' * 64)
    print(f"{'Class':<20} | {'IoU (%)':<10} | {'F1 (%)':<10}")
    print('-' * 64)
    for i in range(n):
        print(f'{labels[i]:<20} | {res["iou"][i] * 100:<10.2f} | {res["f1"][i] * 100:<10.2f}')
    print('-' * 64)
    print(f'{"MEAN":<20} | {res["mIoU"] * 100:<10.2f} | {res["mF1"] * 100:<10.2f}')
    print(f'Overall Accuracy : {res["oa"] * 100:.2f}%')
    print(f'Kappa            : {res["kappa"]:.4f}')
    print('-' * 64)


# =============================================================================
# Full-resolution inference
# =============================================================================
def predict_image(model, image, dsm, cfg, device, batch_size=None, stride=None):
    """Run full-resolution inference for a single tile.

    Args:
        model : a :class:`casdnet.CASDNet` instance (already moved to ``device``).
        image : ``[H, W, 3]`` float array in ``[0, 1]``.
        dsm   : ``[H, W]`` float array normalised exactly as during training.
        cfg   : :class:`config.DatasetConfig`.
        device: ``torch.device``.

    Returns:
        ``[H, W]`` uint8 class map.
    """
    model.eval()
    n_classes = len(cfg.labels)
    window_size = cfg.window_size
    stride = stride if stride is not None else cfg.stride
    batch_size = batch_size if batch_size is not None else cfg.batch_size

    with torch.no_grad():
        # Hunan (index labels) uses whole, pre-tiled images: one forward pass.
        if cfg.label_format == 'index':
            img_t = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
            dsm_t = torch.from_numpy(dsm).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(img_t, dsm_t)[0]
            return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype('uint8')

        # ISPRS: sliding-window accumulation over the (large) ortho tile.
        acc = np.zeros(image.shape[:2] + (n_classes,), dtype='float32')
        windows = sliding_window(image, step=stride, window_size=window_size)
        for coords in grouper(batch_size, windows):
            img_patches = np.asarray(
                [np.copy(image[x:x + w, y:y + h]).transpose(2, 0, 1) for x, y, w, h in coords])
            dsm_patches = np.asarray(
                [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords])

            img_t = torch.from_numpy(img_patches).to(device)
            dsm_t = torch.from_numpy(dsm_patches).unsqueeze(1).to(device)

            logits = model(img_t, dsm_t)[0]                 # [B, C, H, W]
            outs = logits.data.cpu().numpy()
            for out, (x, y, w, h) in zip(outs, coords):
                acc[x:x + w, y:y + h] += out.transpose(1, 2, 0)

        return np.argmax(acc, axis=-1).astype('uint8')
