# CASD-Net

**Context-Aware Statistical Proxy with Dynamic Modality Arbitration** for
multimodal (optical + elevation) remote-sensing semantic segmentation.

This repository is the official training / inference code accompanying the
CASD-Net paper. The module and class names below map **one-to-one** to the
methodology sections of the paper:

| Paper concept                       | Code symbol | Description                                                  |
| ----------------------------------- | ----------- | ------------------------------------------------------------ |
| Overall architecture                | `CASDNet`   | dual-stream pipeline: frozen SAM ViT-B encoder (+ LoRA), 4-level feature pyramid, UNetFormer-style decoder |
| Context-Aware Statistical Proxy     | `CASP`      | maps the mean / std / max statistics of each stream to a per-channel condition $S^s$ |
| Dynamic Modality Arbitration        | `DMA`       | fuses the optical/elevation pair through a dense, channel-wise sigmoid gate $A^s$ |
| Modality Orthogonal Constraint Loss | `MOCLoss`   | penalises the absolute cosine similarity of **corresponding** channels of the deepest features |

Key design points (see `casdnet.py` for the full implementation):

- The SAM ViT-B encoder is **shared and frozen**; only the LoRA adapters
  (parameters whose name contains `lora_`) are trainable, realising the
  parameter-efficient setting reported in the paper (~6.5 M trainable params).
- `CASP` is **level-specific**: one instance per pyramid level, each a two-layer
  MLP (hidden width 128, ReLU) that projects the 6×C statistics
  (`[μ, σ, γ]` of optical ⊕ `[μ, σ, γ]` of elevation) to a channel condition.
- `DMA` performs **channel-wise** modulation: `Z = conv([F_O, F_E]) ⊙ S`,
  `A = σ(Z)`, `F_fused = A⊙F_O + (1−A)⊙F_E`.
- `MOCLoss` uses the **diagonal** of the channel cross-correlation matrix only
  (corresponding channels), not the full matrix.

------

## Directory layout

```
CASD-Net/
├── casdnet.py          # network: CASDNet, CASP, DMA, MOCLoss + decoder blocks
├── config.py           # dataset registry (paths, splits, palettes, hyper-params)
├── utils.py            # data loading, sliding-window inference, metrics
├── train.py            # training script
├── predict.py          # unified inference/evaluation script (all three datasets)
├── MedSAM/             # vendored, minimal SAM encoder (build_sam + ViT blocks)
├── weights/            # place sam_vit_b_01ec64.pth here (see below)
└── requirements.txt
```

------

## Installation

```bash
conda create -n casdnet python=3.10 -y
conda activate casdnet
pip install -r requirements.txt
```

### SAM backbone weights

The encoder is initialised from the SAM ViT-B checkpoint (375 MB). Download it
and place it under `weights/`:

```bash
mkdir -p weights
# download sam_vit_b_01ec64.pth from Meta AI and put it at
#   weights/sam_vit_b_01ec64.pth
```

If the file is missing at build time, `build_sam.py` will offer to download it
interactively. The weights are **not** bundled with this repository.

------

## Data preparation

Edit the `root` field of the dataset you want to use in [`config.py`](config.py).
The file templates follow the canonical ISPRS layout (Vaihingen / Potsdam) and
the Hunan pre-processing layout; only `root` needs to change.

```python
VAIHINGEN = DatasetConfig(name='Vaihingen', root='D:/data/isprs', ...)
POTSDAM   = DatasetConfig(name='Potsdam',   root='D:/data/potsdam', ...)
HUNAN     = DatasetConfig(name='Hunan',     root='D:/data/HN/Hunan_Dataset_cor_process1', ...)
```

Expected layout (Vaihingen example):

```
<root>/Vaihingen/
├── top/top_mosaic_09cm_area{id}.tif
├── dsm/dsm_09cm_matching_area{id}.tif
├── gts_for_participants/top_mosaic_09cm_area{id}.tif
└── gts_eroded_for_participants/top_mosaic_09cm_area{id}_noBoundary.tif
```

> DSM normalisation is handled automatically: ISPRS tiles are min-max scaled
> per tile; Hunan tiles are assumed to be already globally normalised.

------

## Training

```bash
# Vaihingen (defaults follow the paper: AdamW lr=6e-5, cosine, λ=0.1)
python train.py --dataset Vaihingen --exp-name vaihingen_run

# Potsdam
python train.py --dataset Potsdam --exp-name potsdam_run

# Hunan (whole-tile training -> batch size 1)
python train.py --dataset Hunan --exp-name hunan_run

# override hyper-parameters
python train.py --dataset Vaihingen --epochs 200 --lr 1e-4 --moc-weight 0.2
```

The training recipe is:

- **Optimiser** `AdamW(lr=6e-5, weight_decay=1e-2)` with a **cosine** schedule.
- **Total loss** `L = L_seg + λ · L_MOC` with `λ = 0.1`.

Checkpoints are written to `results/` (`{exp_name}_best.pth` and per-epoch
snapshots). Validation runs on the official test split every `--val-freq`
epochs.

------

## Inference / evaluation

```bash
python predict.py --dataset Vaihingen --checkpoint results/vaihingen_run_best.pth --save-color
python predict.py --dataset Potsdam   --checkpoint results/potsdam_run_best.pth --save-color
python predict.py --dataset Hunan     --checkpoint results/hunan_run_best.pth   --save-color
```

The script handles the dataset-specific inference mode automatically
(sliding-window for the ISPRS ortho tiles, whole-tile for Hunan), writes the
colour-coded predictions to `predictions/`, and prints OA / mIoU / per-class
IoU & F1 / kappa.

------

## Citation / acknowledgement

The encoder is a slimmed, vendored copy of
[Segment Anything](https://github.com/facebookresearch/segment-anything)
(Meta AI, Apache-2.0) adapted for the dual-stream, LoRA-finetuned setting used
by CASD-Net. See `MedSAM/LICENSE` for the original notice.
