# -*- coding: utf-8 -*-
"""
Unified inference / evaluation script for CASD-Net.

One script covers all three datasets:

    Vaihingen / Potsdam : large ortho tiles, sliding-window inference
    Hunan               : pre-tiled images, whole-tile inference

Usage:
    python predict.py --dataset Vaihingen --checkpoint results/my_run_best.pth
    python predict.py --dataset Hunan     --checkpoint results/hunan_best.pth

For each test tile the colour-coded prediction is written to ``--output-dir``
and the aggregate metrics (OA / mIoU / per-class IoU & F1 / kappa) are printed.
"""

import argparse
import os

import numpy as np
import torch
from skimage import io
from tqdm import tqdm

import config
from casdnet import CASDNet
from utils import (convert_to_color, metrics, predict_image, print_metrics,
                   read_dsm, read_image, read_label)


def parse_args():
    parser = argparse.ArgumentParser(description='Predict with CASD-Net')
    parser.add_argument('--dataset', default='Vaihingen',
                        choices=list(config.DATASETS.keys()))
    parser.add_argument('--checkpoint', required=True,
                        help='path to a .pth produced by train.py')
    parser.add_argument('--output-dir', default='predictions')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--save-color', action='store_true',
                        help='write colour-coded prediction PNGs')
    parser.add_argument('--sam-checkpoint', default='weights/sam_vit_b_01ec64.pth')
    return parser.parse_args()


def load_model(cfg, args, device):
    net = CASDNet(num_classes=len(cfg.labels),
                  sam_checkpoint=args.sam_checkpoint).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('model', ckpt)
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    return net


def main():
    args = parse_args()
    cfg = config.DATASETS[args.dataset]
    device = torch.device(args.device)

    if not cfg.root:
        raise SystemExit(
            f'Please set ``root`` for dataset "{cfg.name}" in config.py before predicting.')

    net = load_model(cfg, args, device)
    os.makedirs(args.output_dir, exist_ok=True)

    preds, gts = [], []
    for id_ in tqdm(cfg.test_ids, desc=f'Predicting {cfg.name}'):
        img = read_image(cfg.data_template.format(id=id_)).transpose(1, 2, 0)  # [H, W, 3]
        dsm = read_dsm(cfg.dsm_template.format(id=id_), cfg.dsm_normalization)
        gt = read_label(cfg.eroded_template.format(id=id_), cfg.label_format, cfg.palette)

        pred = predict_image(net, img, dsm, cfg, device,
                             batch_size=args.batch_size, stride=args.stride)

        if args.save_color:
            color = convert_to_color(pred, cfg.palette)
            io.imsave(os.path.join(args.output_dir, f'{cfg.name}_{id_}_pred.png'), color)

        preds.append(pred.ravel())
        gts.append(gt.ravel())

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    valid = (gts >= 0) & (gts < len(cfg.labels))

    res = metrics(preds[valid], gts[valid], cfg.labels)
    print(f'\n===== {cfg.name} results =====')
    print_metrics(res, cfg.labels)


if __name__ == '__main__':
    main()
