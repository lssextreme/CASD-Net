# -*- coding: utf-8 -*-
"""
Training script for CASD-Net.

Usage:
    python train.py --dataset Vaihingen --exp-name my_run
    python train.py --dataset Hunan   --epochs 200 --batch-size 1

The training recipe follows the CASD-Net paper:
    - optimizer : AdamW (lr = 6e-5, cosine decay over ``epochs``)
    - total loss: L = L_seg + lambda * L_MOC, with lambda = 0.1
    - encoder  : frozen SAM ViT-B; only the LoRA adapters are trainable

Set the dataset ``root`` in ``config.py`` before running.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import config
from casdnet import CASDNet, MOCLoss
from utils import (ISPRS_dataset, metrics, print_metrics, predict_image,
                   read_dsm, read_image, read_label, seg_loss)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CASD-Net')
    parser.add_argument('--dataset', default='Vaihingen',
                        choices=list(config.DATASETS.keys()))
    parser.add_argument('--exp-name', default='casdnet', type=str)
    parser.add_argument('--epochs', type=int, default=None,
                        help='override the config epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='override the config batch size')
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--moc-weight', type=float, default=0.1,
                        help='lambda weighting the Modality Orthogonal '
                             'Constraint loss (paper: 0.1)')
    parser.add_argument('--val-freq', type=int, default=10,
                        help='run validation every N epochs')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--resume', default=None, help='checkpoint path to resume')
    parser.add_argument('--sam-checkpoint', default='weights/sam_vit_b_01ec64.pth')
    return parser.parse_args()


def build_model(cfg, args, device):
    net = CASDNet(num_classes=len(cfg.labels),
                  sam_checkpoint=args.sam_checkpoint).to(device)
    # Only LoRA adapters (and the FPN / decoder / CASP / DMA heads) are
    # trainable; the frozen SAM parameters carry requires_grad == False.
    trainable = [p for p in net.parameters() if p.requires_grad]
    frozen = [p for p in net.parameters() if not p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in frozen)
    print(f'[model] trainable params: {n_train:,}  |  frozen params: {n_frozen:,}')
    return net


def evaluate(net, cfg, device):
    """Sliding-window / whole-tile evaluation over the official test split."""
    net.eval()
    preds, gts = [], []
    for id_ in tqdm(cfg.test_ids, desc='Evaluating', leave=False):
        img = read_image(cfg.data_template.format(id=id_)).transpose(1, 2, 0)
        dsm = read_dsm(cfg.dsm_template.format(id=id_), cfg.dsm_normalization)
        gt = read_label(cfg.eroded_template.format(id=id_), cfg.label_format, cfg.palette)

        pred = predict_image(net, img, dsm, cfg, device)
        preds.append(pred.ravel())
        gts.append(gt.ravel())

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    valid = (gts >= 0) & (gts < len(cfg.labels))
    return metrics(preds[valid], gts[valid], cfg.labels)


def main():
    args = parse_args()
    cfg = config.DATASETS[args.dataset]
    device = torch.device(args.device)

    epochs = args.epochs or cfg.epochs
    batch_size = args.batch_size or cfg.batch_size
    if not cfg.root:
        raise SystemExit(
            f'Please set ``root`` for dataset "{cfg.name}" in config.py before training.')

    # --- data ---
    train_set = ISPRS_dataset(cfg, cfg.train_ids, cache=True)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers)

    # --- model / loss / optimizer ---
    net = build_model(cfg, args, device)
    moc_criterion = MOCLoss().to(device)
    seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    trainable = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 1
    best_miou = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt.get('model', ckpt), strict=False)
        start_epoch = ckpt.get('epoch', 1) + 1
        print(f'[resume] loaded {args.resume}, restarting at epoch {start_epoch}')

    os.makedirs(args.output_dir, exist_ok=True)

    # --- training loop ---
    for epoch in range(start_epoch, epochs + 1):
        net.train()
        running_seg = running_moc = 0.0
        start = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False)
        for data, dsm, target in pbar:
            data, dsm, target = data.to(device), dsm.to(device), target.to(device)

            optimizer.zero_grad()

            # CASDNet.forward returns (logits, f_optical, f_elevation).
            logits, f_optical, f_elevation = net(data, dsm)

            loss_seg = seg_criterion(logits, target)
            loss_moc = moc_criterion(f_optical, f_elevation)
            loss = loss_seg + args.moc_weight * loss_moc

            loss.backward()
            optimizer.step()

            running_seg += loss_seg.item()
            running_moc += loss_moc.item()
            pbar.set_postfix(seg=f'{loss_seg.item():.4f}',
                             moc=f'{loss_moc.item():.4f}',
                             total=f'{loss.item():.4f}')

        scheduler.step()

        n = len(train_loader)
        print(f'Epoch {epoch}/{epochs} | seg={running_seg / n:.4f} '
              f'moc={running_moc / n:.4f} | {time.time() - start:.1f}s')

        # --- validation + checkpointing ---
        if epoch % args.val_freq == 0 or epoch == epochs:
            res = evaluate(net, cfg, device)
            print_metrics(res, cfg.labels)
            miou = res['mIoU']
            if miou > best_miou:
                best_miou = miou
                torch.save({'epoch': epoch, 'model': net.state_dict(),
                            'mIoU': miou, 'config': cfg.name},
                           os.path.join(args.output_dir, f'{args.exp_name}_best.pth'))
                print(f'  -> new best mIoU {miou:.4f}, checkpoint saved')

        if epoch % cfg.save_epoch == 0:
            torch.save({'epoch': epoch, 'model': net.state_dict(),
                        'config': cfg.name},
                       os.path.join(args.output_dir,
                                    f'{args.exp_name}_epoch{epoch}.pth'))

    print(f'Training finished. Best mIoU: {best_miou:.4f}')


if __name__ == '__main__':
    main()
