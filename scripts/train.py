"""
MRTC-Net Training Script
========================
Paper §3 – Experimental Settings

Usage:
    python scripts/train.py --data /path/to/GH-Tomato-MOTC --epochs 100
    python scripts/train.py --data /path/to/GH-Tomato-MOTC --epochs 100 --resume checkpoints/last.pt

Key hyperparameters follow paper experimental settings:
    - Base model  : RT-DETR-L (ResNet50 backbone)
    - Image size  : 640×640
    - Batch size  : 8
    - Optimizer   : AdamW, lr=1e-4, weight_decay=1e-4
    - Scheduler   : Cosine annealing with warmup
    - Loss weights: det_cls=1.0, det_box=5.0, depth=1.0, track=1.0, plant=1.0, geo=0.5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Allow importing from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import MRTCNet
from datasets.gh_tomato_motc import build_dataloader


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, epoch, scaler=None):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        rgb    = batch["image"].to(device)
        depth  = batch["depth"].to(device)

        targets = {
            "boxes"    : batch["boxes"].to(device),
            "classes"  : batch["classes"].to(device),
            "plant_ids": batch["plant_ids"].to(device),
        }

        optimizer.zero_grad()

        if scaler is not None:  # AMP
            with torch.cuda.amp.autocast():
                outputs = model(rgb, depth, prev_state=None)
                losses  = model.compute_loss(outputs, targets)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(rgb, depth, prev_state=None)
            losses  = model.compute_loss(outputs, targets)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

        total_loss += losses["total"].item()

        if batch_idx % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}  [{batch_idx:4d}/{len(loader):4d}]  "
                  f"loss: {losses['total'].item():.4f}  "
                  f"det_cls: {losses.get('det_cls', torch.tensor(0)).item():.3f}  "
                  f"det_box: {losses.get('det_box', torch.tensor(0)).item():.3f}  "
                  f"plant: {losses.get('plant', torch.tensor(0)).item():.3f}  "
                  f"elapsed: {elapsed:.1f}s")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        rgb    = batch["image"].to(device)
        depth  = batch["depth"].to(device)
        targets = {
            "boxes"    : batch["boxes"].to(device),
            "classes"  : batch["classes"].to(device),
            "plant_ids": batch["plant_ids"].to(device),
        }
        outputs = model(rgb, depth, prev_state=None)
        losses  = model.compute_loss(outputs, targets)
        total_loss += losses["total"].item()

    return total_loss / max(len(loader), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader = build_dataloader(args.data, "train", (args.imgsz, args.imgsz),
                                    args.batch, args.workers)
    val_loader   = build_dataloader(args.data, "val",   (args.imgsz, args.imgsz),
                                    args.batch, args.workers, augment=False)

    # ── Model ────────────────────────────────────────────────────────────────
    model = MRTCNet(
        embed_dim=256,
        num_classes=4,
        num_plants=210,
        num_queries=300,
        num_dec_layers=6,
        max_depth_m=0.75,
        pretrained_rtdetr=args.pretrained,
    ).to(device)

    print(f"[Train] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.amp) else None

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"[Train] Resumed from epoch {start_epoch} (best val loss: {best_val:.4f})")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        # Warmup: linearly ramp LR for first 3 epochs
        if epoch < 3:
            warmup_factor = (epoch + 1) / 3
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * warmup_factor

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, scaler)
        val_loss   = validate(model, val_loader, device)
        scheduler.step()

        print(f"\n[Epoch {epoch:3d}]  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}\n")

        # Save checkpoint
        ckpt = {
            "epoch"    : epoch,
            "model"    : model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val" : best_val,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print(f"  ✅ New best model saved  (val_loss: {best_val:.4f})")

    print(f"\n[Train] Done. Best val loss: {best_val:.4f}")
    print(f"[Train] Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRTC-Net Training")
    parser.add_argument("--data",         type=str,   required=True,
                        help="Path to GH-Tomato-MOTC dataset root")
    parser.add_argument("--pretrained",   type=str,   default=None,
                        help="Path to pre-trained RT-DETR-L weights (.pt)")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch",        type=int,   default=8)
    parser.add_argument("--imgsz",        type=int,   default=640)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        dest="weight_decay")
    parser.add_argument("--workers",      type=int,   default=4)
    parser.add_argument("--save-dir",     type=str,   default="checkpoints",
                        dest="save_dir")
    parser.add_argument("--resume",       type=str,   default=None)
    parser.add_argument("--amp",          action="store_true",
                        help="Use automatic mixed precision")
    args = parser.parse_args()
    main(args)
