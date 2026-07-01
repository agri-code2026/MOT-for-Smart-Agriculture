"""
GH-Tomato-MOTC Dataset Utilities
==================================
Paper §2.1 – Data Preparation

Dataset statistics (Table 1 in paper):
| Split | Images | BBoxes | Tracks | Plant IDs | Avg/frame |
|-------|--------|--------|--------|-----------|-----------|
| Train | 14400 | 173840 | 5889 | 167 | 12.1 |
| Val | 1800 | 23940 | 752 | 22 | 13.3 |
| Test | 1800 | 21420 | 726 | 21 | 11.9 |
Label format (YOLO-style extended):
    class_id  cx  cy  w  h  track_id  plant_id
    (all values normalized to [0,1] except integer IDs)

RGB-D alignment:
    Depth images are pre-aligned to the RGB camera coordinate using
    Azure Kinect's pyk4a.transformation.depth_image_to_color_camera().
    Depth pixel values are in millimeters (uint16); loaded with
    cv2.IMREAD_UNCHANGED to preserve precision.

Depth normalization:
    depth_normalized = depth_mm / max_depth_mm
    Invalid pixels (depth == 0) are treated as 0 (ignored by DGA).
"""

import os
import glob
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["unripe", "semi_mature", "mature", "stem"]
CLASS_COLORS = {
    "unripe"     : (0,   200,  0),   # green
    "semi_mature": (0,   165, 255),  # orange
    "mature"     : (0,    0,  255),  # red
    "stem"       : (255, 255,  0),   # cyan
}

MAX_DEPTH_MM   = 3000   # 3 meters — clip depth values beyond this
NEAR_DEPTH_M   = 0.75   # effective operating radius for harvesting manipulators
IMG_MEAN       = (0.485, 0.456, 0.406)  # ImageNet mean
IMG_STD        = (0.229, 0.224, 0.225)  # ImageNet std


# ─────────────────────────────────────────────────────────────────────────────
# Depth utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_depth_image(path: str, max_depth_mm: int = MAX_DEPTH_MM) -> np.ndarray:
    """
    Load a 16-bit depth image and normalize to [0, 1].

    Azure Kinect depth maps: uint16, unit = mm.
    Must load with cv2.IMREAD_UNCHANGED, otherwise OpenCV forces 8-bit.

    Args:
        path         : path to depth image (.png)
        max_depth_mm : depth values above this are clipped to 0 (invalid)

    Returns:
        depth_norm   : (H, W) float32, values in [0, 1]
                       0 = invalid / missing depth
    """
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # (H, W) uint16
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")

    depth = depth.astype(np.float32)
    # Invalid pixels (raw value = 0) remain 0 after normalization
    mask  = (depth > 0) & (depth <= max_depth_mm)
    depth_norm = np.zeros_like(depth)
    depth_norm[mask] = depth[mask] / max_depth_mm
    return depth_norm


def depth_to_meters(depth_norm: np.ndarray, max_depth_mm: int = MAX_DEPTH_MM) -> np.ndarray:
    """Convert normalized depth [0,1] back to meters."""
    return depth_norm * (max_depth_mm / 1000.0)


# ─────────────────────────────────────────────────────────────────────────────
# Label parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_label_file(label_path: str) -> List[Dict]:
    """
    Parse extended YOLO-format label file.

    Expected format per line:
        class_id  cx  cy  w  h  track_id  plant_id

    Returns:
        list of dicts with keys: 'cls', 'cx', 'cy', 'w', 'h', 'track_id', 'plant_id'
    """
    annotations = []
    if not os.path.exists(label_path):
        return annotations

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            ann = {
                "cls"      : int(parts[0]),
                "cx"       : float(parts[1]),
                "cy"       : float(parts[2]),
                "w"        : float(parts[3]),
                "h"        : float(parts[4]),
                "track_id" : int(parts[5]) if len(parts) > 5 else -1,
                "plant_id" : int(parts[6]) if len(parts) > 6 else -1,
            }
            annotations.append(ann)
    return annotations


# ─────────────────────────────────────────────────────────────────────────────
# Data augmentation
# ─────────────────────────────────────────────────────────────────────────────

def random_horizontal_flip(
    rgb: np.ndarray,
    depth: np.ndarray,
    boxes: np.ndarray,
    p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontally flip RGB, depth, and bounding boxes (in-place on boxes)."""
    if random.random() < p:
        rgb   = cv2.flip(rgb,   1)
        depth = cv2.flip(depth, 1)
        if len(boxes):
            boxes[:, 1] = 1.0 - boxes[:, 1]  # flip cx
    return rgb, depth, boxes


def random_color_jitter(
    rgb: np.ndarray,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
) -> np.ndarray:
    """Apply random brightness / contrast / saturation jitter to simulate illumination variation."""
    # Convert to HSV for saturation control
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= 1 + random.uniform(-brightness, brightness)   # value
    hsv[:, :, 1] *= 1 + random.uniform(-saturation, saturation)   # saturation
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Contrast
    alpha = 1 + random.uniform(-contrast, contrast)
    rgb   = np.clip(alpha * rgb.astype(np.float32), 0, 255).astype(np.uint8)
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class GHTomatoMOTCDataset(Dataset):
    """
    GH-Tomato-MOTC Dataset loader for MRTC-Net training.

    Expected directory structure:
        root/
          images/
            train/  *.jpg or *.png
            val/
            test/
          depths/
            train/  *-d1.png (16-bit, aligned to RGB)
            val/
            test/
          labels/
            train/  *.txt
            val/
            test/

    Args:
        root        : dataset root directory
        split       : 'train' / 'val' / 'test'
        img_size    : (H, W) to resize images for model input
        augment     : apply data augmentation (train only)
        max_depth_mm: depth clip value
    """

    def __init__(
        self,
        root:         str,
        split:        str = "train",
        img_size:     Tuple[int, int] = (640, 640),
        augment:      bool = True,
        max_depth_mm: int  = MAX_DEPTH_MM,
    ):
        super().__init__()
        self.split        = split
        self.img_size     = img_size
        self.augment      = augment and (split == "train")
        self.max_depth_mm = max_depth_mm

        img_dir   = os.path.join(root, "images",  split)
        depth_dir = os.path.join(root, "depths",  split)
        label_dir = os.path.join(root, "labels",  split)

        # Collect image files
        self.samples = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in sorted(glob.glob(os.path.join(img_dir, ext))):
                stem       = os.path.splitext(os.path.basename(img_path))[0]
                depth_path = os.path.join(depth_dir, stem + "-d1.png")
                label_path = os.path.join(label_dir, stem + ".txt")
                self.samples.append({
                    "image" : img_path,
                    "depth" : depth_path,
                    "label" : label_path,
                })

        if not self.samples:
            raise RuntimeError(f"No images found in {img_dir}")

        print(f"[Dataset] {split}: {len(self.samples)} samples loaded from {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # ── Load RGB ────────────────────────────────────────────────────────
        rgb = cv2.imread(sample["image"])
        if rgb is None:
            raise FileNotFoundError(f"Cannot read image: {sample['image']}")

        # ── Load depth ──────────────────────────────────────────────────────
        if os.path.exists(sample["depth"]):
            depth_norm = load_depth_image(sample["depth"], self.max_depth_mm)
        else:
            # Missing depth → fill with zeros (DGA handles zeros as invalid)
            depth_norm = np.zeros(rgb.shape[:2], dtype=np.float32)

        # ── Load labels ─────────────────────────────────────────────────────
        anns = parse_label_file(sample["label"])

        # Build box array: [cls, cx, cy, w, h, track_id, plant_id]
        if anns:
            boxes = np.array([[
                a["cls"], a["cx"], a["cy"], a["w"], a["h"],
                a["track_id"], a["plant_id"],
            ] for a in anns], dtype=np.float32)
        else:
            boxes = np.zeros((0, 7), dtype=np.float32)

        # ── Augmentation ────────────────────────────────────────────────────
        if self.augment:
            if len(boxes):
                box_arr = boxes[:, 1:5]  # cx, cy, w, h
            else:
                box_arr = np.zeros((0, 4), dtype=np.float32)

            rgb, depth_norm, box_arr = random_horizontal_flip(rgb, depth_norm, box_arr)
            rgb = random_color_jitter(rgb)

            if len(boxes):
                boxes[:, 1:5] = box_arr

        # ── Resize ──────────────────────────────────────────────────────────
        H, W = self.img_size
        rgb   = cv2.resize(rgb,   (W, H)).astype(np.float32) / 255.0
        depth_resized = cv2.resize(depth_norm, (W, H), interpolation=cv2.INTER_NEAREST)

        # ── Normalize RGB ────────────────────────────────────────────────────
        mean = np.array(IMG_MEAN, dtype=np.float32)
        std  = np.array(IMG_STD,  dtype=np.float32)
        rgb  = (rgb - mean) / std

        # ── Convert to tensors ───────────────────────────────────────────────
        rgb_t   = torch.from_numpy(rgb.transpose(2, 0, 1)).float()          # (3, H, W)
        depth_t = torch.from_numpy(depth_resized[None]).float()             # (1, H, W)

        # Target tensors
        gt_boxes    = torch.from_numpy(boxes[:, 1:5]).float()  if len(boxes) else torch.zeros(0, 4)
        gt_classes  = torch.from_numpy(boxes[:, 0]).long()     if len(boxes) else torch.zeros(0, dtype=torch.long)
        gt_track_ids= torch.from_numpy(boxes[:, 5]).long()     if len(boxes) else torch.zeros(0, dtype=torch.long)
        gt_plant_ids= torch.from_numpy(boxes[:, 6]).long()     if len(boxes) else torch.zeros(0, dtype=torch.long)

        return {
            "image"     : rgb_t,
            "depth"     : depth_t,
            "boxes"     : gt_boxes,
            "classes"   : gt_classes,
            "track_ids" : gt_track_ids,
            "plant_ids" : gt_plant_ids,
            "img_path"  : sample["image"],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate: pad variable-length annotation lists."""
    images    = torch.stack([b["image"]  for b in batch])
    depths    = torch.stack([b["depth"]  for b in batch])

    max_obj = max(b["boxes"].size(0) for b in batch)

    def pad(tensors, pad_val=0.0):
        result = []
        for t in tensors:
            n = t.size(0)
            if n < max_obj:
                pad_t = torch.full((max_obj - n, *t.shape[1:]),
                                   pad_val, dtype=t.dtype)
                t = torch.cat([t, pad_t], dim=0)
            result.append(t)
        return torch.stack(result)

    return {
        "image"     : images,
        "depth"     : depths,
        "boxes"     : pad([ b["boxes"]      for b in batch], 0.0),
        "classes"   : pad([ b["classes"]    for b in batch], -1).long(),
        "track_ids" : pad([ b["track_ids"]  for b in batch], -1).long(),
        "plant_ids" : pad([ b["plant_ids"]  for b in batch], -1).long(),
        "img_paths" : [ b["img_path"] for b in batch],
    }


def build_dataloader(
    root:       str,
    split:      str = "train",
    img_size:   Tuple[int, int] = (640, 640),
    batch_size: int  = 8,
    num_workers: int = 4,
    augment:    bool = True,
) -> DataLoader:
    """Convenience function to build a dataloader for any split."""
    dataset = GHTomatoMOTCDataset(root, split, img_size, augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )
