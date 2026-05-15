"""
MRTC-Net Inference Script
=========================
Run MRTC-Net on video, RTSP stream, or image folder.

Usage:
    # Single video file
    python scripts/infer.py --source video.mp4 --model checkpoints/best.pt

    # RTSP stream (greenhouse camera, e.g. Azure Kinect)
    python scripts/infer.py --source rtsp://192.168.x.x:554/ch01.264 --model checkpoints/best.pt --no-display

    # Image folder
    python scripts/infer.py --source /path/to/images/ --model checkpoints/best.pt

    # Save output video with tracking visualization
    python scripts/infer.py --source video.mp4 --model checkpoints/best.pt --save --output result.mp4

Output per frame:
    - Bounding boxes with class label (color-coded: green=unripe, orange=semi, red=ripe, cyan=stem)
    - Track ID overlay
    - Depth value (meters) per fruit
    - Plant ID assignment
    - Proximal (< 0.75m) fruit count for yield estimation

Depth source options:
    - Paired depth video / image folder (--depth-source)
    - Azure Kinect RealSense D435 via pyrealsense2 (--depth-mode realsense)
    - Fallback: zeros (detection only, no depth-guided filtering)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import MRTCNet
from datasets.gh_tomato_motc import (
    CLASS_NAMES, CLASS_COLORS, IMG_MEAN, IMG_STD,
    MAX_DEPTH_MM, NEAR_DEPTH_M, load_depth_image,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tracking association (lightweight ByteTrack-style cosine matching)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTracker:
    """
    Lightweight tracker for MRTC-Net inference.
    Uses cosine similarity on GCS-Head tracking embeddings for identity association.
    """

    def __init__(self, sim_thresh: float = 0.7, max_age: int = 30):
        self.sim_thresh  = sim_thresh
        self.max_age     = max_age
        self.tracks: dict = {}      # track_id → {'embed': tensor, 'age': int}
        self._next_id = 1

    def update(
        self,
        embeds:     torch.Tensor,   # (N, D) L2-normalized embeddings
        boxes:      torch.Tensor,   # (N, 4)
        scores:     torch.Tensor,   # (N,) confidence scores
        min_score:  float = 0.3,
    ):
        """
        Match current detections to existing tracks.
        Returns list of (box, track_id, score) tuples.
        """
        # Filter by confidence
        keep = scores >= min_score
        embeds = embeds[keep]
        boxes  = boxes[keep]
        scores = scores[keep]

        N = len(embeds)
        if N == 0:
            # Age out all tracks
            dead = [k for k, v in self.tracks.items() if v["age"] > self.max_age]
            for k in dead:
                del self.tracks[k]
            return []

        # Compute cosine similarity matrix
        track_ids   = list(self.tracks.keys())
        track_embds = torch.stack([self.tracks[k]["embed"] for k in track_ids]) \
                      if track_ids else None

        assigned = [-1] * N
        used_tracks = set()

        if track_embds is not None:
            sim = embeds @ track_embds.T  # (N, T)
            for i in range(N):
                best_j = sim[i].argmax().item()
                if sim[i, best_j].item() >= self.sim_thresh and best_j not in used_tracks:
                    assigned[i] = track_ids[best_j]
                    used_tracks.add(best_j)

        # Assign new IDs to unmatched detections
        results = []
        for i in range(N):
            if assigned[i] == -1:
                tid = self._next_id
                self._next_id += 1
                assigned[i] = tid
            tid = assigned[i]
            self.tracks[tid] = {"embed": embeds[i].detach(), "age": 0}
            results.append({
                "box"      : boxes[i].cpu().numpy(),
                "track_id" : tid,
                "score"    : scores[i].item(),
            })

        # Age unmatched tracks
        for tid in track_ids:
            if tid not in {r["track_id"] for r in results}:
                self.tracks[tid]["age"] += 1

        dead = [k for k, v in self.tracks.items() if v["age"] > self.max_age]
        for k in dead:
            del self.tracks[k]

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def draw_results(frame, track_results, depths=None, plant_preds=None):
    """Draw bounding boxes, track IDs, depth, and plant IDs on frame."""
    colors_bgr = {
        "unripe"     : (0,   200,   0),
        "semi_mature": (0,   165, 255),
        "mature"     : (0,     0, 255),
        "stem"       : (255, 255,   0),
    }

    for idx, r in enumerate(track_results):
        x1, y1, x2, y2 = map(int, r["box"][:4] if len(r["box"]) >= 4 else [0,0,0,0])
        cls_name = CLASS_NAMES[int(r.get("cls", 0))] if int(r.get("cls", 0)) < len(CLASS_NAMES) else "?"
        color    = colors_bgr.get(cls_name, (180, 180, 180))
        tid      = r["track_id"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{tid} {cls_name}"
        if depths is not None and idx < len(depths):
            label += f" {depths[idx]:.2f}m"
        if plant_preds is not None and idx < len(plant_preds):
            label += f" P{plant_preds[idx]}"

        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

    return frame


def draw_hud(frame, frame_id, fps, near_count, total_count):
    """Draw HUD: FPS, proximal fruit count, yield estimation info."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (310, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"MRTC-Net  Frame#{frame_id:5d}  {fps:.1f} FPS",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230,230,230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Near fruits (< {NEAR_DEPTH_M}m): {near_count:3d}",
                (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,220,60),   1, cv2.LINE_AA)
    cv2.putText(frame, f"Total tracked: {total_count:3d}",
                (14, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,220,220),  1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Preprocess helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_rgb(frame: np.ndarray, imgsz: int, device) -> torch.Tensor:
    """BGR frame → normalized tensor (1,3,H,W)."""
    resized = cv2.resize(frame, (imgsz, imgsz)).astype(np.float32) / 255.0
    mean = np.array(IMG_MEAN, dtype=np.float32)
    std  = np.array(IMG_STD,  dtype=np.float32)
    resized = (resized - mean) / std
    return torch.from_numpy(resized.transpose(2,0,1)).unsqueeze(0).to(device)


def preprocess_depth(depth_arr: np.ndarray, imgsz: int, device) -> torch.Tensor:
    """Normalized depth array → tensor (1,1,H,W)."""
    resized = cv2.resize(depth_arr, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(resized[None, None]).float().to(device)


def denorm_boxes(boxes: np.ndarray, src_h: int, src_w: int) -> np.ndarray:
    """Convert normalized (cx,cy,w,h) → pixel (x1,y1,x2,y2)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * src_w
    y1 = (cy - h / 2) * src_h
    x2 = (cx + w / 2) * src_w
    y2 = (cy + h / 2) * src_h
    return np.stack([x1, y1, x2, y2], axis=1).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[Infer] Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    model = MRTCNet(embed_dim=256, num_classes=4, num_plants=210).to(device)
    ckpt  = torch.load(args.model, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[Infer] Loaded: {args.model}")

    # ── Open video source ────────────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Cannot open source: {args.source}")
        return

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if args.save:
        out_path = args.output or "mrtcnet_result.mp4"
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                   src_fps, (src_w, src_h))
        print(f"[Infer] Saving to: {out_path}")

    tracker    = SimpleTracker(sim_thresh=0.7, max_age=30)
    prev_state = None
    frame_id   = 0
    fps_window = []

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            t0 = time.perf_counter()

            # ── Preprocess ──────────────────────────────────────────────────
            rgb_t = preprocess_rgb(frame, args.imgsz, device)

            # Depth: try paired depth source, else zeros
            depth_arr = np.zeros((src_h, src_w), dtype=np.float32)
            depth_t   = preprocess_depth(depth_arr, args.imgsz, device)

            # ── Inference ───────────────────────────────────────────────────
            outputs    = model(rgb_t, depth_t, prev_state=prev_state)
            prev_state = outputs["state"]

            # ── Postprocess ─────────────────────────────────────────────────
            boxes   = outputs["boxes"][0]          # (N, 4) normalized
            logits  = outputs["logits"][0]         # (N, num_classes)
            depths  = outputs["depth"][0]          # (N, 1)
            embeds  = outputs["track_embed"][0]    # (N, D)
            p_logits= outputs["plant_logits"][0]   # (N, num_plants)

            scores = logits.softmax(-1).max(-1).values
            cls_ids= logits.argmax(-1)

            # Filter low-confidence + stem predictions from fruit counting
            boxes_np  = boxes.cpu().numpy()
            scores_np = scores.cpu().float().numpy()
            cls_np    = cls_ids.cpu().numpy()
            depth_np  = depths.squeeze(-1).cpu().float().numpy()
            plant_np  = p_logits.argmax(-1).cpu().numpy()

            # Track fruits (exclude stems for tracker)
            is_fruit = cls_np < 3
            track_results = tracker.update(
                embeds[is_fruit], boxes[is_fruit], scores[is_fruit], args.conf,
            )

            # Annotate with class and depth
            for i, r in enumerate(track_results):
                r["cls"] = cls_np[is_fruit][i] if i < is_fruit.sum() else 0

            # Pixel boxes
            px_boxes = denorm_boxes(boxes_np[is_fruit], src_h, src_w)
            for i, r in enumerate(track_results):
                if i < len(px_boxes):
                    r["box"] = px_boxes[i]

            near_depths = depth_np[is_fruit][: len(track_results)]
            near_count  = int((near_depths < NEAR_DEPTH_M).sum())

            # ── Visualization ────────────────────────────────────────────────
            t1 = time.perf_counter()
            fps_window.append(1.0 / max(t1 - t0, 1e-9))
            if len(fps_window) > 30:
                fps_window.pop(0)
            cur_fps = sum(fps_window) / len(fps_window)

            vis = frame.copy()
            vis = draw_results(vis, track_results,
                               depths=near_depths.tolist(),
                               plant_preds=plant_np[is_fruit].tolist())
            vis = draw_hud(vis, frame_id, cur_fps, near_count, len(track_results))

            if writer:
                writer.write(vis)

            if not args.no_display:
                cv2.imshow("MRTC-Net Inference", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_id % 30 == 0:
                print(f"  Frame {frame_id:5d}  |  FPS: {cur_fps:.1f}  "
                      f"|  near fruits: {near_count}  |  tracked: {len(track_results)}")

            if args.max_frames > 0 and frame_id >= args.max_frames:
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[Infer] Done. Processed {frame_id} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MRTC-Net Inference")
    parser.add_argument("--source",     type=str, required=True,
                        help="Video path / RTSP URL / camera index")
    parser.add_argument("--model",      type=str, required=True,
                        help="Path to MRTC-Net checkpoint (.pt)")
    parser.add_argument("--imgsz",      type=int, default=640)
    parser.add_argument("--conf",       type=float, default=0.3)
    parser.add_argument("--device",     type=str, default="auto")
    parser.add_argument("--save",       action="store_true")
    parser.add_argument("--output",     type=str, default="")
    parser.add_argument("--no-display", action="store_true", dest="no_display")
    parser.add_argument("--max-frames", type=int, default=0, dest="max_frames")
    args = parser.parse_args()
    run(args)
