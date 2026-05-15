"""
FPS Benchmark for MRTC-Net (and baselines)
==========================================
Reproduces the FPS measurements reported in the paper (end-to-end pipeline):

    Platform             FPS (paper)
    RTX 3060             46.7
    Jetson AGX Orin      31.5
    Jetson Orin NX       24.9

Usage:
    # MRTC-Net
    python tools/fps_benchmark.py --model checkpoints/best.pt --source test.mp4

    # Baseline comparison: YOLO + ByteTrack
    python tools/fps_benchmark.py --baseline yolo --model best_yolo.pt --source test.mp4

    # RT-DETR + ByteTrack
    python tools/fps_benchmark.py --baseline rtdetr --model bestRT.pt --source test.mp4

    # No display (server / headless)
    python tools/fps_benchmark.py --model checkpoints/best.pt --source test.mp4 --no-display

Metrics reported:
    - End-to-end FPS (avg / min / max)
    - Per-stage latency (preproc / inference / tracking)
    - P50 / P95 / P99 latency percentiles
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def percentile(data, pct):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100
    f = int(k)
    return s[f] + (k - f) * (s[min(f+1, len(s)-1)] - s[f])


def run_mrtcnet(args, device):
    from models import MRTCNet
    model = MRTCNet(embed_dim=256, num_classes=4, num_plants=210).to(device)
    ckpt  = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()

    def tick():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Cannot open: {args.source}")
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n  Source: {args.source}  |  {src_w}x{src_h}  |  {total} frames")

    times_pre, times_inf, times_total = [], [], []
    window = deque(maxlen=30)
    frame_id = 0
    prev_state = None

    print(f"\n  [Warmup: {args.warmup} frames]\n")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            # Preprocess
            t1 = tick()
            rgb   = cv2.resize(frame, (args.imgsz, args.imgsz)).astype(np.float32) / 255.0
            rgb_t = torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).to(device)
            dep_t = torch.zeros(1, 1, args.imgsz, args.imgsz, device=device)
            t2 = tick()

            # Inference + tracking (MRTC-Net is end-to-end, no separate tracker)
            outputs    = model(rgb_t, dep_t, prev_state=prev_state)
            prev_state = outputs["state"]
            t3 = tick()

            if frame_id <= args.warmup:
                continue

            pre_ms   = (t2 - t1) * 1000
            total_ms = (t3 - t1) * 1000
            times_pre.append(pre_ms)
            times_inf.append(total_ms - pre_ms)
            times_total.append(total_ms)
            window.append(total_ms)

            if frame_id % 30 == 0:
                cur_fps = 1000 / (sum(window) / len(window)) if window else 0
                avg_fps = 1000 / (sum(times_total) / len(times_total))
                print(f"  Frame {frame_id:5d}  |  cur FPS: {cur_fps:5.1f}  "
                      f"|  avg FPS: {avg_fps:5.1f}  |  infer: {times_inf[-1]:.1f}ms")

            if args.max_frames > 0 and frame_id >= args.max_frames + args.warmup:
                break

    cap.release()
    return times_pre, times_inf, times_total


def run_baseline(args, device, mode="yolo"):
    """Run YOLO or RT-DETR + ByteTrack baseline for comparison."""
    if mode == "rtdetr":
        from ultralytics import RTDETR
        model = RTDETR(args.model)
    else:
        from ultralytics import YOLO
        model = YOLO(args.model)
    model.to(device.type)

    def tick():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Cannot open: {args.source}")
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n  Source: {args.source}  |  {src_w}x{src_h}  |  {total} frames")

    times_pre, times_inf, times_total = [], [], []
    window = deque(maxlen=30)
    frame_id = 0

    print(f"\n  [Warmup: {args.warmup} frames]\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        t1 = tick()
        resized = cv2.resize(frame, (args.imgsz, args.imgsz))
        t2 = tick()

        with torch.no_grad():
            results = model.track(resized, persist=True, conf=0.3,
                                  tracker="bytetrack.yaml", verbose=False)
        t3 = tick()

        if frame_id <= args.warmup:
            continue

        pre_ms   = (t2 - t1) * 1000
        total_ms = (t3 - t1) * 1000
        times_pre.append(pre_ms)
        times_inf.append(total_ms - pre_ms)
        times_total.append(total_ms)
        window.append(total_ms)

        if frame_id % 30 == 0:
            cur_fps = 1000 / (sum(window) / len(window)) if window else 0
            avg_fps = 1000 / (sum(times_total) / len(times_total))
            print(f"  Frame {frame_id:5d}  |  cur FPS: {cur_fps:5.1f}  |  avg FPS: {avg_fps:5.1f}")

        if args.max_frames > 0 and frame_id >= args.max_frames + args.warmup:
            break

    cap.release()
    return times_pre, times_inf, times_total


def print_report(times_pre, times_inf, times_total, model_name, device_name):
    n = len(times_total)
    if n == 0:
        print("[Report] No valid frames.")
        return

    def s(lst):
        avg = sum(lst) / len(lst)
        return avg, min(lst), max(lst), percentile(lst, 50), percentile(lst, 95), percentile(lst, 99)

    pre_s = s(times_pre)
    inf_s = s(times_inf)
    tot_s = s(times_total)
    fps_avg = 1000 / tot_s[0]
    fps_min = 1000 / tot_s[2]
    fps_max = 1000 / tot_s[1]
    pct_pre = pre_s[0] / tot_s[0] * 100
    pct_inf = inf_s[0] / tot_s[0] * 100

    W = 68
    print(f"\n{'='*W}")
    print(f"  FPS Benchmark Report  |  Model: {model_name}  |  Device: {device_name}")
    print(f"  Valid frames: {n}")
    print(f"{'='*W}")
    print(f"  {'Metric':<22}  {'avg':>8}  {'min':>8}  {'max':>8}  {'P50':>8}  {'P95':>8}  {'P99':>8}")
    print(f"  {'-'*64}")

    def row(name, st, unit="ms"):
        print(f"  {name:<22}  {st[0]:>7.2f}{unit}  {st[1]:>7.2f}{unit}  "
              f"{st[2]:>7.2f}{unit}  {st[3]:>7.2f}{unit}  {st[4]:>7.2f}{unit}  {st[5]:>7.2f}{unit}")

    row("Preprocess",  pre_s)
    row("Inference",   inf_s)
    row("End-to-end",  tot_s)
    print(f"  {'-'*64}")
    print(f"  {'FPS (end-to-end)':<22}  {fps_avg:>7.1f}fps  {fps_min:>7.1f}fps  {fps_max:>7.1f}fps")
    print(f"\n  Stage breakdown — preproc: {pct_pre:.1f}%  |  inference+track: {pct_inf:.1f}%")
    print(f"\n  ✅ Average FPS : {fps_avg:.2f}")
    print(f"  ✅ P95 latency : {tot_s[4]:.2f} ms")
    if fps_avg >= 30:
        print(f"  🟢 Real-time   : PASS (≥ 30 FPS)")
    elif fps_avg >= 20:
        print(f"  🟡 Real-time   : Marginal (20–30 FPS)")
    else:
        print(f"  🔴 Real-time   : FAIL (< 20 FPS)")
    print(f"{'='*W}\n")


def main(args):
    device_str = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    device = torch.device(device_str)

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n  Device : {gpu_name}")
    print(f"  Model  : {args.model}")
    print(f"  Mode   : {args.baseline or 'mrtcnet'}")
    print(f"  ImgSz  : {args.imgsz}x{args.imgsz}")
    print(f"  Warmup : {args.warmup} frames")

    if args.baseline in ("yolo", "rtdetr"):
        t_pre, t_inf, t_total = run_baseline(args, device, mode=args.baseline)
    else:
        t_pre, t_inf, t_total = run_mrtcnet(args, device)

    model_label = os.path.basename(args.model)
    print_report(t_pre, t_inf, t_total, model_label, gpu_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MRTC-Net / Baseline FPS Benchmark")
    parser.add_argument("--source",     type=str, required=True)
    parser.add_argument("--model",      type=str, required=True)
    parser.add_argument("--baseline",   type=str, default=None,
                        choices=[None, "yolo", "rtdetr"],
                        help="Run a baseline model instead of MRTC-Net")
    parser.add_argument("--imgsz",      type=int, default=640)
    parser.add_argument("--device",     type=str, default="auto")
    parser.add_argument("--warmup",     type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0, dest="max_frames")
    parser.add_argument("--no-display", action="store_true", dest="no_display")
    args = parser.parse_args()
    main(args)
