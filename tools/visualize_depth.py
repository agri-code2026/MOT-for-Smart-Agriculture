"""
Depth Visualization Tools
=========================
Paper §3.3.1 – Depth-guided proximal fruit identification

Visualizes the depth-threshold screening process shown in Fig. 11 of the paper:
    - Blue masks  : distant fruits (depth > threshold)
    - Red/orange/green masks : proximal ripe/semi/unripe fruits within threshold

Usage:
    python tools/visualize_depth.py --rgb 501.png --depth 501-d1.png
    python tools/visualize_depth.py --rgb 501.png --depth 501-d1.png --threshold 0.75
    python tools/visualize_depth.py --rgb 501.png --depth 501-d1.png --progressive --output depth_viz/
"""

import argparse
import os
import sys

import cv2
import numpy as np


def load_depth_raw(path: str) -> np.ndarray:
    """Load 16-bit depth image (Azure Kinect output, uint16, mm)."""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth: {path}")
    return depth.astype(np.float32)


def apply_depth_mask(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    threshold_m: float,
    near_color: tuple = None,
    far_color: tuple = (50, 50, 200),
    fade_far: bool = True,
) -> np.ndarray:
    """
    Apply depth-threshold mask to RGB image.

    Args:
        rgb          : BGR image
        depth_mm     : raw depth in mm
        threshold_m  : near/far boundary in meters
        near_color   : BGR overlay color for near region (None = original colors)
        far_color    : BGR overlay color for far region (default: blue tint)
        fade_far     : if True, fade far pixels (Fig. 11 style in paper)

    Returns:
        result : annotated BGR image
    """
    threshold_mm = threshold_m * 1000
    near_mask = (depth_mm > 0) & (depth_mm <= threshold_mm)
    far_mask  = (depth_mm > threshold_mm)

    result = rgb.copy()

    if fade_far:
        # Fade far region (retain 20% brightness, matches paper Fig. 11)
        faded = (rgb.astype(np.float32) * 0.2).astype(np.uint8)
        result = faded.copy()
        result[near_mask] = rgb[near_mask]  # restore near pixels to original

    # Optional: tint far region with blue mask
    if far_color is not None:
        overlay = result.copy()
        overlay[far_mask] = (
            np.array(far_color, dtype=np.uint8) * 0.5
            + result[far_mask].astype(np.float32) * 0.5
        ).astype(np.uint8)
        result = overlay

    return result


def generate_progressive_masks(
    rgb_path: str,
    depth_path: str,
    min_m: float = 0.0,
    max_m: float = 1.0,
    step_m: float = 0.1,
    output_folder: str = "depth_progressive",
):
    """
    Generate a series of images with progressively increasing depth thresholds.
    Reproduces Fig. 11 in the paper.

    Args:
        rgb_path      : path to RGB image
        depth_path    : path to 16-bit depth image
        min_m / max_m : depth range in meters
        step_m        : threshold increment
        output_folder : save directory
    """
    rgb      = cv2.imread(rgb_path)
    depth_mm = load_depth_raw(depth_path)

    if rgb is None:
        raise FileNotFoundError(f"Cannot read RGB: {rgb_path}")

    os.makedirs(output_folder, exist_ok=True)

    threshold = min_m + step_m
    generated = []
    while threshold <= max_m + 1e-6:
        result = apply_depth_mask(rgb, depth_mm, threshold, fade_far=True)

        # Annotate threshold value on frame
        cv2.putText(result, f"Depth threshold: {threshold:.2f} m",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, f"Depth threshold: {threshold:.2f} m",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)

        fname = os.path.join(output_folder, f"focus_within_{threshold:.2f}m.png")
        cv2.imwrite(fname, result)
        generated.append(fname)
        print(f"  Saved: {fname}")
        threshold += step_m

    print(f"\n  [Done] {len(generated)} images saved to: {output_folder}")
    return generated


def single_threshold_view(
    rgb_path: str,
    depth_path: str,
    threshold_m: float = 0.75,
    show: bool = True,
    save_path: str = None,
):
    """
    Show single depth threshold visualization (paper operating range = 0.75 m).
    """
    rgb      = cv2.imread(rgb_path)
    depth_mm = load_depth_raw(depth_path)

    result = apply_depth_mask(rgb, depth_mm, threshold_m, fade_far=True)

    # Stats
    valid_depth = depth_mm[depth_mm > 0]
    near_pixels = int(((depth_mm > 0) & (depth_mm <= threshold_m * 1000)).sum())
    total_valid = int((depth_mm > 0).sum())

    print(f"\n  Threshold   : {threshold_m:.2f} m")
    print(f"  Near pixels : {near_pixels:,} / {total_valid:,} ({100*near_pixels/max(total_valid,1):.1f}%)")
    if len(valid_depth):
        print(f"  Depth range : {valid_depth.min()/1000:.2f} m – {valid_depth.max()/1000:.2f} m")

    cv2.putText(result, f"Threshold: {threshold_m:.2f} m  |  Near: {near_pixels} px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, result)
        print(f"  Saved: {save_path}")

    if show:
        cv2.imshow(f"Depth threshold: {threshold_m:.2f}m", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Depth Visualization Tool")
    parser.add_argument("--rgb",         type=str, required=True,  help="RGB image path")
    parser.add_argument("--depth",       type=str, required=True,  help="16-bit depth image path")
    parser.add_argument("--threshold",   type=float, default=0.75, help="Depth threshold in meters")
    parser.add_argument("--progressive", action="store_true",
                        help="Generate progressive depth threshold images (Fig. 11 style)")
    parser.add_argument("--min-m",       type=float, default=0.0,  dest="min_m")
    parser.add_argument("--max-m",       type=float, default=1.0,  dest="max_m")
    parser.add_argument("--step-m",      type=float, default=0.1,  dest="step_m")
    parser.add_argument("--output",      type=str, default="depth_progressive",
                        help="Output folder (for --progressive) or save path")
    parser.add_argument("--no-display",  action="store_true", dest="no_display")
    args = parser.parse_args()

    if args.progressive:
        generate_progressive_masks(
            args.rgb, args.depth,
            min_m=args.min_m, max_m=args.max_m, step_m=args.step_m,
            output_folder=args.output,
        )
    else:
        single_threshold_view(
            args.rgb, args.depth,
            threshold_m=args.threshold,
            show=not args.no_display,
            save_path=args.output if args.output.endswith(".png") else None,
        )
