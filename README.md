# MOT-for-Smart-Agriculture

**MRTC-Net: A Multi-modal Real-time Tracking and Counting Network for Tomato Yield Estimation in Greenhouse**

> This repository contains the official code for the paper:  
> *"MRTC-Net: A Multi-modal Real-time Tracking and Counting Network for Tomato Yield Estimation in Greenhouse"*

---

## Overview

MRTC-Net is an end-to-end multimodal framework for **real-time tomato tracking, fruit–plant association, and plant-level yield estimation** under complex greenhouse environments. It builds upon RT-DETR-L and introduces three novel modules:

| Module | Role |
|--------|------|
| **DGM-Encoder** | Depth-Gated Multimodal Encoder — suppresses far-field noise via DGA; illumination-robust fusion via M-AIFI |
| **UCL-Decoder** | Uncertainty-aware Closed-Loop Decoder — maintains trajectory continuity under dense occlusion via dual query sets and closed-loop feedback |
| **GCS-Head** | Geometry-Constrained Structured Head — simultaneous detection, depth regression, tracking embedding, and plant identification |

### Key Results (GH-Tomato-MOTC Dataset)

| Metric | MRTC-Net | Best Baseline |
|--------|----------|---------------|
| mAP@0.5 | **98.1%** | 97.1% |
| HOTA | **65.7%** | 64.2% |
| IDF1 | **86.3%** | 74.8% (↑15%) |
| IDS ↓ | **3** | 9 |
| MAE ↓ | **3.1** | 4.3 (↓28%) |
| RMSE ↓ | **4.91** | 6.4 (↓29%) |

**Real-time performance:**

| Platform | FPS |
|----------|-----|
| NVIDIA RTX 3060 | 46.7 |
| Jetson AGX Orin | 31.5 |
| Jetson Orin NX | 28.7 |

---

## Repository Structure

```
MOT-for-Smart-Agriculture/
├── models/
│   ├── __init__.py
│   ├── mrtcnet.py          # Full MRTC-Net model
│   ├── dgm_encoder.py      # DGM-Encoder (DGA + M-AIFI)
│   ├── ucl_decoder.py      # UCL-Decoder (dual query set + GRU feedback)
│   └── gcs_head.py         # GCS-Head (multi-task + geometric constraint)
├── datasets/
│   └── gh_tomato_motc.py   # GH-Tomato-MOTC dataset loader
├── scripts/
│   ├── train.py            # Training script
│   ├── infer.py            # Inference on video / RTSP / images
│   └── eval.py             # Evaluation (mAP / HOTA / IDF1 / MAE / RMSE)
├── tools/
│   ├── fps_benchmark.py    # FPS benchmark (MRTC-Net + baselines)
│   └── visualize_depth.py  # Depth threshold visualization (paper Fig. 11)
├── configs/
│   └── mrtcnet.yaml        # Model & training hyperparameters
├── distance.py             # Azure Kinect depth preprocessing utility
├── read.py                 # RGB-D image sequence viewer
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/xuanjingwei-ahau/MOT-for-Smart-Agriculture.git
cd MOT-for-Smart-Agriculture
pip install -r requirements.txt
```

Requires Python ≥ 3.8 and PyTorch ≥ 1.13.

---

## Dataset — GH-Tomato-MOTC

| Split | Images | BBoxes | Tracks | Plant IDs | Avg/frame |
|-------|--------|--------|--------|-----------|-----------|
| Train | 12,600 | 151,200 | 5,000 | 150 | 12.0 |
| Val | 2,700 | 35,825 | 1,210 | 30 | 13.3 |
| Test | 2,700 | 32,175 | 1,157 | 30 | 11.9 |
| **Total** | **18,000** | **219,200** | **7,367** | **210** | **12.2** |

**Label format** (extended YOLO, `.txt`):
```
class_id  cx  cy  w  h  track_id  plant_id
```
All values normalized to [0,1] except integer IDs.

**Depth images**: 16-bit PNG, unit = mm (Azure Kinect DK output).  
Pre-aligned to RGB camera via `pyk4a.transformation.depth_image_to_color_camera()`.

### RGB-D data fusion for tracking counting and yield estimation of single fruit tomatoes



#### 1.Deep distance output



Please refer to `distance.py`.

16-bit depth data: The depth map output from Azure Kinect is usually in uint16 format, with the unit in millimeters (mm). When using cv2.imread, you must add cv2.IMREAD_UNCHANGED, otherwise OpenCV will force it into 8-bit, causing precision loss and incorrect values.

Alignment precondition:

If your rgb.png is the original 4K/1080P image and depth.png is the original 512x512 image, the above code will give an error.

You must use pyk4a.transformation.depth_image_to_color_camera() to transform the depth map into the RGB perspective at the frame extraction stage.

Handling the "black hole" (invalid depth):

The areas in the depth map where the value is 0 are usually invalid regions caused by insufficient infrared reflection or occlusion. When setting min_dist, it is generally recommended to set it to a value greater than 0 (e.g., min_dist=1), which automatically filters out these noise points.

#### 2.Dynamically demonstrate the change of depth information

Please refer to `distance.py`.

```html
Quickly train the model by referencing the data in the link: https://pan.baidu.com/s/1bRHnlXS6u-MmMJl_1AmZwA 提取码: 36a3
Alternatively, you can obtain resources from our published datasets：https://www.scidb.cn/detail?dataSetId=c303b6269e3e43f087bec4e87735a42e
If you are interested in the methods we used in processing multimodal datasets, please contact us via email.
```



---

## Training

```bash
python scripts/train.py \
    --data /path/to/GH-Tomato-MOTC \
    --pretrained /path/to/rtdetr-l.pt \
    --epochs 100 \
    --batch 8 \
    --amp
```

Resume from checkpoint:
```bash
python scripts/train.py --data /path/to/GH-Tomato-MOTC --resume checkpoints/last.pt
```

---

## Inference

```bash
# Video file
python scripts/infer.py --source video.mp4 --model checkpoints/best.pt

# RTSP stream (greenhouse camera)
python scripts/infer.py \
    --source rtsp://192.168.x.x:554/ch01.264 \
    --model checkpoints/best.pt \
    --no-display --save --output result.mp4
```

---

## Evaluation

```bash
python scripts/eval.py \
    --data /path/to/GH-Tomato-MOTC \
    --model checkpoints/best.pt \
    --split test
```

---

## FPS Benchmark

```bash
# MRTC-Net
python tools/fps_benchmark.py --source test.mp4 --model checkpoints/best.pt

# Baseline: YOLO + ByteTrack
python tools/fps_benchmark.py --source test.mp4 --model best_yolo.pt --baseline yolo

# Baseline: RT-DETR + ByteTrack
python tools/fps_benchmark.py --source test.mp4 --model bestRT.pt --baseline rtdetr

# Headless server
python tools/fps_benchmark.py --source test.mp4 --model checkpoints/best.pt --no-display
```

---

## Depth Visualization

Reproduces Fig. 11 in the paper (progressive depth threshold screening):

```bash
# Single threshold (default 0.75 m = harvesting manipulator radius)
python tools/visualize_depth.py --rgb 501.png --depth 501-d1.png --threshold 0.75

# Progressive series (0.1 m to 1.0 m, step 0.1 m)
python tools/visualize_depth.py \
    --rgb 501.png --depth 501-d1.png \
    --progressive --min-m 0.1 --max-m 1.0 --step-m 0.1 \
    --output depth_viz/
```

---

## Architecture Details

### DGM-Encoder

```
RGB feature (B,C,H,W) ──┐
                         ├──→ DGA (depth gating) ──→ gated RGB
Depth (B,1,H,W)  ────────┘
                         └──→ depth tokens
                                    │
                         [M-AIFI: multimodal attention]
                                    │
                         enriched feature (B,C,H,W)
```

- **DGA**: 1×1 conv → 3×3 conv → Sigmoid gate (inversely proportional to depth)
- **M-AIFI**: Q from RGB; K,V from concat(RGB, depth); adaptive fusion weight β via MLP+Sigmoid

### UCL-Decoder

Three query types per frame:
| Query Type | Condition | Purpose |
|------------|-----------|---------|
| Initialization | new detections | Discover newly visible fruits |
| Tracking | uncertainty ≤ τ_low (0.3) | Maintain stable trajectories |
| Recovery | τ_low < uncertainty ≤ τ_high (0.7) | Re-identify occluded targets |

### GCS-Head

Multi-task branches from shared 256-dim features:

| Branch | Architecture | Output |
|--------|-------------|--------|
| Detection | 256 → 256 → 4+C | boxes + class logits |
| Depth | 256 → 128 → 32 → 1 + Sigmoid | depth (meters) + confidence |
| Tracking | 256 → 512 → 256 + L2-norm | identity embedding |
| Plant ID | 256 → 128 → 32 + spatial attn → N_plants | plant association |

Geometric constraint loss guides fruit–stem association using:
- **S_adj**: Gaussian of center distance  
- **S_cluster**: cluster distribution  
- **S_dir**: cosine of angular alignment with stem direction

---

## Citation

```bibtex
@article{mrtcnet2025,
  title   = {MRTC-Net: A Multi-modal Real-time Tracking and Counting Network 
             for Tomato Yield Estimation in Greenhouse},
  author  = {Xuanjingwei, et al.},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026}
}
```

---

## Contact

If you have any problems, please contact us by email: **xuanjingwei@stu.ahua.edu.cn**

*Anhui Agricultural University, School of Electronics and Electrical Engineering*
