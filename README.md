<p align="center">
  <img src="assets/logo.jpg" alt="YOLO CRAFT logo" width="360">
</p>

<h1 align="center">YOLO CRAFT</h1>

<p align="center">
  <a href="https://github.com/NovaAixYoussef/yolo-craft/actions/workflows/ci.yml"><img src="https://github.com/NovaAixYoussef/yolo-craft/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/NovaAixYoussef/yolo-craft/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/ultralytics-YOLOv8-111F68.svg" alt="Ultralytics YOLOv8">
</p>

YOLO CRAFT is a complete, scriptable active-learning workflow for YOLO object detection. It captures difficult frames, labels them with a fast OpenCV interface, builds a YOLO dataset, fine-tunes a model, tests it on webcams or streams, previews augmentations, exports deployment formats, and produces a Markdown report.

The repository ships without runtime data. Source code, tests, documentation, and the project logo are tracked; captured frames, datasets, checkpoints, plots, benchmarks, and exports are written to `data/` and `outputs/`, which are ignored by git.

## Highlights

- One command-line entry point: `python -m yolo_craft`
- Active-learning loop from webcam, video file, or RTSP/HTTP stream
- Multi-class labeling with class cycling, undo, background frames, and dataset splitting
- YOLO training with sensible augmentation and early-stopping defaults
- Live inference with confidence, IoU, FPS, and optional video recording
- Real-time augmentation preview using YOLO-style letterboxing, mosaic, scaling, color shifts, and flips
- Export to ONNX, TFLite, CoreML, and TensorRT engine
- Training-curve plots and a generated `REPORT.md`
- Before/after checkpoint benchmark on a validation image
- Pytest test suite and GitHub Actions CI

## Requirements

- Python 3.10 or newer
- `pip`
- A webcam, video file, or network stream for capture and live testing
- Optional: an NVIDIA GPU with a working CUDA installation for faster training

## Installation

```bash
git clone https://github.com/NovaAixYoussef/yolo-craft.git
cd yolo-craft
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# Linux or macOS
source .venv/bin/activate
```

Install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the test tools as well:

```bash
python -m pip install -r requirements-dev.txt
```

Install as an editable package and use the `yolo-craft` command:

```bash
python -m pip install -e ".[dev]"
yolo-craft --help
```

`python -m yolo_craft ...` works without installing the package when you run it from the repository root.

## Project structure

```text
yolo-craft/
├── .github/
│   └── workflows/
│       └── ci.yml
├── assets/
│   └── logo.jpg
├── data/
│   ├── raw/                  # Uncertain frames waiting for labels
│   ├── processed/            # Labeled images and YOLO label files
│   └── dataset/              # Generated YOLO train/val dataset
├── models/                   # Baseline and fine-tuned checkpoints
├── outputs/
│   ├── runs/                 # Ultralytics training runs
│   ├── exports/              # ONNX, TFLite, CoreML, and TensorRT files
│   ├── benchmarks/           # Before/after metrics and figures
│   └── reports/              # Training curves
├── tests/
├── yolo_craft/
│   ├── active.py
│   ├── archiving.py
│   ├── augmentation.py
│   ├── benchmarking.py
│   ├── capture.py
│   ├── cli.py
│   ├── common.py
│   ├── exporting.py
│   ├── inference.py
│   ├── labeling.py
│   ├── metrics.py
│   ├── reporting.py
│   └── training.py
├── .gitignore
├── LICENSE
├── README.md
├── REPORT.md
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

The `data/`, `models/`, and `outputs/` directories keep their `.gitkeep` files while their generated contents are ignored.

## Command overview

| Command | Purpose |
| --- | --- |
| `capture` | Capture uncertain frames from a webcam, video, or stream |
| `label` | Draw boxes, assign classes, and split the dataset |
| `train` | Fine-tune a YOLO checkpoint |
| `test` | Run a trained model on a webcam, video, image, or stream |
| `active` | Capture hard frames, label them, and rebuild the dataset |
| `augment` | Preview YOLO-style training augmentations |
| `export` | Build deployment artifacts |
| `report` | Plot training curves and write `REPORT.md` |
| `benchmark` | Plot runs and compare baseline vs. fine-tuned detections |
| `archive` | Zip the generated dataset |

Show help for any command:

```bash
python -m yolo_craft --help
python -m yolo_craft train --help
```

## Quick start

### 1. Capture uncertain frames

The baseline model saves frames when it finds no objects or when a detection confidence is below `--uncertain-conf`.

```bash
python -m yolo_craft capture --source 0
```

Video file:

```bash
python -m yolo_craft capture --source data/raw/demo.mp4
```

RTSP stream:

```bash
python -m yolo_craft capture --source "rtsp://user:password@192.168.1.20:554/stream"
```

Controls:

- `s` saves the current frame manually.
- `a` toggles automatic uncertain-frame capture.
- `q` closes the window.

Captured frames go to `data/raw/`.

### 2. Label and split the dataset

```bash
python -m yolo_craft label
```

Existing class names are loaded from `data/dataset/data.yaml`. To start a new multi-class dataset, override them explicitly:

```bash
python -m yolo_craft label --classes "leaf,stem"
```

Labeling controls:

- Drag with the left mouse button to draw a box.
- `l` cycles through classes.
- `1` through `9` select a class directly.
- `s` saves the frame and its YOLO labels.
- `b` saves the frame as a background image with an empty label file.
- `c` removes the most recent box.
- `n` skips the frame.
- `q` closes the session and still performs the dataset split.

Labeled images and labels are written to `data/processed/`. The split command creates:

```text
data/dataset/
├── data.yaml
├── background/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Use a different split ratio or seed when needed:

```bash
python -m yolo_craft label --train-ratio 0.85 --seed 12
```

### 3. Train

```bash
python -m yolo_craft train
```

The default configuration uses `models/yolov8n.pt` and writes to `outputs/runs/train/`.

Useful overrides:

```bash
python -m yolo_craft train --epochs 150 --imgsz 640 --batch 8 --device 0
```

If the baseline checkpoint is missing, the training command downloads `yolov8n.pt` into `models/`.

Place a custom checkpoint in `models/` and pass it explicitly:

```bash
python -m yolo_craft train --model models/custom-base.pt
```

### 4. Test the model

```bash
python -m yolo_craft test --source 0
```

The test command uses `outputs/runs/train/weights/best.pt` by default. Run the baseline checkpoint explicitly before fine-tuning:

```bash
python -m yolo_craft test --weights models/yolov8n.pt --conf 0.25 --source 0
```

The baseline weights are downloaded automatically when they are missing.

Test a single image. The window stays open until you press a key:

```bash
python -m yolo_craft test --weights models/best.pt --source path/to/image.jpg
```

Save an annotated video:

```bash
python -m yolo_craft test --source input.mp4 --save outputs/annotated.mp4
```

### 5. Run the active-learning loop

```bash
python -m yolo_craft active
```

The loop captures uncertain frames into `data/raw/`, closes the capture window when you press `q`, opens the labeler, removes labeled source frames, and rebuilds `data/dataset/`. Existing class names are loaded from `data/dataset/data.yaml`; pass `--classes` only when defining a new class set.

Capture only, without opening the labeler:

```bash
python -m yolo_craft active --no-label
```

## Augmentation preview

```bash
python -m yolo_craft augment
```

The preview uses images from `data/dataset/images/train/`. Controls:

- `m` mosaic
- `s` scale and shift
- `c` color jitter
- `f` horizontal flip
- `r` random mode
- `space` pause
- `n` next image
- `q` quit

Use a different image folder:

```bash
python -m yolo_craft augment --images data/processed --labels data/processed
```

## Benchmarks

Plot all training runs and compare two checkpoints:

```bash
python -m yolo_craft benchmark --image path/to/test.jpg
```

If `--image` is omitted, the first validation image in `data/dataset/images/val/` is used, then the first training image.

The command writes:

```text
outputs/benchmarks/
├── before.png
├── after.png
├── before_after.png
├── metrics.json
└── curves/
    ├── sample_curves.png
    ├── summary.csv
    └── <run>_curves.png
```

Choose explicit checkpoints:

```bash
python -m yolo_craft benchmark \
  --image path/to/test.jpg \
  --baseline models/yolov8n.pt \
  --finetuned outputs/runs/train/weights/best.pt \
  --conf 0.25 \
  --iou 0.7
```

Write illustrative curves when no training runs exist:

```bash
python -m yolo_craft benchmark --image path/to/test.jpg --sample-curves
```

## Report

```bash
python -m yolo_craft report
```

The report command parses `outputs/runs/train/results.csv`, writes curves to `outputs/reports/training_curves.png`, and updates `REPORT.md`.

Generate an explicitly illustrative report when no training run exists:

```bash
python -m yolo_craft report --sample
```

Custom paths:

```bash
python -m yolo_craft report \
  --csv outputs/runs/experiment/results.csv \
  --run-dir outputs/runs/experiment \
  --curves outputs/reports/experiment_curves.png \
  --report outputs/reports/experiment.md
```

The report includes:

- Best and final mAP values
- Final box and class losses
- Dataset image and box counts
- Per-class instance counts
- Before/after benchmark values when `outputs/benchmarks/metrics.json` exists
- Training configuration from `args.yaml`
- Reproduction commands and next steps

## Export

Export every supported format:

```bash
python -m yolo_craft export --format all
```

Export one format:

```bash
python -m yolo_craft export --format onnx --imgsz 640 --simplify
python -m yolo_craft export --format tflite
python -m yolo_craft export --format coreml
python -m yolo_craft export --format engine --device 0
```

Artifacts are copied to `outputs/exports/`.

| Format | Typical target | Notes |
| --- | --- | --- |
| `onnx` | Desktop, server, browser | Requires ONNX dependencies pulled by Ultralytics |
| `tflite` | Android, edge devices | Requires TensorFlow and platform-specific setup |
| `coreml` | iOS and macOS | Requires CoreML tools and macOS |
| `engine` | NVIDIA GPUs | Requires TensorRT and a matching CUDA environment |

## Dataset archive

```bash
python -m yolo_craft archive
```

The default archive is `outputs/dataset.zip`. Customize paths:

```bash
python -m yolo_craft archive --source data/dataset --output outputs/dataset.zip
```

## Data contract

YOLO label files contain one box per line:

```text
class_id center_x center_y width height
```

All coordinates are normalized to the interval `[0, 1]`. The labeler writes this format directly. Every image should have a matching `.txt` file; background images use an empty file.

The generated `data/dataset/data.yaml` uses this structure:

```yaml
path: /absolute/path/to/data/dataset
train: images/train
val: images/val
names:
  0: leaf
  1: stem
```

## Testing

Run the full test suite:

```bash
python -m pytest
```

Run a syntax check:

```bash
python -m compileall -q yolo_craft tests
```

The suite covers source parsing, capture filenames, label splitting, dataset YAML generation, augmentation geometry, CSV parsing, report generation, archiving, CLI wiring, and export artifact copying.

## Configuration notes

- Weights are ignored by git. Keep them in `models/` or `outputs/runs/.../weights/`.
- Training projects go under `outputs/runs/`.
- Export output goes under `outputs/exports/`.
- Benchmark output goes under `outputs/benchmarks/`.
- Curves and reports go under `outputs/reports/`.
- All Python source, CLI output, and documentation use English.
- Python files contain no comments or docstrings.

## Troubleshooting

### Webcam does not open

Try another index:

```bash
python -m yolo_craft test --source 1
```

### CUDA is not available

Use CPU explicitly:

```bash
python -m yolo_craft train --device cpu
```

### Training cannot find the dataset

Check that this file exists and that its paths are valid:

```text
data/dataset/data.yaml
```

Then rerun:

```bash
python -m yolo_craft label
```

### Export fails for one format

ONNX, TFLite, CoreML, and TensorRT need different optional dependencies and, for CoreML, macOS. The other formats still complete and the command reports which format failed.

### No validation image for a benchmark

Pass an image explicitly:

```bash
python -m yolo_craft benchmark --image path/to/test.jpg
```

## License

This project is licensed under the GNU Affero General Public License v3.0. See `LICENSE`.
