<p align="center">
  <img src="demo/media/promo.gif" alt="YOLO CRAFT promo video" width="720">
</p>

<p align="center">
  <a href="demo/app/YOLO_CRAFT_Promo_1080p.mp4"><strong>Full 1080p video (MP4)</strong></a>
  &middot;
  <a href="https://novaaixyoussef.github.io/yolo-craft/demo/app/yolo-craft-video.html">Video player app</a>
  &middot;
  <a href="https://novaaixyoussef.github.io/yolo-craft/demo/app/yolo-craft-pixel-v2.html">Pixel motion graphic</a>
</p>

---

<p align="center">
  <img src="assets/logo.jpg" alt="YOLO CRAFT logo" width="320">
</p>

<h1 align="center">YOLO CRAFT</h1>

<p align="center">
  <a href="https://github.com/NovaAixYoussef/yolo-craft/actions/workflows/ci.yml"><img src="https://github.com/NovaAixYoussef/yolo-craft/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/NovaAixYoussef/yolo-craft/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/ultralytics-YOLOv8-111F68.svg" alt="Ultralytics YOLOv8">
  <img src="https://img.shields.io/badge/version-1.0.0-green.svg" alt="Version 1.0.0">
</p>

<p align="center">
  A scriptable active-learning workflow for YOLO object detection: capture hard frames, label them,
  fine-tune a model, test it on video or streams, and ship deployment artifacts.
</p>

## Results first

**TRAINED WITH JUST 100 IMAGES IN A FEW MINUTES**

<p align="center">
  <img src="demo/poster.png" alt="YOLO CRAFT Apple vs Banana result poster" width="900">
</p>

<p align="center">
  <img src="demo/media/demo.gif" alt="YOLO CRAFT Apple vs Banana detection demo" width="480">
</p>

<p align="center">
  <a href="demo/media/demo.mp4">Full detection video (MP4)</a> &middot;
  <a href="demo/">All run artifacts in one folder</a>
</p>

| Metric | Value |
| --- | --- |
| Precision | 0.958 |
| Recall | 1.000 |
| mAP50 | 0.995 |
| mAP50-95 | 0.664 |
| Training time | 2.4 minutes |

The example fine-tunes COCO-pretrained `yolov8n.pt` on 50 apple and 50 banana images. The two
provided test images were run through the trained checkpoint:

| Test image | Detection | Confidence |
| --- | --- | --- |
| ![Apple detection](demo/detections/apple_detected.jpg) | Apple | 0.94 |
| ![Banana detection](demo/detections/banana_detected.jpg) | Banana | 0.87 |

Everything for the example is in one folder: [`demo/`](demo/). It contains the weights,
the full training run with plots, the source images, the annotated test images, the video, and the
GIF. No webcam or personal footage is used.

## Why YOLO CRAFT

- One entry point: `python -m yolo_craft`
- Full active-learning loop from a webcam, video file, or RTSP/HTTP stream
- Fast OpenCV labeler with class cycling, undo, background frames, and dataset splitting
- YOLO training with tunable loss, schedule, and augmentation parameters
- Inference with class filters, confidence, IoU, FPS, folder batches, and JSON/TXT exports
- Whole-frame annotated previews that scale to the available screen
- Augmentation preview with YOLO-style letterboxing, mosaic, color shifts, and flips
- Export to ONNX, TFLite, CoreML, and TensorRT engine
- Training-curve plots and a generated `REPORT.md`
- Before/after checkpoint benchmark on a validation image
- Pytest suite and GitHub Actions CI

## Requirements

- Python 3.10 or newer
- `pip`
- A webcam, video file, or network stream for capture and live testing
- Optional: an NVIDIA GPU with CUDA for faster training

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

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Optional editable install for the `yolo-craft` command:

```bash
python -m pip install -e ".[dev]"
yolo-craft --help
```

Running `python -m yolo_craft ...` from the repository root works without the editable install.

## Five-command quick start

```bash
python -m yolo_craft capture --source 0
python -m yolo_craft label --classes "Apple,Banana"
python -m yolo_craft train --epochs 100
python -m yolo_craft test --source 0
python -m yolo_craft report
```

To reproduce the shipped demo without a webcam, start with the 100 images and the full training run
in [`demo/`](demo/).

## Command reference

| Command | Purpose |
| --- | --- |
| `capture` | Capture uncertain frames from a webcam, video, or stream |
| `label` | Draw boxes, assign classes, and split the dataset |
| `train` | Fine-tune a YOLO checkpoint |
| `test` | Run a trained model on a webcam, video, image, folder, or stream |
| `active` | Capture hard frames, label them, and rebuild the dataset |
| `augment` | Preview YOLO-style training augmentations |
| `export` | Build deployment artifacts |
| `report` | Plot training curves and write `REPORT.md` |
| `benchmark` | Plot runs and compare baseline vs. fine-tuned detections |
| `archive` | Zip the generated dataset |

```bash
python -m yolo_craft --help
python -m yolo_craft train --help
```

## Workflow

### 1. Capture uncertain frames

The baseline model saves frames when it finds no objects or when detection confidence is below
`--uncertain-conf`.

```bash
python -m yolo_craft capture --source 0
python -m yolo_craft capture --source data/raw/demo.mp4
python -m yolo_craft capture --source "rtsp://user:password@192.168.1.20:554/stream"
```

Controls: `s` saves the current frame, `a` toggles automatic capture, `q` closes the window.
Captured frames go to `data/raw/`.

### 2. Label and split the dataset

```bash
python -m yolo_craft label --classes "Apple,Banana"
```

Existing class names are loaded from `data/dataset/data.yaml`. Pass `--classes` only when defining
a new class set.

Labeling controls:

- Left-drag to draw a box.
- `l` cycles through classes.
- `1` through `9` select a class directly.
- `s` saves the frame and its YOLO labels.
- `b` saves the frame as a background image with an empty label file.
- `c` removes the most recent box.
- `n` skips the frame.
- `q` closes the session and still performs the dataset split.

Custom split:

```bash
python -m yolo_craft label --train-ratio 0.85 --seed 12
```

### 3. Train

```bash
python -m yolo_craft train --epochs 150 --imgsz 640 --batch 8 --device 0
```

Defaults: `models/yolov8n.pt`, `outputs/runs/train/`, cosine learning rate, `close_mosaic=10`,
`box=7.5`, `cls=0.5`, `fliplr=0.5`, and `mosaic=1.0`.

| Option | Default | Purpose |
| --- | --- | --- |
| `--cos-lr` / `--no-cos-lr` | enabled | Cosine learning-rate schedule |
| `--close-mosaic` | `10` | Disable mosaic for the final epochs |
| `--box` | `7.5` | Box loss gain |
| `--cls` | `0.5` | Class loss gain |
| `--fliplr` | `0.5` | Horizontal flip probability |
| `--mosaic` | `1.0` | Mosaic augmentation probability |

```bash
python -m yolo_craft train --epochs 150 --device 0 --no-cos-lr --close-mosaic 15 --box 6.0 --cls 0.7 --fliplr 0.3 --mosaic 0.8
```

If the baseline checkpoint is missing, it is downloaded into `models/` automatically.

### 4. Test the model

```bash
python -m yolo_craft test --source 0
python -m yolo_craft test --weights outputs/runs/train/weights/best.pt --conf 0.25 --source 0
python -m yolo_craft test --weights models/best.pt --source image.jpg --conf 0.1 --imgsz 1024 --augment
python -m yolo_craft test --weights models/best.pt --source image.jpg --save outputs/annotated.jpg
python -m yolo_craft test --source input.mp4 --save outputs/annotated.mp4
python -m yolo_craft test --source 0 --classes "Apple,Banana"
python -m yolo_craft test --source data/raw --save outputs/annotated --save-txt outputs/labels --save-json outputs/detections.json --no-display
```

The test command uses `outputs/runs/train/weights/best.pt` by default. Folder runs write
`<name>_pred.<ext>` images, normalized YOLO labels, and a JSON file with class names, confidence
values, and pixel boxes. Use `--no-labels`, `--no-conf`, `--line-width`, `--device`, and
`--no-display` to control annotation style, device, and preview behavior.

### 5. Run the active-learning loop

```bash
python -m yolo_craft active
python -m yolo_craft active --no-label
```

The loop captures uncertain frames, opens the labeler, removes labeled source frames, and rebuilds
the dataset.

## Augmentation preview

See exactly how YOLOv8 changes training images before a single epoch runs. The live preview renders
YOLO-style letterboxing, a four-image mosaic, scale and shift, HSV color jitter, and horizontal flip.
Boxes are remapped onto the augmented canvas so you can confirm labels stay aligned.

```bash
python -m yolo_craft augment
```

Controls: `m` mosaic, `s` scale and shift, `c` color jitter, `f` horizontal flip, `r` random mode,
`space` pause, `n` next image, `q` quit.

```bash
python -m yolo_craft augment --images data/processed --labels data/processed
```

## Benchmarks

```bash
python -m yolo_craft benchmark --image path/to/test.jpg
python -m yolo_craft benchmark --image path/to/test.jpg --baseline models/yolov8n.pt --finetuned outputs/runs/train/weights/best.pt
python -m yolo_craft benchmark --image path/to/test.jpg --sample-curves
```

Outputs go to `outputs/benchmarks/` with `before.png`, `after.png`, `before_after.png`,
`metrics.json`, and `curves/`.

## Report

```bash
python -m yolo_craft report
python -m yolo_craft report --sample
```

The report command parses `outputs/runs/train/results.csv`, writes
`outputs/reports/training_curves.png`, and updates `REPORT.md` with best and final mAP, losses,
dataset counts, per-class instances, benchmark values, and the training configuration.

## Export

```bash
python -m yolo_craft export --format all
python -m yolo_craft export --format onnx --imgsz 640 --simplify
python -m yolo_craft export --format tflite
python -m yolo_craft export --format coreml
python -m yolo_craft export --format engine --device 0
```

Artifacts are copied to `outputs/exports/`.

| Format | Typical target | Notes |
| --- | --- | --- |
| `onnx` | Desktop, server, browser | Requires ONNX dependencies pulled by Ultralytics |
| `tflite` | Android, edge devices | Requires TensorFlow and platform setup |
| `coreml` | iOS and macOS | Requires CoreML tools and macOS |
| `engine` | NVIDIA GPUs | Requires TensorRT and a matching CUDA environment |

## Dataset archive

```bash
python -m yolo_craft archive
python -m yolo_craft archive --source data/dataset --output outputs/dataset.zip
```

## Data contract

YOLO label files contain one box per line:

```text
class_id center_x center_y width height
```

Coordinates are normalized to `[0, 1]`. Every image should have a matching `.txt` file; background
images use an empty file.

```yaml
train: images/train
val: images/val
names:
  0: Apple
  1: Banana
```

## Project structure

```text
yolo-craft/
+-- .github/workflows/ci.yml
+-- assets/               logo and project graphics
+-- data/                 ignored runtime data
+-- demo/                 complete Apple vs Banana demo and web apps
|   +-- app/              promo video player and pixel motion graphic
|   +-- media/            GIF, MP4 and augmentation preview
|   +-- scripts/          media and preview generators
+-- models/               baseline and fine-tuned checkpoints
+-- outputs/              ignored runs, exports, benchmarks, reports
+-- tests/
+-- yolo_craft/
|   +-- active.py
|   +-- archiving.py
|   +-- augmentation.py
|   +-- benchmarking.py
|   +-- capture.py
|   +-- cli.py
|   +-- common.py
|   +-- exporting.py
|   +-- inference.py
|   +-- labeling.py
|   +-- metrics.py
|   +-- reporting.py
|   +-- training.py
|   +-- __init__.py
|   +-- __main__.py
+-- LICENSE
+-- README.md
+-- REPORT.md
+-- pyproject.toml
+-- requirements.txt
+-- requirements-dev.txt
```

The `data/`, `models/`, and `outputs/` directories keep their `.gitkeep` files while generated
contents are ignored.

## Testing

```bash
python -m pytest
python -m compileall -q yolo_craft tests demo\scripts
```

The suite covers source parsing, capture filenames, label splitting, dataset YAML generation,
training parameter forwarding, inference class filters, folder batches, JSON/TXT exports,
augmentation geometry, CSV parsing, report generation, archiving, CLI wiring, and export copying.

## Configuration notes

- Weights are ignored by git outside the shipped demo. Keep them in `models/`, `outputs/runs/.../weights/`, or `demo/model/`.
- Training projects go under `outputs/runs/`.
- Export output goes under `outputs/exports/`.
- Benchmark output goes under `outputs/benchmarks/`.
- Curves and reports go under `outputs/reports/`.
- All Python source, CLI output, and documentation use English.
- Python files contain no comments or docstrings.

## Troubleshooting

### Webcam does not open

```bash
python -m yolo_craft test --source 1
```

### CUDA is not available

```bash
python -m yolo_craft train --device cpu
```

### Training cannot find the dataset

Check that `data/dataset/data.yaml` exists and its paths are valid, then rerun:

```bash
python -m yolo_craft label
```

### Export fails for one format

ONNX, TFLite, CoreML, and TensorRT need different optional dependencies, and CoreML needs macOS.
The other formats still complete and the command reports which format failed.

### No validation image for a benchmark

```bash
python -m yolo_craft benchmark --image path/to/test.jpg
```

## Roadmap

- Host the example checkpoint with the v1.0.0 release
- Add a Colab notebook for the Apple vs Banana example
- Add CI artifacts for the generated demo media

## License

This project is licensed under the GNU Affero General Public License v3.0. See `LICENSE`. The demo
images come from the MIT-licensed `PedroSampaio/fruits-360` dataset on Hugging Face.
