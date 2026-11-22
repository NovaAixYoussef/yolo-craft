# YOLO CRAFT - Training Report

Generated: **2026-11-21 13:05 UTC**

Run directory: `outputs/runs/train`

> **Measured data.** Parsed from `outputs/runs/train/results.csv`.

## Summary

| Metric | Value |
| --- | ---: |
| Epochs recorded | 30 |
| Best epoch | 30 |
| Best mAP@0.5:0.95 | 0.6973 |
| Final mAP@0.5 | 0.9950 |
| Final mAP@0.5:0.95 | 0.6973 |
| Final box loss | 1.1397 |
| Final class loss | 1.5643 |

## Training curves

![Training performance curves](outputs/reports/training_curves.png)

## Dataset

| Item | Value |
| --- | ---: |
| Training images | 58 |
| Validation images | 15 |
| Labeled boxes | 60 |
| Classes | 1 |

| Class id | Name | Instances |
| ---: | --- | ---: |
| 0 | Calculator | 60 |

## Training configuration

| Setting | Value |
| --- | --- |
| `model` | `models\yolov8n.pt` |
| `data` | `C:\Users\youss\Desktop\Yolo Craft\data\dataset\data.yaml` |
| `epochs` | `30` |
| `imgsz` | `640` |
| `batch` | `8` |
| `patience` | `25` |
| `optimizer` | `auto` |
| `lr0` | `0.01` |
| `lrf` | `0.01` |
| `momentum` | `0.937` |
| `weight_decay` | `0.0005` |
| `warmup_epochs` | `3.0` |
| `cos_lr` | `True` |
| `close_mosaic` | `10` |
| `mosaic` | `1.0` |
| `fliplr` | `0.5` |
| `box` | `7.5` |
| `cls` | `0.5` |
| `device` | `cpu` |
| `seed` | `42` |

## Artifacts

- Best checkpoint: `outputs\runs\train\weights\best.pt`

## Reproduce

```bash
python -m yolo_craft train
python -m yolo_craft report
python -m yolo_craft benchmark --image <path>
python -m yolo_craft export --format all
```

## Next steps

- The best epoch is the final epoch. Consider training longer.
- Add hard negatives and varied lighting to the next labeling round.
- Tune the confidence and IoU thresholds on a held-out validation set.
- Export the best checkpoint for the target device.
