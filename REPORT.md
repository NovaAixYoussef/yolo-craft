# YOLO CRAFT - Training Report

Generated: **2026-10-25 16:08 UTC**

Run directory: `outputs/runs/train`

> **Illustrative data.** No training results were found at `outputs/runs/train/results.csv`. The generated sample curves are for illustration only. Run `python -m yolo_craft train` and then `python -m yolo_craft report` to replace this report with measured results.

## Summary

| Metric | Value |
| --- | ---: |
| Epochs recorded | 60 |
| Best epoch | 59 |
| Best mAP@0.5:0.95 | 0.6210 |
| Final mAP@0.5 | 0.8909 |
| Final mAP@0.5:0.95 | 0.6138 |
| Final box loss | 0.5060 |
| Final class loss | 0.3723 |

## Training curves

Curves file: `outputs/reports/training_curves.png`

## Dataset

No dataset found. Capture and label images with `python -m yolo_craft label` to populate this section.

## Reproduce

```bash
python -m yolo_craft train
python -m yolo_craft report
python -m yolo_craft benchmark --image <path>
python -m yolo_craft export --format all
```

## Next steps

- Train a model so this report reflects real measurements.
- Add hard negatives and varied lighting to the next labeling round.
- Tune the confidence and IoU thresholds on a held-out validation set.
- Export the best checkpoint for the target device.
