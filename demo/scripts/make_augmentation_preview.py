import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from yolo_craft.augmentation import (
    build_frame,
    draw_boxes,
    letterbox,
    load_entry,
    load_names,
)

IMAGES_DIR = ROOT / "data" / "dataset" / "images" / "train"
LABELS_DIR = ROOT / "data" / "dataset" / "labels" / "train"
DATA_YAML = ROOT / "data" / "dataset" / "data.yaml"
OUT = ROOT / "demo" / "media" / "augmentation_preview.png"
SIZE = 320
MODES = ("original", "mosaic", "scale", "color", "flip", "random")


def main():
    paths = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    entries = []
    for path in paths:
        entry = load_entry(path, LABELS_DIR)
        if entry is not None:
            entries.append(entry)
    if not entries:
        raise SystemExit("No readable training images found.")

    names = load_names(str(DATA_YAML))
    rng = random.Random(7)
    panels = []
    for mode in MODES:
        if mode == "original":
            frame, boxes = letterbox(
                entries[0]["image"], entries[0]["boxes"], SIZE, SIZE
            )
        else:
            frame, boxes = build_frame(
                mode, entries, rng.randrange(len(entries)), SIZE, rng
            )
        draw_boxes(frame, boxes, names)
        cv2.putText(
            frame,
            mode.upper(),
            (12, SIZE - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(frame)

    grid = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), grid)
    print("entries", len(entries))
    print("classes", names)
    print("preview", OUT, grid.shape)


if __name__ == "__main__":
    main()
