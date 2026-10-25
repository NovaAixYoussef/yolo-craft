import random
import time
from pathlib import Path

import cv2
import numpy as np

from .common import (
    DATA_YAML,
    DATASET_DIR,
    PROCESSED_DIR,
    collect_images,
    load_yaml,
    normalize_names,
)

PALETTE = [
    (0, 193, 255),
    (0, 220, 130),
    (255, 170, 0),
    (255, 99, 132),
    (180, 120, 255),
    (0, 255, 255),
    (120, 255, 0),
    (255, 120, 220),
]
PAD_COLOR = (17, 20, 25)
TEXT_COLOR = (245, 247, 250)
MUTED_COLOR = (154, 164, 178)
ACCENT_COLOR = (11, 158, 245)
MODES = ("mosaic", "scale", "color", "flip", "random")


def resolve_images_dir(requested):
    candidates = [
        Path(requested),
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "images" / "val",
        PROCESSED_DIR,
    ]
    for candidate in candidates:
        if collect_images(candidate):
            return candidate
    return Path(requested)


def infer_labels_dir(images_dir):
    parts = list(Path(images_dir).parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts)
    return None


def load_names(data_path):
    payload = load_yaml(data_path)
    names = normalize_names(payload.get("names", {}))
    return names or {0: "object"}


def read_labels(label_path, width, height):
    if label_path is None or not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            center_x = float(parts[1]) * width
            center_y = float(parts[2]) * height
            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height
        except ValueError:
            continue
        x1 = center_x - box_width / 2.0
        y1 = center_y - box_height / 2.0
        x2 = center_x + box_width / 2.0
        y2 = center_y + box_height / 2.0
        if x2 - x1 > 1 and y2 - y1 > 1:
            boxes.append((class_id, x1, y1, x2, y2))
    return boxes


def class_color(class_id):
    return PALETTE[class_id % len(PALETTE)]


def draw_boxes(image, boxes, names):
    for class_id, x1, y1, x2, y2 in boxes:
        color = class_color(class_id)
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(image, p1, p2, color, 2)
        name = names.get(class_id, str(class_id))
        label = f"{name} [{class_id}]"
        font_scale = 0.45
        thickness = 1
        size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        top = max(0, p1[1] - size[1] - baseline - 6)
        cv2.rectangle(
            image,
            (p1[0], top),
            (p1[0] + size[0] + 8, top + size[1] + baseline + 6),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (p1[0] + 4, top + size[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (10, 12, 16),
            thickness,
            cv2.LINE_AA,
        )


def clip_box(box, width, height):
    class_id, x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return class_id, x1, y1, x2, y2


def letterbox(image, boxes, width, height):
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    new_width = max(1, int(round(source_width * scale)))
    new_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), PAD_COLOR, dtype=np.uint8)
    offset_x = (width - new_width) // 2
    offset_y = (height - new_height) // 2
    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = (
        resized
    )
    transformed = []
    for class_id, x1, y1, x2, y2 in boxes:
        box = clip_box(
            (
                class_id,
                x1 * scale + offset_x,
                y1 * scale + offset_y,
                x2 * scale + offset_x,
                y2 * scale + offset_y,
            ),
            width,
            height,
        )
        if box is not None:
            transformed.append(box)
    return canvas, transformed


def load_entry(image_path, labels_dir):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    label_path = None
    if labels_dir is not None:
        label_path = Path(labels_dir) / f"{image_path.stem}.txt"
    boxes = read_labels(label_path, width, height)
    return {"path": image_path, "image": image, "boxes": boxes}


def scale_transform(image, boxes, size, rng):
    base, base_boxes = letterbox(image, boxes, size, size)
    scale = rng.uniform(0.7, 1.35)
    shift_x = rng.uniform(-0.12, 0.12) * size
    shift_y = rng.uniform(-0.12, 0.12) * size
    matrix = np.float32([[scale, 0, shift_x], [0, scale, shift_y]])
    transformed = cv2.warpAffine(
        base,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=PAD_COLOR,
    )
    transformed_boxes = []
    for class_id, x1, y1, x2, y2 in base_boxes:
        box = clip_box(
            (
                class_id,
                x1 * scale + shift_x,
                y1 * scale + shift_y,
                x2 * scale + shift_x,
                y2 * scale + shift_y,
            ),
            size,
            size,
        )
        if box is not None:
            transformed_boxes.append(box)
    return transformed, transformed_boxes


def color_transform(image, boxes, size, rng):
    base, base_boxes = letterbox(image, boxes, size, size)
    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-12, 12)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.65, 1.45), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.75, 1.3), 0, 255)
    transformed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return transformed, base_boxes


def flip_transform(image, boxes, size, rng):
    base, base_boxes = letterbox(image, boxes, size, size)
    transformed = cv2.flip(base, 1)
    transformed_boxes = []
    for class_id, x1, y1, x2, y2 in base_boxes:
        box = clip_box((class_id, size - x2, y1, size - x1, y2), size, size)
        if box is not None:
            transformed_boxes.append(box)
    return transformed, transformed_boxes


def mosaic_transform(entries, size, rng):
    half = size // 2
    quadrants = [
        (0, 0, half, half),
        (half, 0, size - half, half),
        (0, half, half, size - half),
        (half, half, size - half, size - half),
    ]
    canvas = np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    all_boxes = []
    selected = [rng.choice(entries) for _ in range(4)]
    for entry, (offset_x, offset_y, width, height) in zip(selected, quadrants):
        panel, panel_boxes = letterbox(
            entry["image"],
            entry["boxes"],
            width,
            height,
        )
        canvas[
            offset_y : offset_y + height,
            offset_x : offset_x + width,
        ] = panel
        for class_id, x1, y1, x2, y2 in panel_boxes:
            box = clip_box(
                (
                    class_id,
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                ),
                size,
                size,
            )
            if box is not None:
                all_boxes.append(box)
    return canvas, all_boxes


def build_frame(mode, entries, index, size, rng):
    if mode == "mosaic":
        return mosaic_transform(entries, size, rng)
    entry = entries[index % len(entries)]
    if mode == "scale":
        return scale_transform(entry["image"], entry["boxes"], size, rng)
    if mode == "color":
        return color_transform(entry["image"], entry["boxes"], size, rng)
    if mode == "flip":
        return flip_transform(entry["image"], entry["boxes"], size, rng)
    selected = rng.choice(MODES[:4])
    if selected == "mosaic":
        return mosaic_transform(entries, size, rng)
    if selected == "scale":
        return scale_transform(entry["image"], entry["boxes"], size, rng)
    if selected == "color":
        return color_transform(entry["image"], entry["boxes"], size, rng)
    return flip_transform(entry["image"], entry["boxes"], size, rng)


def add_overlay(canvas, mode, image_name, paused, box_count):
    header = canvas.copy()
    cv2.rectangle(header, (0, 0), (canvas.shape[1], 78), (15, 17, 21), -1)
    cv2.addWeighted(header, 0.88, canvas, 0.12, 0, canvas)
    state = "PAUSED" if paused else "LIVE"
    cv2.putText(
        canvas,
        f"Augmentation: {mode.upper()}  {state}  Boxes {box_count}",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        ACCENT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "[m] mosaic   [s] scale   [c] color   [f] flip   [r] random",
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        MUTED_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"[space] pause   [n] next   [q] quit   |   {image_name}",
        (12, 69),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        MUTED_COLOR,
        1,
        cv2.LINE_AA,
    )


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "augment",
        help="Preview YOLO-style augmentations in real time.",
    )
    parser.add_argument(
        "--images",
        default=str(DATASET_DIR / "images" / "train"),
    )
    parser.add_argument("--labels", default=None)
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--interval", type=float, default=0.45)
    parser.add_argument("--window", default="YOLO CRAFT - Augmentation")
    parser.set_defaults(func=run)
    return parser


def run(args):
    images_dir = resolve_images_dir(args.images)
    images = collect_images(images_dir)
    if not images:
        print(f"Error: no training images found in '{images_dir}'.")
        return 1

    labels_dir = Path(args.labels) if args.labels else infer_labels_dir(images_dir)
    names = load_names(args.data)
    entries = []
    for image_path in images:
        entry = load_entry(image_path, labels_dir)
        if entry is not None:
            entries.append(entry)
    if not entries:
        print("Error: no readable images were found.")
        return 1

    rng = random.Random(args.seed)
    mode = "random"
    paused = False
    index = 0
    last_update = 0.0
    size = max(320, int(args.imgsz))
    canvas = None

    print(f"Images: {len(entries)} from '{images_dir}'")
    print(f"Labels: {labels_dir if labels_dir is not None else 'none'}")
    print("Controls: m mosaic, s scale, c color, f flip, r random")
    print("          space pause, n next image, q quit")

    try:
        while True:
            now = time.monotonic()
            if canvas is None or (
                not paused and now - last_update >= args.interval
            ):
                frame, boxes = build_frame(mode, entries, index, size, rng)
                draw_boxes(frame, boxes, names)
                add_overlay(
                    frame,
                    mode,
                    entries[index % len(entries)]["path"].name,
                    paused,
                    len(boxes),
                )
                canvas = frame
                last_update = now

            cv2.imshow(args.window, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            if key == ord("m"):
                mode = "mosaic"
                canvas = None
            elif key == ord("s"):
                mode = "scale"
                canvas = None
            elif key == ord("c"):
                mode = "color"
                canvas = None
            elif key == ord("f"):
                mode = "flip"
                canvas = None
            elif key == ord("r"):
                mode = "random"
                canvas = None
            elif key == ord(" "):
                paused = not paused
                canvas = None
            elif key == ord("n"):
                index = (index + 1) % len(entries)
                canvas = None
            if cv2.getWindowProperty(args.window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
    return 0
