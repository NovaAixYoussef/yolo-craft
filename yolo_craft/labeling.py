import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from .common import (
    BACKGROUND_DIR,
    DATA_YAML,
    DATASET_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    collect_images,
    read_image,
    write_image,
)

BOX_COLORS = [
    (0, 193, 255),
    (0, 220, 130),
    (255, 170, 0),
    (255, 99, 132),
    (180, 120, 255),
    (0, 255, 255),
    (120, 255, 0),
    (255, 120, 220),
]
HEADER_COLOR = (15, 17, 21)
TEXT_COLOR = (245, 247, 250)
MUTED_COLOR = (154, 164, 178)
ACCENT_COLOR = (11, 158, 245)
CURSOR_COLOR = (255, 255, 255)
HEADER_HEIGHT = 88
HEADER_MIN_WIDTH = 720
MAX_DISPLAY_WIDTH = 1500
MAX_DISPLAY_HEIGHT = 950
MAX_UPSCALE = 40.0
TARGET_DISPLAY_WIDTH = 1100.0
TARGET_DISPLAY_HEIGHT = 820.0
SCREEN_MARGIN = 20
MIN_BOX_SIZE = 5
_SCREEN_SIZE = None


def parse_class_override(value):
    if not value:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        return None
    return {index: name for index, name in enumerate(names)}


def load_classes(data_path, override=None):
    if override:
        return override
    data_path = Path(data_path)
    if data_path.exists():
        try:
            payload = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
        except (yaml.YAMLError, OSError):
            payload = {}
        names = payload.get("names")
        if isinstance(names, dict):
            parsed = {}
            for key, value in names.items():
                try:
                    parsed[int(key)] = str(value)
                except (TypeError, ValueError):
                    continue
            if parsed:
                return dict(sorted(parsed.items()))
        if isinstance(names, list) and names:
            return {index: str(value) for index, value in enumerate(names)}
    return {0: "object"}


def write_data_yaml(data_path, dataset_dir, classes):
    payload = {
        "path": str(Path(dataset_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {int(key): str(value) for key, value in sorted(classes.items())},
    }
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def split_dataset(
    dataset_dir,
    processed_dir,
    classes,
    train_ratio=0.8,
    seed=42,
    data_path=None,
):
    dataset_dir = Path(dataset_dir)
    processed_dir = Path(processed_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    images = collect_images(processed_dir)

    if not images:
        print("No processed images available for the dataset split.")
        return 0, 0

    for subset in ("train", "val"):
        for folder in (images_dir / subset, labels_dir / subset):
            folder.mkdir(parents=True, exist_ok=True)
            for item in folder.iterdir():
                if item.is_file():
                    item.unlink()

    random.Random(seed).shuffle(images)
    if len(images) == 1:
        split_index = 1
    else:
        split_index = int(len(images) * train_ratio)
        split_index = min(max(1, split_index), len(images) - 1)

    train_images = images[:split_index]
    val_images = images[split_index:]

    def copy_set(items, subset):
        for image_path in items:
            destination_image = images_dir / subset / image_path.name
            destination_image.write_bytes(image_path.read_bytes())
            source_label = processed_dir / f"{image_path.stem}.txt"
            destination_label = labels_dir / subset / f"{image_path.stem}.txt"
            if source_label.exists():
                destination_label.write_bytes(source_label.read_bytes())
            else:
                destination_label.write_text("", encoding="utf-8")

    copy_set(train_images, "train")
    copy_set(val_images, "val")
    data_file = Path(data_path) if data_path else dataset_dir / "data.yaml"
    write_data_yaml(data_file, dataset_dir, classes)

    print(
        f"Dataset split complete: {len(train_images)} train images, "
        f"{len(val_images)} validation images."
    )
    return len(train_images), len(val_images)


def screen_workspace():
    global _SCREEN_SIZE
    if _SCREEN_SIZE is not None:
        return _SCREEN_SIZE
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            rect = wintypes.RECT()
            success = ctypes.windll.user32.SystemParametersInfoW(
                0x0030, 0, ctypes.byref(rect), 0
            )
            width = rect.right - rect.left
            height = rect.bottom - rect.top
            if success and width > 0 and height > 0:
                _SCREEN_SIZE = (width, height)
                return _SCREEN_SIZE
        except Exception:
            pass
    _SCREEN_SIZE = (1280, 800)
    return _SCREEN_SIZE


def compute_display_scale(width, height):
    if width <= 0 or height <= 0:
        return 1.0
    screen_width, screen_height = screen_workspace()
    available_width = max(640, screen_width - 2 * SCREEN_MARGIN)
    available_height = max(
        400,
        screen_height - HEADER_HEIGHT - 2 * SCREEN_MARGIN,
    )
    max_width = min(MAX_DISPLAY_WIDTH, available_width)
    max_height = min(MAX_DISPLAY_HEIGHT, available_height)
    target_width = min(TARGET_DISPLAY_WIDTH, max_width)
    target_height = min(TARGET_DISPLAY_HEIGHT, max_height)
    downscale = min(max_width / width, max_height / height, 1.0)
    upscale = min(
        MAX_UPSCALE,
        target_width / width,
        target_height / height,
    )
    return max(downscale, upscale)


def display_to_image(x, y, scale, offset_y, image_width, image_height):
    if scale <= 0 or image_width <= 0 or image_height <= 0:
        return None
    image_y = y - offset_y
    if x < 0 or image_y < 0:
        return None
    if x >= image_width * scale or image_y >= image_height * scale:
        return None
    return int(x / scale), int(image_y / scale)


def clamp_display_to_image(x, y, scale, offset_y, image_width, image_height):
    if scale <= 0 or image_width <= 0 or image_height <= 0:
        return None
    image_x = int(x / scale)
    image_y = int((y - offset_y) / scale)
    return (
        max(0, min(image_width - 1, image_x)),
        max(0, min(image_height - 1, image_y)),
    )


def class_color(class_id, classes):
    keys = list(classes.keys())
    index = keys.index(class_id) if class_id in keys else 0
    return BOX_COLORS[index % len(BOX_COLORS)]


def draw_box(image, box, classes, scale=1.0, offset_y=0):
    x1, y1, x2, y2, class_id = box
    image_height, image_width = image.shape[:2]
    color = class_color(class_id, classes)
    display_x1 = max(0, min(image_width - 1, int(round(x1 * scale))))
    display_y1 = max(0, min(image_height - 1, offset_y + int(round(y1 * scale))))
    display_x2 = max(0, min(image_width - 1, int(round(x2 * scale))))
    display_y2 = max(0, min(image_height - 1, offset_y + int(round(y2 * scale))))
    if display_x2 <= display_x1 or display_y2 <= display_y1:
        return
    cv2.rectangle(
        image,
        (display_x1, display_y1),
        (display_x2, display_y2),
        color,
        2,
    )
    name = classes.get(class_id, str(class_id))
    label = f"{name} [{class_id}]"
    font_scale = 0.48
    thickness = 1
    size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    top = max(offset_y, display_y1 - size[1] - baseline - 6)
    cv2.rectangle(
        image,
        (display_x1, top),
        (display_x1 + size[0] + 10, top + size[1] + baseline + 6),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (display_x1 + 5, top + size[1] + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (10, 12, 16),
        thickness,
        cv2.LINE_AA,
    )


def render_frame(
    image,
    boxes,
    classes,
    selected_id,
    index,
    total,
    preview=None,
    cursor=None,
):
    height, width = image.shape[:2]
    scale = compute_display_scale(width, height)
    display_width = max(1, int(round(width * scale)))
    display_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (display_width, display_height),
        interpolation=interpolation,
    )
    canvas_width = max(display_width, HEADER_MIN_WIDTH)
    canvas = np.zeros(
        (HEADER_HEIGHT + display_height, canvas_width, 3),
        dtype=np.uint8,
    )
    canvas[HEADER_HEIGHT : HEADER_HEIGHT + display_height, :display_width] = resized

    for box in boxes:
        draw_box(canvas, box, classes, scale=scale, offset_y=HEADER_HEIGHT)

    if preview is not None:
        x1, y1, x2, y2 = preview
        color = class_color(selected_id, classes)
        display_x1 = max(0, min(display_width - 1, int(round(x1 * scale))))
        display_y1 = max(
            HEADER_HEIGHT,
            min(HEADER_HEIGHT + display_height - 1, HEADER_HEIGHT + int(round(y1 * scale))),
        )
        display_x2 = max(0, min(display_width - 1, int(round(x2 * scale))))
        display_y2 = max(
            HEADER_HEIGHT,
            min(HEADER_HEIGHT + display_height - 1, HEADER_HEIGHT + int(round(y2 * scale))),
        )
        if display_x2 > display_x1 and display_y2 > display_y1:
            cv2.rectangle(
                canvas,
                (display_x1, display_y1),
                (display_x2, display_y2),
                color,
                2,
            )

    if cursor is not None:
        cursor_x = int(round(cursor[0] * scale))
        cursor_y = HEADER_HEIGHT + int(round(cursor[1] * scale))
        if 0 <= cursor_x < display_width and HEADER_HEIGHT <= cursor_y < canvas.shape[0]:
            cv2.line(
                canvas,
                (max(0, cursor_x - 8), cursor_y),
                (min(canvas_width - 1, cursor_x + 8), cursor_y),
                CURSOR_COLOR,
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                canvas,
                (cursor_x, max(HEADER_HEIGHT, cursor_y - 8)),
                (cursor_x, min(canvas.shape[0] - 1, cursor_y + 8)),
                CURSOR_COLOR,
                1,
                cv2.LINE_AA,
            )

    cv2.rectangle(
        canvas,
        (0, 0),
        (canvas_width, HEADER_HEIGHT),
        HEADER_COLOR,
        -1,
    )
    progress = int((index / total) * 100) if total else 0
    cv2.putText(
        canvas,
        f"Image {index + 1}/{total}  {progress}%  Boxes {len(boxes)}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    class_ids = sorted(classes) if classes else []
    selected_index = class_ids.index(selected_id) if selected_id in class_ids else 0
    class_name = classes.get(selected_id, str(selected_id))
    cv2.putText(
        canvas,
        f"[l] Class {selected_index + 1}/{len(class_ids)}: "
        f"{class_name} [{selected_id}]",
        (12, 51),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        ACCENT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "[s] Save  [b] Background  [c] Undo  [n] Skip  [q] Quit",
        (12, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        MUTED_COLOR,
        1,
        cv2.LINE_AA,
    )
    return canvas, scale


def label_images(
    source_dir,
    processed_dir,
    background_dir,
    classes,
    remove_source=False,
    window="YOLO CRAFT - Labeler",
):
    source_dir = Path(source_dir)
    processed_dir = Path(processed_dir)
    background_dir = Path(background_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    background_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(source_dir)
    if not images:
        print(f"No images found in '{source_dir}'.")
        return {"saved": 0, "background": 0, "skipped": 0}

    if not classes:
        classes = {0: "object"}

    class_ids = sorted(classes)
    selected_index = 0
    selected_id = class_ids[selected_index]
    current_boxes = []
    drawing = False
    start = None
    end = None
    cursor = None
    start_display = None
    scale = 1.0
    width = 0
    height = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start, end, cursor, start_display
        point = display_to_image(x, y, scale, HEADER_HEIGHT, width, height)
        if event == cv2.EVENT_LBUTTONDOWN:
            if point is None:
                return
            drawing = True
            start = point
            end = point
            cursor = point
            start_display = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if point is not None:
                cursor = point
                if drawing:
                    end = point
            elif drawing:
                point = clamp_display_to_image(
                    x, y, scale, HEADER_HEIGHT, width, height
                )
                if point is not None:
                    end = point
        elif event == cv2.EVENT_LBUTTONUP:
            if point is None:
                point = clamp_display_to_image(
                    x, y, scale, HEADER_HEIGHT, width, height
                )
            if point is not None:
                end = point
                cursor = point
            if (
                not drawing
                or start is None
                or end is None
                or start_display is None
            ):
                drawing = False
                return
            if (
                abs(x - start_display[0]) <= MIN_BOX_SIZE
                or abs(y - start_display[1]) <= MIN_BOX_SIZE
            ):
                drawing = False
                start = None
                end = None
                start_display = None
                return
            x1, x2 = sorted((start[0], end[0]))
            y1, y2 = sorted((start[1], end[1]))
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            if x2 > x1 and y2 > y1:
                current_boxes.append((x1, y1, x2, y2, selected_id))
            drawing = False
            start = None
            end = None
            start_display = None

    try:
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, mouse_callback)
    except cv2.error as exc:
        print(f"Error: could not open labeling window: {exc}")
        return {"saved": 0, "background": 0, "skipped": 0}

    saved = 0
    background = 0
    skipped = 0
    quit_requested = False

    for index, image_path in enumerate(images):
        image = read_image(image_path)
        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        height, width = image.shape[:2]
        scale = compute_display_scale(width, height)
        current_boxes = []
        drawing = False
        start = None
        end = None
        cursor = None
        start_display = None
        break_loop = False

        while True:
            preview = None
            if drawing and start is not None and end is not None:
                preview = (start[0], start[1], end[0], end[1])
            canvas, _ = render_frame(
                image,
                current_boxes,
                classes,
                selected_id,
                index,
                len(images),
                preview=preview,
                cursor=cursor,
            )
            try:
                cv2.imshow(window, canvas)
                key = cv2.waitKey(20) & 0xFF
            except cv2.error:
                quit_requested = True
                break_loop = True
                break

            if key == ord("q") or key == 27:
                quit_requested = True
                break_loop = True
                break
            if key == ord("c") and current_boxes:
                current_boxes.pop()
            if key == ord("l"):
                selected_index = (selected_index + 1) % len(class_ids)
                selected_id = class_ids[selected_index]
            if ord("1") <= key <= ord("9"):
                slot = key - ord("1")
                if slot < len(class_ids):
                    selected_index = slot
                    selected_id = class_ids[selected_index]
            if key == ord("s"):
                destination_image = processed_dir / image_path.name
                if not write_image(destination_image, image):
                    print(f"Error: could not write image '{destination_image}'.")
                    continue
                lines = []
                for x1, y1, x2, y2, class_id in current_boxes:
                    center_x = ((x1 + x2) / 2.0) / width
                    center_y = ((y1 + y2) / 2.0) / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height
                    lines.append(
                        f"{class_id} {center_x:.6f} {center_y:.6f} "
                        f"{box_width:.6f} {box_height:.6f}"
                    )
                label_path = processed_dir / f"{image_path.stem}.txt"
                content = "\n".join(lines)
                if content:
                    content += "\n"
                label_path.write_text(content, encoding="utf-8")
                saved += 1
                print(
                    f"Saved {image_path.name} with {len(current_boxes)} boxes "
                    f"to '{processed_dir}'."
                )
                if remove_source:
                    image_path.unlink(missing_ok=True)
                break
            if key == ord("b"):
                background_image = background_dir / image_path.name
                destination_image = processed_dir / image_path.name
                if not write_image(background_image, image):
                    print(f"Error: could not write image '{background_image}'.")
                    continue
                if not write_image(destination_image, image):
                    print(f"Error: could not write image '{destination_image}'.")
                    continue
                label_path = processed_dir / f"{image_path.stem}.txt"
                label_path.write_text("", encoding="utf-8")
                background += 1
                print(f"Saved {image_path.name} as background.")
                if remove_source:
                    image_path.unlink(missing_ok=True)
                break
            if key == ord("n"):
                skipped += 1
                print(f"Skipped {image_path.name}.")
                break
            try:
                visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                visible = 1.0
            if visible < 1:
                quit_requested = True
                break_loop = True
                break

        if break_loop:
            break

    cv2.destroyAllWindows()
    print(
        f"Labeling finished: {saved} saved, {background} backgrounds, "
        f"{skipped} skipped."
    )
    if quit_requested:
        print("Session closed early.")
    return {"saved": saved, "background": background, "skipped": skipped}


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "label",
        help="Label frames for YOLO with multi-class support.",
    )
    parser.add_argument("--source", default=str(RAW_DIR))
    parser.add_argument("--processed", default=str(PROCESSED_DIR))
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--background", default=str(BACKGROUND_DIR))
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class names. Overrides data.yaml.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Delete labeled frames from the source folder.",
    )
    parser.add_argument(
        "--window",
        default="YOLO CRAFT - Labeler",
    )
    parser.set_defaults(func=run)
    return parser


def run(args):
    data_path = Path(args.data)
    override = parse_class_override(args.classes)
    classes = load_classes(data_path, override)

    if override or not data_path.exists():
        write_data_yaml(data_path, Path(args.dataset), classes)

    print("Classes: " + ", ".join(f"{key}={value}" for key, value in classes.items()))
    print("Controls: [s] Save  [b] Background  [c] Undo  [n] Skip  [l] Class  [q] Quit")
    label_images(
        source_dir=args.source,
        processed_dir=args.processed,
        background_dir=args.background,
        classes=classes,
        remove_source=args.remove_source,
        window=args.window,
    )
    split_dataset(
        dataset_dir=args.dataset,
        processed_dir=args.processed,
        classes=classes,
        train_ratio=args.train_ratio,
        seed=args.seed,
        data_path=data_path,
    )
    return 0
