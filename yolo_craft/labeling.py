import random
from pathlib import Path

import cv2
import yaml

from .common import (
    BACKGROUND_DIR,
    DATA_YAML,
    DATASET_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    collect_images,
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
        except yaml.YAMLError:
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


def class_color(class_id, classes):
    keys = list(classes.keys())
    index = keys.index(class_id) if class_id in keys else 0
    return BOX_COLORS[index % len(BOX_COLORS)]


def draw_box(image, box, classes):
    x1, y1, x2, y2, class_id = box
    color = class_color(class_id, classes)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    name = classes.get(class_id, str(class_id))
    label = f"{name} [{class_id}]"
    font_scale = 0.48
    thickness = 1
    size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    top = max(0, y1 - size[1] - baseline - 6)
    cv2.rectangle(
        image,
        (x1, top),
        (x1 + size[0] + 10, top + size[1] + baseline + 6),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 5, top + size[1] + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (10, 12, 16),
        thickness,
        cv2.LINE_AA,
    )


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

    class_ids = list(classes.keys())
    selected_index = 0
    selected_id = class_ids[selected_index]
    current_boxes = []
    drawing = False
    ix = iy = fx = fy = -1
    saved = 0
    background = 0
    skipped = 0

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, ix, iy, fx, fy
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            fx, fy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            fx, fy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            fx, fy = x, y
            if abs(fx - ix) > 5 and abs(fy - iy) > 5:
                x1, x2 = sorted((ix, fx))
                y1, y2 = sorted((iy, fy))
                current_boxes.append((x1, y1, x2, y2, selected_id))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    quit_requested = False
    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        height, width = image.shape[:2]
        current_boxes = []
        break_loop = False

        while True:
            canvas = image.copy()
            for box in current_boxes:
                draw_box(canvas, box, classes)

            if drawing:
                x1, x2 = sorted((ix, fx))
                y1, y2 = sorted((iy, fy))
                color = class_color(selected_id, classes)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

            header_height = 88
            cv2.rectangle(
                canvas,
                (0, 0),
                (canvas.shape[1], header_height),
                HEADER_COLOR,
                -1,
            )
            progress = int((index / len(images)) * 100)
            cv2.putText(
                canvas,
                f"Image {index + 1}/{len(images)}  {progress}%  "
                f"Boxes {len(current_boxes)}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
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

            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                break_loop = True
                quit_requested = True
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
                cv2.imwrite(str(destination_image), image)
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
                cv2.imwrite(str(background_dir / image_path.name), image)
                cv2.imwrite(str(processed_dir / image_path.name), image)
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
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break_loop = True
                quit_requested = True
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
