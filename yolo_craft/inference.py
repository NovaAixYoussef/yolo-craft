import json
import math
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from .common import (
    IMAGE_SUFFIXES,
    MODELS_DIR,
    TRAIN_DIR,
    collect_images,
    ensure_runtime_dirs,
    is_image_file,
    is_stream,
    is_video_file,
    parse_source,
    resolve_existing,
    source_label,
)


def sequence_values(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def class_name(names, class_id):
    class_id = int(class_id)
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def parse_class_filter(spec, names):
    if not spec:
        return None
    name_to_id = {
        str(name).strip().lower(): int(class_id)
        for class_id, name in names.items()
    }
    valid_ids = {int(class_id) for class_id in names}
    selected = []
    unknown = []
    for token in str(spec).replace(";", ",").split(","):
        value = token.strip()
        if not value:
            continue
        if value.lstrip("+-").isdigit():
            class_id = int(value)
        else:
            class_id = name_to_id.get(value.lower())
        if class_id is None or class_id not in valid_ids:
            unknown.append(value)
            continue
        if class_id not in selected:
            selected.append(class_id)
    if unknown:
        raise ValueError(
            "Unknown class filter value(s): " + ", ".join(unknown) + "."
        )
    if not selected:
        raise ValueError("Class filter did not contain any class names or IDs.")
    return selected


def detection_records(result, names):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    coordinates = sequence_values(boxes.xyxy)
    classes = sequence_values(boxes.cls)
    confidences = sequence_values(boxes.conf)
    records = []
    for box, class_id, score in zip(coordinates, classes, confidences):
        records.append(
            {
                "class_id": int(class_id),
                "class": class_name(names, class_id),
                "confidence": round(float(score), 6),
                "box": [round(float(value), 2) for value in box],
            }
        )
    return records


def detection_summary(records):
    if not records:
        return 0, "", 0.0
    best = max(records, key=lambda record: record["confidence"])
    return len(records), best["class"], float(best["confidence"])


def yolo_label_data(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], []
    return sequence_values(boxes.cls), sequence_values(boxes.xywhn)


def write_yolo_labels(folder, stem, class_ids, coordinates):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for class_id, coordinate in zip(class_ids, coordinates):
        x, y, width, height = coordinate
        lines.append(
            f"{int(class_id)} {float(x):.6f} {float(y):.6f} "
            f"{float(width):.6f} {float(height):.6f}"
        )
    path = folder / f"{stem}.txt"
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")
    return path


def save_detection_json(path, source, weights, names, frames):
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = [
        {"id": int(class_id), "name": class_name(names, class_id)}
        for class_id in sorted(names, key=lambda item: int(item))
    ]
    payload = {
        "source": source_label(source),
        "weights": str(weights),
        "classes": classes,
        "frames": frames,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def predict_frame(model, frame, args, class_ids):
    options = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "augment": args.augment,
        "verbose": False,
    }
    if class_ids is not None:
        options["classes"] = class_ids
    if args.device:
        options["device"] = args.device
    result = model.predict(frame, **options)[0]
    records = detection_records(result, model.names)
    annotated = result.plot(
        conf=not args.no_conf,
        labels=not args.no_labels,
        line_width=args.line_width,
    )
    return annotated, records, result


def draw_status(image, text):
    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 34),
        (15, 17, 21),
        -1,
    )
    cv2.putText(
        image,
        text,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 247, 250),
        1,
        cv2.LINE_AA,
    )
    return image


def annotate_frame(image, records, frame_index, started, still):
    count, top_class, top = detection_summary(records)
    if still:
        status = (
            f"Objects {count} | {top_class} {top:.2f}"
            if count
            else "Objects 0 | no detections"
        )
    else:
        elapsed = max(time.monotonic() - started, 1e-6)
        fps = frame_index / elapsed
        status = (
            f"Objects {count} | Top {top:.2f} | FPS {fps:.1f} | "
            f"Frame {frame_index}"
        )
    return draw_status(image, status)


def detection_message(records, confidence):
    if not records:
        return (
            f"No detections above confidence {confidence:.2f}. "
            "Try --conf, --imgsz, --augment, or a model trained for this object."
        )
    labels = ", ".join(
        f"{record['class']} {record['confidence']:.2f}" for record in records
    )
    return f"Detections: {labels}"


def label_stem(source, image_path, frame_index):
    if image_path is not None:
        return image_path.stem
    if is_image_file(source):
        return Path(source).stem
    if isinstance(source, str) and not is_stream(source):
        path = Path(source)
        if path.suffix:
            return f"{path.stem}_{frame_index:06d}"
    return f"frame_{frame_index:06d}"


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "test",
        help="Run a trained YOLO model on a webcam, video, image, folder, or stream.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Checkpoint. Defaults to outputs/runs/train/weights/best.pt.",
    )
    parser.add_argument("--source", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use test-time augmentation for slower but stronger recall.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class IDs or names to keep.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Inference device, for example cpu, 0, or 0,1.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Annotated output file or output folder for an image directory.",
    )
    parser.add_argument(
        "--save-txt",
        default=None,
        help="Directory for YOLO-format detection labels.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="JSON file with detections for every processed frame.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide class labels in annotated output.",
    )
    parser.add_argument(
        "--no-conf",
        action="store_true",
        help="Hide confidence values in annotated output.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=None,
        help="Box line width for annotated output.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless without opening a preview window.",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window", default="YOLO CRAFT - Test")
    parser.set_defaults(func=run)
    return parser


def run(args):
    ensure_runtime_dirs()
    weights = resolve_existing(
        args.weights,
        [
            str(TRAIN_DIR / "weights" / "best.pt"),
            str(MODELS_DIR / "best.pt"),
        ],
    )
    if weights is None:
        print("Error: model weights not found.")
        print("Train a model first or pass --weights <path>.")
        return 1

    if args.line_width is not None and args.line_width < 1:
        print("Error: --line-width must be at least 1.")
        return 1

    source = parse_source(args.source)
    source_path = Path(source) if isinstance(source, str) else None
    folder_source = bool(source_path and source_path.is_dir())
    image_source = is_image_file(source)
    images = []
    if folder_source:
        images = collect_images(source_path)
        if not images:
            print(f"Error: no images found in '{source_path}'.")
            return 1

    print(f"Loading model: {weights}")
    print(f"Source: {source_label(source)}")
    model = YOLO(str(weights))
    print(f"Classes: {', '.join(str(name) for name in model.names.values())}")

    try:
        class_ids = parse_class_filter(args.classes, model.names)
    except ValueError as error:
        print(f"Error: {error}")
        return 1
    if class_ids is not None:
        selected = ", ".join(
            f"{class_id}={class_name(model.names, class_id)}"
            for class_id in class_ids
        )
        print(f"Class filter: {selected}")

    save_path = None
    save_image = False
    folder_save_dir = None
    if folder_source and args.save:
        folder_save_dir = Path(args.save)
        if folder_save_dir.suffix:
            print("Error: --save must be a folder without a suffix for an image folder.")
            return 1
        folder_save_dir.mkdir(parents=True, exist_ok=True)
    elif args.save:
        save_path = Path(args.save)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".jpg" if image_source else ".mp4")
        save_image = image_source and save_path.suffix.lower() in IMAGE_SUFFIXES
        save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_txt:
        Path(args.save_txt).mkdir(parents=True, exist_ok=True)

    json_path = Path(args.save_json) if args.save_json else None
    if json_path is not None:
        if not json_path.suffix:
            json_path = json_path.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)

    capture = None
    if not folder_source:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            print(f"Error: could not open source '{source_label(source)}'.")
            return 1

    if image_source and not args.no_display:
        print("Press any key in the window to close it.")
    if folder_source:
        print(f"Image folder: {len(images)} images.")

    writer = None
    frame_index = 0
    started = time.monotonic()
    json_frames = []

    try:
        if folder_source:
            for image_path in images:
                if args.max_frames > 0 and frame_index >= args.max_frames:
                    break
                frame_index += 1
                frame = cv2.imread(str(image_path))
                if frame is None:
                    print(f"Skipping unreadable image: {image_path.name}")
                    continue

                annotated, records, result = predict_frame(
                    model, frame, args, class_ids
                )
                annotated = annotate_frame(
                    annotated, records, frame_index, started, True
                )

                if folder_save_dir is not None:
                    target = (
                        folder_save_dir
                        / f"{image_path.stem}_pred{image_path.suffix.lower()}"
                    )
                    if not cv2.imwrite(str(target), annotated):
                        print(f"Error: could not write image '{target}'.")

                if args.save_txt:
                    class_values, coordinates = yolo_label_data(result)
                    write_yolo_labels(
                        args.save_txt,
                        image_path.stem,
                        class_values,
                        coordinates,
                    )

                if json_path is not None:
                    json_frames.append(
                        {
                            "frame": frame_index,
                            "source": image_path.name,
                            "detections": records,
                        }
                    )

                print(
                    f"{image_path.name}: "
                    + detection_message(records, args.conf)
                )

                if not args.no_display:
                    cv2.imshow(args.window, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if cv2.getWindowProperty(
                        args.window, cv2.WND_PROP_VISIBLE
                    ) < 1:
                        break
        else:
            while True:
                ok, frame = capture.read()
                if not ok:
                    if args.loop and is_video_file(source):
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("Source finished or disconnected.")
                    break

                if args.max_frames > 0 and frame_index >= args.max_frames:
                    break
                frame_index += 1

                annotated, records, result = predict_frame(
                    model, frame, args, class_ids
                )
                annotated = annotate_frame(
                    annotated, records, frame_index, started, image_source
                )

                if save_path is not None:
                    if save_image:
                        if not cv2.imwrite(str(save_path), annotated):
                            print(f"Error: could not write image '{save_path}'.")
                            save_path = None
                    else:
                        if writer is None:
                            fps_source = capture.get(cv2.CAP_PROP_FPS)
                            if not fps_source or not math.isfinite(fps_source):
                                fps_source = 30.0
                            codec = (
                                "MJPG"
                                if save_path.suffix.lower() in {".avi", ".mjpg"}
                                else "mp4v"
                            )
                            writer = cv2.VideoWriter(
                                str(save_path),
                                cv2.VideoWriter_fourcc(*codec),
                                float(fps_source),
                                (annotated.shape[1], annotated.shape[0]),
                            )
                            if not writer.isOpened():
                                print(
                                    "Error: could not create output video "
                                    f"'{save_path}'."
                                )
                                writer = None
                                save_path = None
                        if writer is not None:
                            writer.write(annotated)

                if args.save_txt:
                    class_values, coordinates = yolo_label_data(result)
                    write_yolo_labels(
                        args.save_txt,
                        label_stem(source, None, frame_index),
                        class_values,
                        coordinates,
                    )

                if json_path is not None:
                    frame_record = {
                        "frame": frame_index,
                        "detections": records,
                    }
                    if image_source:
                        frame_record["source"] = Path(source).name
                    json_frames.append(frame_record)

                if image_source:
                    print(detection_message(records, args.conf))

                if not args.no_display:
                    cv2.imshow(args.window, annotated)
                    key = cv2.waitKey(0 if image_source else 1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if cv2.getWindowProperty(
                        args.window, cv2.WND_PROP_VISIBLE
                    ) < 1:
                        break
                if image_source:
                    break
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if json_path is not None:
        saved_json = save_detection_json(
            json_path,
            source,
            weights,
            model.names,
            json_frames,
        )
        print(f"Saved detections JSON: {saved_json}")

    if folder_save_dir is not None:
        print(f"Saved annotated images: {folder_save_dir}")
    elif save_path is not None:
        kind = "image" if save_image else "video"
        print(f"Saved annotated {kind}: {save_path}")

    if folder_source:
        print(f"Processed {frame_index} images.")
    else:
        print(f"Processed {frame_index} frames.")
    return 0
