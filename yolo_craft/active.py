import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from .common import (
    BACKGROUND_DIR,
    DATA_YAML,
    DATASET_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    TRAIN_DIR,
    collect_images,
    ensure_runtime_dirs,
    is_video_file,
    next_capture_path,
    parse_source,
    resolve_existing,
    source_label,
)
from .labeling import (
    label_images,
    load_classes,
    parse_class_override,
    split_dataset,
    write_data_yaml,
)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "active",
        help="Capture hard frames, label them, and rebuild the dataset.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Trained checkpoint. Defaults to outputs/runs/train/weights/best.pt.",
    )
    parser.add_argument("--source", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--uncertain-conf", type=float, default=0.5)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--frames", default=str(RAW_DIR))
    parser.add_argument("--processed", default=str(PROCESSED_DIR))
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--background", default=str(BACKGROUND_DIR))
    parser.add_argument("--classes", default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-label", action="store_true")
    parser.add_argument("--window", default="YOLO CRAFT - Active Loop")
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
        print("Error: trained model not found.")
        print("Train a model first or pass --weights with a checkpoint path.")
        return 1

    source = parse_source(args.source)
    frames_dir = Path(args.frames)
    frames_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    classes = load_classes(data_path, parse_class_override(args.classes))
    if args.classes or not data_path.exists():
        write_data_yaml(data_path, Path(args.dataset), classes)

    saved = len(collect_images(frames_dir))
    last_save = 0.0
    frame_index = 0

    print(f"Loading model: {weights}")
    print(f"Source: {source_label(source)}")
    print("Classes: " + ", ".join(f"{key}={value}" for key, value in classes.items()))
    print("Controls: [q] finish capture and start labeling")

    model = YOLO(str(weights))
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Error: could not open source '{source_label(source)}'.")
        return 1

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if args.loop and is_video_file(source):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("Source finished or disconnected.")
                break

            frame_index += 1
            if args.max_frames > 0 and frame_index > args.max_frames:
                break

            results = model.predict(frame, conf=args.conf, verbose=False)
            result = results[0]
            boxes = result.boxes
            uncertain = len(boxes) == 0
            if not uncertain:
                for confidence in boxes.conf:
                    if float(confidence) < args.uncertain_conf:
                        uncertain = True
                        break

            now = time.monotonic()
            if uncertain and now - last_save >= args.cooldown:
                path = next_capture_path(frames_dir, "uncertain")
                if cv2.imwrite(str(path), frame):
                    saved += 1
                    last_save = now

            annotated = result.plot()
            status = (
                f"Saved {saved} | Frame {frame_index} | "
                f"Uncertain {args.uncertain_conf:.2f}"
            )
            cv2.rectangle(
                annotated,
                (0, 0),
                (annotated.shape[1], 34),
                (15, 17, 21),
                -1,
            )
            cv2.putText(
                annotated,
                status,
                (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (245, 247, 250),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(args.window, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(args.window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Capture finished with {saved} frames in '{frames_dir}'.")

    if args.no_label:
        print("Labeling skipped.")
        return 0

    label_images(
        source_dir=frames_dir,
        processed_dir=args.processed,
        background_dir=args.background,
        classes=classes,
        remove_source=True,
        window="YOLO CRAFT - Active Loop Labeler",
    )
    split_dataset(
        dataset_dir=args.dataset,
        processed_dir=args.processed,
        classes=classes,
        data_path=data_path,
    )
    return 0
