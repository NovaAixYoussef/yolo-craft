import math
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from .common import (
    MODELS_DIR,
    TRAIN_DIR,
    ensure_runtime_dirs,
    is_image_file,
    is_video_file,
    parse_source,
    resolve_existing,
    source_label,
)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "test",
        help="Run a trained YOLO model on a webcam, video, image, or stream.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Checkpoint. Defaults to outputs/runs/train/weights/best.pt.",
    )
    parser.add_argument("--source", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output video path for annotated frames.",
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

    source = parse_source(args.source)
    image_source = is_image_file(source)
    print(f"Loading model: {weights}")
    print(f"Source: {source_label(source)}")
    model = YOLO(str(weights))

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Error: could not open source '{source_label(source)}'.")
        return 1

    writer = None
    save_path = None
    if args.save:
        save_path = Path(args.save)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".mp4")
        save_path.parent.mkdir(parents=True, exist_ok=True)

    if image_source:
        print("Press any key in the window to close it.")

    frame_index = 0
    started = time.monotonic()

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

            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()
            boxes = result.boxes
            count = 0 if boxes is None else len(boxes)
            top = 0.0 if count == 0 else float(boxes.conf.max().item())
            if image_source:
                status = f"Objects {count} | Top {top:.2f}"
            else:
                elapsed = max(time.monotonic() - started, 1e-6)
                fps = frame_index / elapsed
                status = (
                    f"Objects {count} | Top {top:.2f} | FPS {fps:.1f} | "
                    f"Frame {frame_index}"
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

            if save_path is not None:
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
                        print(f"Error: could not create output video '{save_path}'.")
                        writer = None
                        save_path = None
                if writer is not None:
                    writer.write(annotated)

            cv2.imshow(args.window, annotated)
            key = cv2.waitKey(0 if image_source else 1) & 0xFF
            if key in (ord("q"), 27):
                break
            if image_source:
                break
            if cv2.getWindowProperty(args.window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if save_path is not None:
        print(f"Saved annotated video: {save_path}")
    print(f"Processed {frame_index} frames.")
    return 0
