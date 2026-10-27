import math
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from .common import (
    IMAGE_SUFFIXES,
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
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use test-time augmentation for slower but stronger recall.",
    )
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
    print(f"Classes: {', '.join(str(name) for name in model.names.values())}")

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Error: could not open source '{source_label(source)}'.")
        return 1

    writer = None
    save_path = None
    save_image = False
    if args.save:
        save_path = Path(args.save)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".jpg" if image_source else ".mp4")
        save_image = image_source and save_path.suffix.lower() in IMAGE_SUFFIXES
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

            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
            frame_index += 1

            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                augment=args.augment,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()
            boxes = result.boxes
            count = 0 if boxes is None else len(boxes)
            top = 0.0 if count == 0 else float(boxes.conf.max().item())
            top_class = ""
            if count:
                best_index = int(boxes.conf.argmax())
                top_class = str(model.names[int(boxes.cls[best_index])])
            if image_source:
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

            if image_source:
                if count:
                    labels = ", ".join(
                        f"{model.names[int(class_id)]} {float(score):.2f}"
                        for class_id, score in zip(
                            boxes.cls.tolist(), boxes.conf.tolist()
                        )
                    )
                    print(f"Detections: {labels}")
                else:
                    print(
                        f"No detections above confidence {args.conf:.2f}. "
                        "Try --conf, --imgsz, --augment, or a model trained "
                        "for this object."
                    )

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
        kind = "image" if save_image else "video"
        print(f"Saved annotated {kind}: {save_path}")
    print(f"Processed {frame_index} frames.")
    return 0
