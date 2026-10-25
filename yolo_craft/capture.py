import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from .common import (
    MODELS_DIR,
    RAW_DIR,
    collect_images,
    ensure_runtime_dirs,
    is_video_file,
    next_capture_path,
    parse_source,
    resolve_existing,
    source_label,
)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "capture",
        help="Capture frames where a YOLO model is uncertain.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index, video file path, or RTSP/HTTP stream URL.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Uncertainty model. Defaults to models/yolov8n.pt.",
    )
    parser.add_argument(
        "--output",
        default=str(RAW_DIR),
        help="Folder for captured frames.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--uncertain-conf",
        type=float,
        default=0.5,
        help="Frames with any box below this confidence are saved.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.0,
        help="Minimum seconds between automatic saves.",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window", default="YOLO CRAFT - Capture")
    parser.set_defaults(func=run)
    return parser


def run(args):
    ensure_runtime_dirs()
    weights = resolve_existing(
        args.weights,
        [str(MODELS_DIR / "yolov8n.pt"), "yolov8n.pt"],
    )
    if weights is None:
        print("Error: baseline model not found.")
        print("Place yolov8n.pt in models/ or pass --weights <path>.")
        return 1

    source = parse_source(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = len(collect_images(output_dir))
    auto_mode = True
    last_save = 0.0
    frame_index = 0

    print(f"Loading model: {weights}")
    print(f"Source: {source_label(source)}")
    print(f"Output: {output_dir.resolve()}")
    print("Controls: [s] save frame  [a] toggle auto capture  [q] quit")

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
            if auto_mode and uncertain and now - last_save >= args.cooldown:
                path = next_capture_path(output_dir, "uncertain")
                if cv2.imwrite(str(path), frame):
                    saved_count += 1
                    last_save = now
                else:
                    print(f"Error: could not write '{path}'.")

            annotated = result.plot()
            status = (
                f"Saved {saved_count} | Auto {'ON' if auto_mode else 'OFF'} | "
                f"Frame {frame_index} | Uncertain {args.uncertain_conf:.2f}"
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
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("s"):
                path = next_capture_path(output_dir, "manual")
                if cv2.imwrite(str(path), frame):
                    saved_count += 1
                    print(f"Saved manually: {path}")
                else:
                    print(f"Error: could not write '{path}'.")
            if key == ord("a"):
                auto_mode = not auto_mode
                state = "enabled" if auto_mode else "disabled"
                print(f"Automatic capture {state}.")
            if cv2.getWindowProperty(args.window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Session finished with {saved_count} images in '{output_dir}'.")
    return 0
