import shutil
from pathlib import Path

from ultralytics import YOLO

from .common import (
    EXPORTS_DIR,
    MODELS_DIR,
    TRAIN_DIR,
    ensure_runtime_dirs,
    resolve_existing,
)

FORMATS = ("onnx", "tflite", "coreml", "engine")


def copy_artifact(exported, output_dir):
    source = Path(exported)
    if not source.exists():
        return None
    destination = Path(output_dir) / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    return destination


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "export",
        help="Export a YOLO model for edge and universal deployment.",
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument(
        "--format",
        default="all",
        choices=(*FORMATS, "all"),
        help="Target format. Use 'all' for every supported format.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--workspace", type=float, default=4.0)
    parser.add_argument("--output-dir", default=str(EXPORTS_DIR))
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
        print("Error: no model weights were found.")
        print("Pass --weights with a checkpoint path, for example best.pt.")
        return 1

    formats = FORMATS if args.format == "all" else (args.format,)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {weights}")
    print(f"Formats: {', '.join(formats)}")
    print(f"Artifacts: {output_dir.resolve()}")

    exported_artifacts = []
    failures = []

    for target in formats:
        print(f"Exporting {target}.")
        try:
            model = YOLO(str(weights))
            exported = model.export(
                format=target,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                half=args.half,
                int8=args.int8,
                dynamic=args.dynamic,
                simplify=args.simplify,
                opset=args.opset,
                workspace=args.workspace,
            )
            artifact = copy_artifact(exported, output_dir)
            if artifact is None:
                print(f"Export finished but no artifact was found for {target}.")
                failures.append(target)
                continue
            exported_artifacts.append(artifact)
            print(f"Exported {target}: {artifact}")
        except Exception as error:
            failures.append(target)
            print(f"Export failed for {target}: {error}")

    print(f"Completed {len(exported_artifacts)} of {len(formats)} exports.")
    if failures:
        print("Failed formats: " + ", ".join(failures))
        print("Install the format-specific dependencies and try again.")
        return 1
    return 0
