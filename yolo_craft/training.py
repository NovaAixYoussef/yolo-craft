import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from .common import DATA_YAML, MODELS_DIR, RUNS_DIR, ensure_runtime_dirs


def resolve_model(model_name):
    path = Path(model_name)
    if path.exists():
        return str(path)
    if path.name == "yolov8n.pt":
        target = MODELS_DIR / path.name
        if target.exists():
            return str(target)
        print("Baseline weights not found. Downloading yolov8n.pt.")
        YOLO(path.name)
        downloaded = Path(path.name)
        if downloaded.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), str(target))
            return str(target)
    return str(path)


def train_model(
    data_path=None,
    model_name=None,
    epochs=100,
    imgsz=640,
    batch=16,
    patience=25,
    device="",
    project=None,
    name="train",
    seed=42,
    cos_lr=True,
    close_mosaic=10,
    box=7.5,
    cls=0.5,
    fliplr=0.5,
    mosaic=1.0,
):
    ensure_runtime_dirs()
    data_path = Path(data_path) if data_path else DATA_YAML
    model_name = model_name or str(MODELS_DIR / "yolov8n.pt")
    project = str(Path(project or RUNS_DIR).resolve())

    print("Initializing YOLO training.")
    print(f"Data: {data_path}")
    print(f"Base model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Early-stopping patience: {patience}")
    print(f"Cosine learning rate: {cos_lr}")
    print(f"Close mosaic epochs: {close_mosaic}")
    print(f"Box loss gain: {box}")
    print(f"Class loss gain: {cls}")
    print(f"Horizontal flip: {fliplr}")
    print(f"Mosaic: {mosaic}")
    print(f"Output: {Path(project) / name}")

    if not data_path.exists():
        print(f"Error: configuration file '{data_path}' was not found.")
        return None

    model_source = resolve_model(model_name)
    if not Path(model_source).exists() and Path(model_source).name != "yolov8n.pt":
        print(f"Error: base model '{model_source}' was not found.")
        return None

    model = YOLO(model_source)
    results = model.train(
        data=str(data_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        project=project,
        name=name,
        seed=seed,
        cos_lr=cos_lr,
        close_mosaic=close_mosaic,
        box=box,
        cls=cls,
        fliplr=fliplr,
        mosaic=mosaic,
    )
    print("Training completed.")
    return results


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "train",
        help="Fine-tune a YOLO model.",
    )
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--model", default=str(MODELS_DIR / "yolov8n.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default=str(RUNS_DIR))
    parser.add_argument("--name", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cos-lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a cosine learning-rate schedule.",
    )
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.set_defaults(func=run)
    return parser


def run(args):
    result = train_model(
        data_path=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        box=args.box,
        cls=args.cls,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
    )
    return 0 if result is not None else 1
