import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from .common import (
    BENCHMARKS_DIR,
    DATASET_DIR,
    MODELS_DIR,
    RUNS_DIR,
    TRAIN_DIR,
    collect_images,
    resolve_existing,
)
from .metrics import (
    BACKGROUND,
    METRICS,
    MUTED,
    PALETTE,
    SURFACE,
    TEXT,
    configure_style,
    generate_sample,
    parse_results,
    plot_curves,
    save_figure,
    series_for,
)


def find_runs(runs_dir):
    root = Path(runs_dir)
    if not root.exists():
        return []

    runs = []
    for csv_path in sorted(root.rglob("results.csv")):
        if "weights" in csv_path.parts:
            continue
        data = parse_results(csv_path)
        if data["epoch"].size == 0:
            continue
        try:
            label = csv_path.parent.relative_to(root).as_posix()
        except ValueError:
            label = csv_path.parent.name
        if label == ".":
            label = csv_path.parent.name
        runs.append({"label": label, "path": csv_path, "data": data})
    return runs


def slugify(value):
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def plot_single_run(run, out_dir):
    data = run["data"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    fig.suptitle(
        f"Training curves - {run['label']}",
        fontsize=15,
        fontweight="bold",
        color=TEXT,
    )

    for ax, (key, spec) in zip(axes.flat, METRICS.items()):
        values = series_for(data, spec["columns"])
        if values is None:
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                f"{spec['title']}\nnot logged in results.csv",
                ha="center",
                va="center",
                color=MUTED,
            )
            continue

        ax.plot(
            data["epoch"],
            values,
            color=spec["color"],
            linewidth=2.2,
            marker="o" if len(values) <= 24 else None,
            markersize=3.5,
        )
        ax.set_title(spec["title"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, linestyle="--", linewidth=0.6)
        ax.set_facecolor(SURFACE)
        if spec["limit"]:
            ax.set_ylim(*spec["limit"])

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, Path(out_dir) / f"{slugify(run['label'])}_curves.png")


def plot_metric_comparison(runs, key, out_dir):
    spec = METRICS[key]
    fig, ax = plt.subplots(figsize=(10, 5.6))

    for index, run in enumerate(runs):
        values = series_for(run["data"], spec["columns"])
        if values is None:
            continue
        color = PALETTE[index % len(PALETTE)]
        ax.plot(
            run["data"]["epoch"],
            values,
            color=color,
            linewidth=2.4,
            marker="o" if len(values) <= 24 else None,
            markersize=3.5,
            label=run["label"],
        )

    ax.set_title(f"{spec['title']} comparison across runs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(spec["ylabel"])
    ax.grid(True, linestyle="--", linewidth=0.6)
    ax.set_facecolor(SURFACE)
    if spec["limit"]:
        ax.set_ylim(*spec["limit"])
    if len(runs) > 1:
        ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, Path(out_dir) / f"{key}.png")


def write_summary(runs, out_dir):
    rows = []
    for run in runs:
        data = run["data"]
        map50 = series_for(data, METRICS["mAP50"]["columns"])
        map5095 = series_for(data, METRICS["mAP50-95"]["columns"])
        box_loss = series_for(data, METRICS["box_loss"]["columns"])
        cls_loss = series_for(data, METRICS["cls_loss"]["columns"])

        best_epoch = ""
        best_map = ""
        if map5095 is not None and map5095.size and np.isfinite(map5095).any():
            index = int(np.nanargmax(map5095))
            best_epoch = int(data["epoch"][index])
            best_map = float(map5095[index])

        rows.append(
            {
                "run": run["label"],
                "epochs": int(data["epoch"].size),
                "best_epoch": best_epoch,
                "best_map50_95": best_map,
                "final_map50": float(map50[-1]) if map50 is not None else "",
                "final_map50_95": (
                    float(map5095[-1]) if map5095 is not None else ""
                ),
                "final_box_loss": (
                    float(box_loss[-1]) if box_loss is not None else ""
                ),
                "final_cls_loss": (
                    float(cls_loss[-1]) if cls_loss is not None else ""
                ),
            }
        )

    if not rows:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path}")


def generate_sample_curves(out_dir):
    data = generate_sample()
    plot_curves(
        data,
        "Illustrative training curves - no runs found",
        "Sample data generated by the benchmark command.",
        Path(out_dir) / "sample_curves.png",
    )


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def run_detection(weights, image, conf, iou):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    frame = np.ascontiguousarray(image[:, :, ::-1])
    results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
    result = results[0]

    plotted = result.plot(line_width=2, font_size=11)
    if plotted.ndim == 3 and plotted.shape[2] == 3:
        plotted_rgb = plotted[:, :, ::-1].copy()
    else:
        plotted_rgb = plotted

    detections = []
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0])
            detections.append(
                {
                    "class": result.names[class_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "box_xyxy": [
                        round(float(value), 2) for value in box.xyxy[0]
                    ],
                }
            )

    confidences = [item["confidence"] for item in detections]
    speed = {
        key: round(float(value), 2) for key, value in result.speed.items()
    }

    return {
        "weights": str(weights),
        "detections": detections,
        "count": len(detections),
        "mean_confidence": (
            round(float(np.mean(confidences)), 4) if confidences else 0.0
        ),
        "speed_ms": speed,
        "plotted": plotted_rgb,
    }


def load_fonts():
    try:
        regular = font_manager.findfont("DejaVu Sans")
        bold = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight="bold")
        )
        return (
            ImageFont.truetype(regular, 18),
            ImageFont.truetype(bold, 22),
            ImageFont.truetype(regular, 13),
        )
    except Exception:
        fallback = ImageFont.load_default()
        return fallback, fallback, fallback


def compose_comparison(
    before,
    after,
    before_title,
    after_title,
    before_count,
    after_count,
    image_name,
    out_path,
):
    panel_height = min(640, max(before.shape[0], after.shape[0]))
    margin = 22
    gap = 22
    header = 66
    footer = 46

    def resize_panel(image):
        panel = Image.fromarray(image)
        scale = panel_height / panel.height
        width = max(1, int(panel.width * scale))
        return panel.resize((width, panel_height), Image.LANCZOS)

    left = resize_panel(before)
    right = resize_panel(after)
    width = margin * 2 + left.width + gap + right.width
    height = margin * 2 + header + panel_height + footer

    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font_regular, font_bold, font_small = load_fonts()

    def draw_panel(panel, x, title, count, accent):
        draw.rectangle(
            [x, margin, x + panel.width, margin + header - 10],
            fill=SURFACE,
        )
        draw.rectangle(
            [x, margin, x + 5, margin + header - 10],
            fill=accent,
        )
        draw.text((x + 18, margin + 12), title, font=font_bold, fill=TEXT)
        label = (
            f"{count} detection"
            if count == 1
            else f"{count} detections"
        )
        box = draw.textbbox((0, 0), label, font=font_regular)
        draw.text(
            (x + panel.width - (box[2] - box[0]) - 18, margin + 15),
            label,
            font=font_regular,
            fill=accent,
        )
        canvas.paste(panel, (x, margin + header - 10))

    draw_panel(left, margin, before_title, before_count, "#9ca3af")
    draw_panel(
        right,
        margin + left.width + gap,
        after_title,
        after_count,
        PALETTE[0],
    )

    caption = (
        "YOLO CRAFT  |  benchmark  |  "
        f"{image_name}  |  "
        "before: base model  ->  after: fine-tuned model"
    )
    draw.text(
        (margin, height - footer + 6),
        caption,
        font=font_small,
        fill=MUTED,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)
    print(f"Saved comparison image: {out_path}")


def choose_image(explicit, dataset_dir):
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    for folder in (
        Path(dataset_dir) / "images" / "val",
        Path(dataset_dir) / "images" / "train",
    ):
        images = collect_images(folder)
        if images:
            return images[0]
    return None


def run_demo(args, out_dir):
    image_path = choose_image(args.image, args.dataset)
    if image_path is None:
        print("Error: test image not found.")
        print("Pass --image <path> or label a dataset first.")
        return 1

    baseline_path = resolve_existing(
        args.baseline,
        [str(MODELS_DIR / "yolov8n.pt"), "yolov8n.pt"],
    )
    finetuned_path = resolve_existing(
        args.finetuned,
        [
            str(TRAIN_DIR / "weights" / "best.pt"),
            str(MODELS_DIR / "best.pt"),
        ],
    )

    if baseline_path is None:
        print("Error: baseline weights not found. Pass --baseline <path>.")
        return 1
    if finetuned_path is None:
        print("Error: fine-tuned weights not found. Pass --finetuned <path>.")
        return 1

    image = load_rgb(image_path)
    print(f"Test image: {image_path}")
    print(f"Baseline weights: {baseline_path}")
    print(f"Fine-tuned weights: {finetuned_path}")

    before = run_detection(baseline_path, image, args.conf, args.iou)
    after = run_detection(finetuned_path, image, args.conf, args.iou)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    before_image = Image.fromarray(before["plotted"])
    after_image = Image.fromarray(after["plotted"])
    before_image.save(out_dir / "before.png")
    after_image.save(out_dir / "after.png")
    print(f"Saved figure: {out_dir / 'before.png'}")
    print(f"Saved figure: {out_dir / 'after.png'}")

    compose_comparison(
        before["plotted"],
        after["plotted"],
        "BEFORE  -  base model",
        "AFTER  -  fine-tuned model",
        before["count"],
        after["count"],
        image_path.name,
        out_dir / "before_after.png",
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image": str(image_path),
        "conf": args.conf,
        "iou": args.iou,
        "before": {
            key: value for key, value in before.items() if key != "plotted"
        },
        "after": {
            key: value for key, value in after.items() if key != "plotted"
        },
        "delta_detections": after["count"] - before["count"],
        "delta_mean_confidence": round(
            after["mean_confidence"] - before["mean_confidence"], 4
        ),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"Saved metrics: {metrics_path}")

    print(
        f"Baseline: {before['count']} detections | "
        f"Fine-tuned: {after['count']} detections"
    )
    return 0


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "benchmark",
        help="Plot training runs and compare two checkpoints on one image.",
    )
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--output-dir", default=str(BENCHMARKS_DIR))
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument(
        "--image",
        default=None,
        help="Image for the before/after comparison.",
    )
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--finetuned", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--skip-curves", action="store_true")
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument(
        "--sample-curves",
        action="store_true",
        help="Write illustrative curves when no runs are found.",
    )
    parser.set_defaults(func=run)
    return parser


def run(args):
    configure_style()
    out_dir = Path(args.output_dir)
    curves_dir = out_dir / "curves"
    exit_code = 0

    if not args.skip_curves:
        runs = find_runs(args.runs_dir)
        if runs:
            print(f"Found {len(runs)} run(s) in '{args.runs_dir}'.")
            for run_data in runs:
                plot_single_run(run_data, curves_dir)
            for key in METRICS:
                plot_metric_comparison(runs, key, curves_dir)
            write_summary(runs, curves_dir)
        elif args.sample_curves:
            print("No training runs found. Writing illustrative sample curves.")
            generate_sample_curves(curves_dir)
        else:
            print(
                f"No results.csv found under '{args.runs_dir}'. "
                "Train a model first or pass --sample-curves."
            )

    if not args.skip_demo:
        exit_code = run_demo(args, out_dir)

    return exit_code
