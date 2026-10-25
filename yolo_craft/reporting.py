import os
from datetime import datetime, timezone
from pathlib import Path

from .common import (
    BENCHMARKS_DIR,
    DATA_YAML,
    DATASET_DIR,
    EXPORTS_DIR,
    IMAGE_SUFFIXES,
    REPORTS_DIR,
    RUNS_DIR,
    TRAIN_DIR,
    load_yaml,
    normalize_names,
)
from .metrics import (
    configure_style,
    generate_sample,
    parse_results,
    plot_curves,
    summarize,
)


def format_float(value, digits=4):
    if value == "" or value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def dataset_stats(dataset_dir):
    dataset_dir = Path(dataset_dir)
    payload = load_yaml(dataset_dir / "data.yaml")
    names = normalize_names(payload.get("names", {}))
    stats = {"train": 0, "val": 0, "boxes": 0, "per_class": {}, "names": names}
    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        if image_dir.exists():
            stats[split] = len(
                [
                    path
                    for path in image_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_SUFFIXES
                ]
            )
        if not label_dir.exists():
            continue
        for label_path in label_dir.glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                stats["boxes"] += 1
                stats["per_class"][class_id] = (
                    stats["per_class"].get(class_id, 0) + 1
                )
    return stats


def benchmark_stats(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    before = payload.get("before", {})
    after = payload.get("after", {})

    def top_confidence(section):
        detections = section.get("detections", [])
        if not detections:
            return 0.0
        return max(float(item.get("confidence", 0.0)) for item in detections)

    return {
        "image": payload.get("image", ""),
        "conf": payload.get("conf", ""),
        "iou": payload.get("iou", ""),
        "before_count": before.get("count", 0),
        "after_count": after.get("count", 0),
        "before_top": top_confidence(before),
        "after_top": top_confidence(after),
        "before_mean": before.get("mean_confidence", 0.0),
        "after_mean": after.get("mean_confidence", 0.0),
    }


def relative_ref(target, report_path):
    target_path = Path(target).resolve()
    base = Path(report_path).resolve().parent
    try:
        return target_path.relative_to(base).as_posix()
    except ValueError:
        try:
            return Path(os.path.relpath(target_path, base)).as_posix()
        except ValueError:
            return target_path.as_posix()


def build_report(
    summary,
    args,
    sample,
    benchmark,
    dataset,
    train_args,
    curves_ref,
):
    run_dir_ref = Path(args.run_dir).as_posix()
    csv_ref = Path(args.csv).as_posix()
    lines = []
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append("# YOLO CRAFT - Training Report")
    lines.append("")
    lines.append(f"Generated: **{generated}**")
    lines.append("")
    lines.append(f"Run directory: `{run_dir_ref}`")
    lines.append("")
    if sample:
        lines.append(
            "> **Illustrative data.** No training results were found at "
            f"`{csv_ref}`. The generated sample curves are for illustration "
            "only. Run `python -m yolo_craft train` and then "
            "`python -m yolo_craft report` to replace this report with "
            "measured results."
        )
    else:
        lines.append(f"> **Measured data.** Parsed from `{csv_ref}`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Epochs recorded | {summary['epochs']} |")
    lines.append(f"| Best epoch | {summary['best_epoch'] or '-'} |")
    lines.append(
        f"| Best mAP@0.5:0.95 | {format_float(summary['best_map50_95'])} |"
    )
    lines.append(f"| Final mAP@0.5 | {format_float(summary['final_map50'])} |")
    lines.append(
        f"| Final mAP@0.5:0.95 | {format_float(summary['final_map50_95'])} |"
    )
    lines.append(
        f"| Final box loss | {format_float(summary['final_box_loss'])} |"
    )
    lines.append(
        f"| Final class loss | {format_float(summary['final_cls_loss'])} |"
    )
    lines.append("")
    lines.append("## Training curves")
    lines.append("")
    if sample:
        lines.append(f"Curves file: `{curves_ref}`")
    else:
        lines.append(f"![Training performance curves]({curves_ref})")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    if dataset["train"] or dataset["val"] or dataset["names"]:
        lines.append("| Item | Value |")
        lines.append("| --- | ---: |")
        lines.append(f"| Training images | {dataset['train']} |")
        lines.append(f"| Validation images | {dataset['val']} |")
        lines.append(f"| Labeled boxes | {dataset['boxes']} |")
        lines.append(f"| Classes | {len(dataset['names'])} |")
        lines.append("")
        if dataset["names"]:
            lines.append("| Class id | Name | Instances |")
            lines.append("| ---: | --- | ---: |")
            for class_id, name in dataset["names"].items():
                count = dataset["per_class"].get(class_id, 0)
                lines.append(f"| {class_id} | {name} | {count} |")
            lines.append("")
    else:
        lines.append(
            "No dataset found. Capture and label images with "
            "`python -m yolo_craft label` to populate this section."
        )
        lines.append("")
    if benchmark is not None:
        lines.append("## Before and after")
        lines.append("")
        lines.append(
            f"Test image: `{Path(benchmark['image']).as_posix()}` at "
            f"confidence `{benchmark['conf']}` and IoU `{benchmark['iou']}`."
        )
        lines.append("")
        lines.append("| Model | Detections | Top confidence | Mean confidence |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append(
            "| Baseline | "
            f"{benchmark['before_count']} | "
            f"{format_float(benchmark['before_top'], 3)} | "
            f"{format_float(benchmark['before_mean'], 3)} |"
        )
        lines.append(
            "| Fine-tuned | "
            f"{benchmark['after_count']} | "
            f"{format_float(benchmark['after_top'], 3)} | "
            f"{format_float(benchmark['after_mean'], 3)} |"
        )
        lines.append("")
    if train_args:
        keys = [
            "model",
            "data",
            "epochs",
            "imgsz",
            "batch",
            "patience",
            "optimizer",
            "lr0",
            "lrf",
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "cos_lr",
            "close_mosaic",
            "mosaic",
            "fliplr",
            "box",
            "cls",
            "device",
            "seed",
        ]
        rows = [(key, train_args[key]) for key in keys if key in train_args]
        if rows:
            lines.append("## Training configuration")
            lines.append("")
            lines.append("| Setting | Value |")
            lines.append("| --- | --- |")
            for key, value in rows:
                lines.append(f"| `{key}` | `{value}` |")
            lines.append("")
    weights = Path(args.run_dir) / "weights" / "best.pt"
    if weights.exists():
        lines.append("## Artifacts")
        lines.append("")
        lines.append(f"- Best checkpoint: `{weights}`")
        exports = sorted(EXPORTS_DIR.glob("*")) if EXPORTS_DIR.exists() else []
        for artifact in exports:
            lines.append(f"- Export: `{artifact}`")
        lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python -m yolo_craft train")
    lines.append("python -m yolo_craft report")
    lines.append("python -m yolo_craft benchmark --image <path>")
    lines.append("python -m yolo_craft export --format all")
    lines.append("```")
    lines.append("")
    lines.append("## Next steps")
    lines.append("")
    if sample:
        lines.append("- Train a model so this report reflects real measurements.")
    if summary["epochs"] and summary["best_epoch"] == summary["epochs"]:
        lines.append(
            "- The best epoch is the final epoch. Consider training longer."
        )
    lines.append(
        "- Add hard negatives and varied lighting to the next labeling round."
    )
    lines.append(
        "- Tune the confidence and IoU thresholds on a held-out validation set."
    )
    lines.append("- Export the best checkpoint for the target device.")
    lines.append("")
    return "\n".join(lines)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "report",
        help="Generate training curves and a Markdown evaluation report.",
    )
    parser.add_argument(
        "--csv",
        default=str(TRAIN_DIR / "results.csv"),
    )
    parser.add_argument(
        "--run-dir",
        default=str(TRAIN_DIR),
    )
    parser.add_argument("--dataset", default=str(DATASET_DIR))
    parser.add_argument(
        "--benchmark",
        default=str(BENCHMARKS_DIR / "metrics.json"),
    )
    parser.add_argument(
        "--curves",
        default=str(REPORTS_DIR / "training_curves.png"),
    )
    parser.add_argument("--report", default="REPORT.md")
    parser.add_argument("--sample", action="store_true")
    parser.set_defaults(func=run)
    return parser


def run(args):
    configure_style()
    csv_path = Path(args.csv)

    if csv_path.exists():
        data = parse_results(csv_path)
        if data["epoch"].size == 0:
            print(f"Error: no usable rows found in '{csv_path}'.")
            return 1
        sample = False
        title = "YOLO CRAFT - Training curves"
        subtitle = f"Source: {csv_path}"
    elif args.sample:
        data = generate_sample()
        sample = True
        title = "YOLO CRAFT - Illustrative training curves"
        subtitle = "No results.csv found. Sample data for documentation only."
    else:
        print(f"Error: results file not found at '{csv_path}'.")
        print(
            "Train a model first or pass --sample for illustrative curves."
        )
        return 1

    summary = summarize(data)
    plot_curves(data, title, subtitle, args.curves)
    dataset = dataset_stats(args.dataset)
    benchmark = benchmark_stats(args.benchmark)
    train_args = load_yaml(Path(args.run_dir) / "args.yaml")
    curves_ref = relative_ref(args.curves, args.report)
    report = build_report(
        summary,
        args,
        sample,
        benchmark,
        dataset,
        train_args,
        curves_ref,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved report: {report_path}")
    return 0
