import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

BACKGROUND = "#0b0d10"
SURFACE = "#12161c"
GRID = "#232a33"
TEXT = "#f5f7fa"
MUTED = "#9aa4b2"
AMBER = "#f59e0b"
BLUE = "#60a5fa"
GREEN = "#34d399"
PINK = "#f472b6"
PALETTE = ["#f59e0b", "#60a5fa", "#f472b6", "#34d399", "#a78bfa", "#fb923c"]

METRICS = {
    "mAP50": {
        "columns": ["metrics/mAP50(B)"],
        "title": "mAP@0.5",
        "ylabel": "mAP@0.5",
        "color": AMBER,
        "limit": (0.0, 1.0),
    },
    "mAP50-95": {
        "columns": ["metrics/mAP50-95(B)"],
        "title": "mAP@0.5:0.95",
        "ylabel": "mAP@0.5:0.95",
        "color": BLUE,
        "limit": (0.0, 1.0),
    },
    "box_loss": {
        "columns": ["val/box_loss", "train/box_loss"],
        "title": "Box Loss",
        "ylabel": "Loss",
        "color": GREEN,
        "limit": None,
    },
    "cls_loss": {
        "columns": ["val/cls_loss", "train/cls_loss"],
        "title": "Class Loss",
        "ylabel": "Loss",
        "color": PINK,
        "limit": None,
    },
}


def configure_style():
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": BACKGROUND,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "legend.labelcolor": TEXT,
        }
    )


def parse_results(csv_path):
    epochs = []
    columns = {}
    with Path(csv_path).open(
        "r", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        reader = csv.reader(handle)
        try:
            header = [cell.strip() for cell in next(reader)]
        except StopIteration:
            return {"epoch": np.array([]), "columns": {}}
        if "epoch" not in header:
            return {"epoch": np.array([]), "columns": {}}
        epoch_index = header.index("epoch")
        for row in reader:
            if not row or len(row) < len(header):
                continue
            try:
                epoch = float(row[epoch_index])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(epoch):
                continue
            epochs.append(epoch)
            for index, name in enumerate(header):
                if name == "epoch" or not name:
                    continue
                try:
                    value = float(row[index])
                except (TypeError, ValueError):
                    value = float("nan")
                columns.setdefault(name, []).append(value)
    result = {"epoch": np.array(epochs, dtype=float), "columns": {}}
    for name, values in columns.items():
        if len(values) == len(epochs):
            result["columns"][name] = np.array(values, dtype=float)
    return result


def series_for(data, candidates):
    for name in candidates:
        values = data["columns"].get(name)
        if values is not None and len(values) == len(data["epoch"]):
            return values
    return None


def generate_sample():
    rng = np.random.default_rng(7)
    epochs = np.arange(1, 61)
    progress = epochs / epochs.max()

    def learning_curve(start, end, noise):
        values = start + (end - start) * (1.0 - np.exp(-4.2 * progress))
        values = values + noise * rng.standard_normal(epochs.size)
        return np.clip(values, 0.0, 1.0)

    columns = {
        "metrics/mAP50(B)": learning_curve(0.18, 0.91, 0.012),
        "metrics/mAP50-95(B)": learning_curve(0.08, 0.62, 0.010),
        "val/box_loss": 1.92 * np.exp(-3.1 * progress)
        + 0.42
        + 0.015 * rng.standard_normal(epochs.size),
        "val/cls_loss": 2.65 * np.exp(-3.4 * progress)
        + 0.31
        + 0.018 * rng.standard_normal(epochs.size),
    }
    return {"epoch": epochs, "columns": columns}


def summarize(data):
    epoch = data["epoch"]
    map50 = series_for(data, METRICS["mAP50"]["columns"])
    map5095 = series_for(data, METRICS["mAP50-95"]["columns"])
    box_loss = series_for(data, METRICS["box_loss"]["columns"])
    cls_loss = series_for(data, METRICS["cls_loss"]["columns"])
    best_epoch = ""
    best_map = ""
    if map5095 is not None and map5095.size and np.isfinite(map5095).any():
        index = int(np.nanargmax(map5095))
        best_epoch = int(epoch[index])
        best_map = float(map5095[index])
    return {
        "epochs": int(epoch.size),
        "best_epoch": best_epoch,
        "best_map50_95": best_map,
        "final_map50": float(map50[-1]) if map50 is not None else "",
        "final_map50_95": float(map5095[-1]) if map5095 is not None else "",
        "final_box_loss": float(box_loss[-1]) if box_loss is not None else "",
        "final_cls_loss": float(cls_loss[-1]) if cls_loss is not None else "",
    }


def save_figure(fig, path, dpi=170):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"Saved figure: {path}")


def plot_curves(data, title, subtitle, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
    fig.patch.set_facecolor(BACKGROUND)
    fig.suptitle(
        title, color=TEXT, fontsize=17, fontweight="bold", x=0.055, ha="left"
    )
    fig.text(0.055, 0.925, subtitle, color=MUTED, fontsize=10, ha="left")

    for axis, (key, spec) in zip(axes.flat, METRICS.items()):
        axis.set_facecolor(SURFACE)
        for spine in axis.spines.values():
            spine.set_color(GRID)
        axis.tick_params(colors=MUTED)
        axis.grid(True, linestyle="--", linewidth=0.6)
        values = series_for(data, spec["columns"])
        if values is None:
            axis.set_axis_off()
            axis.text(
                0.5,
                0.5,
                f"{spec['title']}\nnot logged in the run",
                ha="center",
                va="center",
                color=MUTED,
                fontsize=11,
            )
            continue
        axis.plot(data["epoch"], values, color=spec["color"], linewidth=2.3)
        if len(values) <= 30:
            axis.scatter(
                data["epoch"],
                values,
                color=spec["color"],
                s=12,
                alpha=0.65,
                edgecolors="none",
            )
        axis.set_title(spec["title"], pad=10)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(spec["ylabel"])
        if spec["limit"]:
            axis.set_ylim(*spec["limit"])

    fig.subplots_adjust(
        left=0.06,
        right=0.97,
        top=0.84,
        bottom=0.08,
        hspace=0.38,
        wspace=0.18,
    )
    save_figure(fig, output_path, dpi=220)
