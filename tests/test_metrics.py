import csv

import numpy as np
import pytest

from yolo_craft.metrics import (
    generate_sample,
    parse_results,
    plot_curves,
    series_for,
    summarize,
)


def write_results(path, epochs=4):
    fieldnames = [
        "epoch",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            writer.writerow(
                {
                    "epoch": epoch,
                    "metrics/mAP50(B)": 0.2 + epoch * 0.1,
                    "metrics/mAP50-95(B)": 0.1 + epoch * 0.05,
                    "val/box_loss": 1.0 - epoch * 0.1,
                    "val/cls_loss": 2.0 - epoch * 0.2,
                }
            )


def test_parse_results_reads_numeric_columns(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_results(csv_path)
    data = parse_results(csv_path)
    assert data["epoch"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert data["columns"]["metrics/mAP50(B)"][-1] == pytest.approx(0.6)
    assert series_for(data, ["val/box_loss"])[-1] == pytest.approx(0.6)


def test_summarize_reports_best_epoch(tmp_path):
    csv_path = tmp_path / "results.csv"
    write_results(csv_path)
    summary = summarize(parse_results(csv_path))
    assert summary["epochs"] == 4
    assert summary["best_epoch"] == 4
    assert summary["best_map50_95"] == pytest.approx(0.3)
    assert summary["final_map50"] == pytest.approx(0.6)


def test_parse_results_ignores_bad_csv(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("not,a,results\nfile\n", encoding="utf-8")
    data = parse_results(csv_path)
    assert data["epoch"].size == 0


def test_sample_and_plot(tmp_path):
    data = generate_sample()
    assert data["epoch"].size == 60
    output = tmp_path / "curves.png"
    plot_curves(data, "Test", "Test subtitle", output)
    assert output.exists()
    assert output.stat().st_size > 1000
    assert np.isfinite(data["columns"]["metrics/mAP50(B)"]).all()
