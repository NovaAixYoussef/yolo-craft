import csv
import json
from pathlib import Path

import yaml

from yolo_craft.cli import main


def write_results(path, epochs=3):
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
                    "metrics/mAP50(B)": 0.5,
                    "metrics/mAP50-95(B)": 0.3,
                    "val/box_loss": 0.8,
                    "val/cls_loss": 1.1,
                }
            )


def test_report_uses_measured_results(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    csv_path = run_dir / "results.csv"
    write_results(csv_path)
    (run_dir / "args.yaml").write_text(
        yaml.safe_dump({"epochs": 3, "imgsz": 640}),
        encoding="utf-8",
    )

    dataset = tmp_path / "dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "labels" / "val").mkdir(parents=True)
    (dataset / "data.yaml").write_text(
        yaml.safe_dump({"names": {0: "joystick"}}),
        encoding="utf-8",
    )
    (dataset / "images" / "train" / "a.jpg").write_bytes(b"image")
    (dataset / "labels" / "train" / "a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    benchmark = tmp_path / "metrics.json"
    benchmark.write_text(
        json.dumps(
            {
                "image": str(tmp_path / "test.jpg"),
                "conf": 0.25,
                "iou": 0.7,
                "before": {"count": 0, "detections": []},
                "after": {
                    "count": 1,
                    "mean_confidence": 0.4,
                    "detections": [{"confidence": 0.4}],
                },
            }
        ),
        encoding="utf-8",
    )

    curves = tmp_path / "output" / "curves.png"
    report = tmp_path / "reports" / "REPORT.md"
    exit_code = main(
        [
            "report",
            "--csv",
            str(csv_path),
            "--run-dir",
            str(run_dir),
            "--dataset",
            str(dataset),
            "--benchmark",
            str(benchmark),
            "--curves",
            str(curves),
            "--report",
            str(report),
        ]
    )

    assert exit_code == 0
    assert curves.exists()
    text = report.read_text(encoding="utf-8")
    assert "Measured data" in text
    assert "docs/" not in text
    assert "output/curves.png" in text
    assert "| Training images | 1 |" in text
    assert "Before and after" in text
    assert "| Fine-tuned | 1 |" in text


def test_report_sample_without_dataset(tmp_path):
    report = tmp_path / "REPORT.md"
    exit_code = main(
        [
            "report",
            "--sample",
            "--csv",
            str(tmp_path / "missing.csv"),
            "--run-dir",
            str(tmp_path / "run"),
            "--dataset",
            str(tmp_path / "dataset"),
            "--benchmark",
            str(tmp_path / "missing.json"),
            "--curves",
            str(tmp_path / "curves.png"),
            "--report",
            str(report),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "curves.png").exists()
    text = report.read_text(encoding="utf-8")
    assert "Illustrative data" in text
    assert "No dataset found" in text
    assert "| Training images | 0 |" not in text


def test_report_requires_results_without_sample(tmp_path):
    exit_code = main(
        [
            "report",
            "--csv",
            str(tmp_path / "missing.csv"),
            "--run-dir",
            str(tmp_path / "run"),
            "--dataset",
            str(tmp_path / "dataset"),
            "--benchmark",
            str(tmp_path / "missing.json"),
            "--curves",
            str(tmp_path / "curves.png"),
            "--report",
            str(tmp_path / "REPORT.md"),
        ]
    )
    assert exit_code == 1
    assert not (tmp_path / "REPORT.md").exists()
