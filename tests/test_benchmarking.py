import csv

from yolo_craft.benchmarking import (
    choose_image,
    find_runs,
    plot_single_run,
    slugify,
    write_summary,
)


def write_results(path, base=0.2):
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
        for epoch in range(1, 4):
            writer.writerow(
                {
                    "epoch": epoch,
                    "metrics/mAP50(B)": base + epoch * 0.1,
                    "metrics/mAP50-95(B)": base + epoch * 0.05,
                    "val/box_loss": 1.0 - epoch * 0.1,
                    "val/cls_loss": 2.0 - epoch * 0.2,
                }
            )


def test_find_runs_and_artifacts(tmp_path):
    runs_dir = tmp_path / "runs"
    first = runs_dir / "train"
    second = runs_dir / "experiment"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    write_results(first / "results.csv", 0.2)
    write_results(second / "results.csv", 0.1)

    runs = find_runs(runs_dir)
    assert [run["label"] for run in runs] == ["experiment", "train"]
    plot_single_run(runs[0], tmp_path / "artifacts")
    write_summary(runs, tmp_path / "artifacts")
    assert (tmp_path / "artifacts" / "experiment_curves.png").exists()
    assert (tmp_path / "artifacts" / "summary.csv").exists()


def test_choose_image_prefers_validation(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "images" / "train" / "train.jpg").write_bytes(b"image")
    (dataset / "images" / "val" / "val.jpg").write_bytes(b"image")
    assert choose_image(None, dataset) == dataset / "images" / "val" / "val.jpg"
    assert choose_image(dataset / "images" / "train" / "train.jpg", dataset) == (
        dataset / "images" / "train" / "train.jpg"
    )
    assert choose_image(tmp_path / "missing.jpg", dataset) is None


def test_slugify():
    assert slugify("runs/detect/train 1") == "runs_detect_train_1"
    assert slugify("a\\b:c") == "a_b_c"
