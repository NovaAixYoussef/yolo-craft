import cv2
import numpy as np
import yaml

from yolo_craft.labeling import (
    load_classes,
    parse_class_override,
    split_dataset,
    write_data_yaml,
)


def make_image(path, width=64, height=48):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (20, 40, 60)
    assert cv2.imwrite(str(path), image)


def test_parse_class_override():
    assert parse_class_override("joystick, button") == {
        0: "joystick",
        1: "button",
    }
    assert parse_class_override("") is None
    assert parse_class_override(" , ") is None


def test_write_and_load_classes(tmp_path):
    data_path = tmp_path / "data.yaml"
    write_data_yaml(data_path, tmp_path, {0: "joystick", 1: "button"})
    assert load_classes(data_path) == {0: "joystick", 1: "button"}
    assert load_classes(tmp_path / "missing.yaml") == {0: "object"}
    assert load_classes(tmp_path / "missing.yaml", {3: "custom"}) == {3: "custom"}


def test_split_dataset_creates_yolo_structure(tmp_path):
    processed = tmp_path / "processed"
    dataset = tmp_path / "dataset"
    processed.mkdir()
    for index in range(5):
        make_image(processed / f"frame_{index}.jpg")
        (processed / f"frame_{index}.txt").write_text(
            f"0 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

    train_count, val_count = split_dataset(
        dataset_dir=dataset,
        processed_dir=processed,
        classes={0: "joystick", 1: "button"},
        train_ratio=0.8,
        seed=3,
        data_path=dataset / "data.yaml",
    )

    assert train_count == 4
    assert val_count == 1
    assert len(list((dataset / "images" / "train").glob("*.jpg"))) == 4
    assert len(list((dataset / "images" / "val").glob("*.jpg"))) == 1
    assert len(list((dataset / "labels" / "train").glob("*.txt"))) == 4
    assert len(list((dataset / "labels" / "val").glob("*.txt"))) == 1
    payload = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    assert payload["names"] == {0: "joystick", 1: "button"}
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"


def test_split_dataset_without_images_returns_zero(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    train_count, val_count = split_dataset(
        dataset_dir=tmp_path / "dataset",
        processed_dir=processed,
        classes={0: "object"},
    )
    assert (train_count, val_count) == (0, 0)
