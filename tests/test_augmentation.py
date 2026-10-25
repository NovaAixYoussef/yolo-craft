import cv2
import numpy as np
import yaml

from yolo_craft.augmentation import (
    build_frame,
    infer_labels_dir,
    letterbox,
    load_entry,
    load_names,
    mosaic_transform,
    read_labels,
)


def make_entry(folder, name, width=160, height=120):
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 90), (10, 200, 240), -1)
    image_path = folder / f"{name}.jpg"
    cv2.imwrite(str(image_path), image)
    label_path = folder / f"{name}.txt"
    label_path.write_text("0 0.3125 0.4583 0.375 0.5833\n", encoding="utf-8")
    return image_path


def test_read_and_letterbox_keep_labels(tmp_path):
    image_path = make_entry(tmp_path, "frame")
    labels = read_labels(tmp_path / "frame.txt", 160, 120)
    assert len(labels) == 1
    entry = load_entry(image_path, tmp_path)
    assert entry is not None
    canvas, boxes = letterbox(entry["image"], entry["boxes"], 640, 640)
    assert canvas.shape == (640, 640, 3)
    assert len(boxes) == 1
    class_id, x1, y1, x2, y2 = boxes[0]
    assert class_id == 0
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 640


def test_mosaic_uses_four_entries(tmp_path):
    entries = [load_entry(make_entry(tmp_path, f"frame_{i}"), tmp_path) for i in range(4)]
    entries = [entry for entry in entries if entry is not None]
    canvas, boxes = mosaic_transform(entries, 640, np.random.default_rng(4))
    assert canvas.shape == (640, 640, 3)
    assert len(boxes) >= 1


def test_build_frame_supports_all_modes(tmp_path):
    entries = [load_entry(make_entry(tmp_path, f"frame_{i}"), tmp_path) for i in range(4)]
    entries = [entry for entry in entries if entry is not None]
    for mode in ("mosaic", "scale", "color", "flip", "random"):
        frame, boxes = build_frame(mode, entries, 0, 320, np.random.default_rng(9))
        assert frame.shape == (320, 320, 3)
        assert isinstance(boxes, list)


def test_load_names_and_label_dir(tmp_path):
    data_path = tmp_path / "data.yaml"
    data_path.write_text(
        yaml.safe_dump({"names": {0: "joystick"}}),
        encoding="utf-8",
    )
    assert load_names(data_path) == {0: "joystick"}
    assert infer_labels_dir(tmp_path / "images" / "train") == (
        tmp_path / "labels" / "train"
    )
    assert infer_labels_dir(tmp_path / "processed") is None
