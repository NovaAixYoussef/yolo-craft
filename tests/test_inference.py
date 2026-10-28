import json

import cv2
import numpy as np
import pytest

from yolo_craft import inference
from yolo_craft.cli import build_parser
from yolo_craft.inference import (
    detection_records,
    detection_summary,
    parse_class_filter,
    save_detection_json,
    write_yolo_labels,
    yolo_label_data,
)


class FakeBoxes:
    def __init__(self):
        self.xyxy = [[10.0, 12.0, 40.0, 44.0]]
        self.cls = [0]
        self.conf = [0.91]
        self.xywhn = [[0.25, 0.25, 0.2, 0.3]]

    def __len__(self):
        return len(self.cls)


class FakeResult:
    def __init__(self):
        self.boxes = FakeBoxes()
        self.plot_kwargs = {}

    def plot(self, **kwargs):
        self.plot_kwargs = kwargs
        return np.zeros((48, 64, 3), dtype=np.uint8)


def test_parse_class_filter_accepts_ids_and_names():
    names = {0: "Parfum", 1: "Bottle", 2: "Box"}
    assert parse_class_filter("Parfum, 1, bottle", names) == [0, 1]
    assert parse_class_filter("2;Parfum;2", names) == [2, 0]
    assert parse_class_filter(None, names) is None
    assert parse_class_filter("", names) is None


def test_parse_class_filter_rejects_unknown_or_empty():
    names = {0: "Parfum"}
    with pytest.raises(ValueError):
        parse_class_filter("Parfum, unknown", names)
    with pytest.raises(ValueError):
        parse_class_filter(" , ; ", names)
    with pytest.raises(ValueError):
        parse_class_filter("3", names)


def test_detection_records_and_summary():
    result = FakeResult()
    records = detection_records(result, {0: "Parfum"})
    assert records == [
        {
            "class_id": 0,
            "class": "Parfum",
            "confidence": 0.91,
            "box": [10.0, 12.0, 40.0, 44.0],
        }
    ]
    assert detection_summary(records) == (1, "Parfum", 0.91)
    assert detection_summary([]) == (0, "", 0.0)
    classes, coordinates = yolo_label_data(result)
    assert classes == [0]
    assert coordinates == [[0.25, 0.25, 0.2, 0.3]]


def test_write_yolo_labels(tmp_path):
    path = write_yolo_labels(
        tmp_path / "labels",
        "frame_0001",
        [0, 2],
        [[0.5, 0.5, 0.2, 0.2], [0.1, 0.2, 0.3, 0.4]],
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "0 0.500000 0.500000 0.200000 0.200000"
    assert lines[1] == "2 0.100000 0.200000 0.300000 0.400000"

    empty = write_yolo_labels(
        tmp_path / "labels",
        "background",
        [],
        [],
    )
    assert empty.read_text(encoding="utf-8") == ""


def test_save_detection_json(tmp_path):
    frames = [
        {
            "frame": 1,
            "source": "image.jpg",
            "detections": [
                {
                    "class_id": 0,
                    "class": "Parfum",
                    "confidence": 0.91,
                    "box": [10.0, 12.0, 40.0, 44.0],
                }
            ],
        }
    ]
    path = save_detection_json(
        tmp_path / "detections",
        "image.jpg",
        tmp_path / "best.pt",
        {0: "Parfum"},
        frames,
    )
    assert path == tmp_path / "detections.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source"] == "image.jpg"
    assert payload["weights"] == str(tmp_path / "best.pt")
    assert payload["classes"] == [{"id": 0, "name": "Parfum"}]
    assert payload["frames"] == frames


def test_run_image_folder_headless(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    for index in range(2):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :] = (20, 40, 60)
        assert cv2.imwrite(str(images / f"frame_{index}.jpg"), image)

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    created = {}

    class FakeModel:
        def __init__(self, source):
            created["model"] = self
            created["source"] = source
            self.names = {0: "Parfum"}
            self.calls = []
            self.results = []

        def predict(self, frame, **kwargs):
            self.calls.append(kwargs)
            result = FakeResult()
            self.results.append(result)
            return [result]

    def fail_imshow(*args, **kwargs):
        raise AssertionError("imshow must not be called in headless mode")

    monkeypatch.setattr(inference, "YOLO", FakeModel)
    monkeypatch.setattr(inference, "ensure_runtime_dirs", lambda: None)
    monkeypatch.setattr(inference.cv2, "imshow", fail_imshow)

    args = build_parser().parse_args(
        [
            "test",
            "--weights",
            str(weights),
            "--source",
            str(images),
            "--save",
            str(tmp_path / "annotated"),
            "--save-txt",
            str(tmp_path / "labels"),
            "--save-json",
            str(tmp_path / "detections"),
            "--classes",
            "Parfum",
            "--device",
            "cpu",
            "--no-conf",
            "--line-width",
            "3",
            "--no-display",
        ]
    )

    assert inference.run(args) == 0
    model = created["model"]
    assert created["source"] == str(weights)
    assert len(model.calls) == 2
    assert model.calls[0]["classes"] == [0]
    assert model.calls[0]["device"] == "cpu"
    assert model.results[0].plot_kwargs == {
        "conf": False,
        "labels": True,
        "line_width": 3,
    }

    annotated = list((tmp_path / "annotated").glob("*_pred.jpg"))
    assert len(annotated) == 2
    labels = list((tmp_path / "labels").glob("*.txt"))
    assert len(labels) == 2
    assert labels[0].read_text(encoding="utf-8").startswith("0 0.250000")

    payload = json.loads(
        (tmp_path / "detections.json").read_text(encoding="utf-8")
    )
    assert len(payload["frames"]) == 2
    assert payload["frames"][0]["source"] == "frame_0.jpg"
    assert payload["frames"][0]["detections"][0]["class"] == "Parfum"
