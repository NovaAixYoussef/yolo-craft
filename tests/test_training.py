from pathlib import Path

from yolo_craft import training


def test_train_resolves_project_path(tmp_path, monkeypatch):
    data_path = tmp_path / "data.yaml"
    data_path.write_text("path: .\n", encoding="utf-8")
    model_path = tmp_path / "base.pt"
    model_path.write_bytes(b"weights")
    captured = {}

    class FakeModel:
        def __init__(self, source):
            captured["source"] = source

        def train(self, **kwargs):
            captured.update(kwargs)
            return "trained"

    monkeypatch.setattr(training, "YOLO", FakeModel)
    monkeypatch.setattr(training, "ensure_runtime_dirs", lambda: None)

    result = training.train_model(
        data_path=data_path,
        model_name=str(model_path),
        epochs=1,
        project="outputs/runs",
        name="unit",
    )

    assert result == "trained"
    assert captured["source"] == str(model_path)
    assert Path(captured["project"]).is_absolute()
    assert captured["project"] == str((Path.cwd() / "outputs" / "runs").resolve())
    assert captured["data"] == str(data_path.resolve())
    assert captured["cos_lr"] is True
    assert captured["close_mosaic"] == 10
    assert captured["box"] == 7.5
    assert captured["cls"] == 0.5
    assert captured["fliplr"] == 0.5
    assert captured["mosaic"] == 1.0


def test_train_forwards_custom_hyperparameters(tmp_path, monkeypatch):
    data_path = tmp_path / "data.yaml"
    data_path.write_text("path: .\n", encoding="utf-8")
    model_path = tmp_path / "base.pt"
    model_path.write_bytes(b"weights")
    captured = {}

    class FakeModel:
        def __init__(self, source):
            captured["source"] = source

        def train(self, **kwargs):
            captured.update(kwargs)
            return "trained"

    monkeypatch.setattr(training, "YOLO", FakeModel)
    monkeypatch.setattr(training, "ensure_runtime_dirs", lambda: None)

    result = training.train_model(
        data_path=data_path,
        model_name=str(model_path),
        epochs=2,
        cos_lr=False,
        close_mosaic=3,
        box=5.25,
        cls=0.75,
        fliplr=0.2,
        mosaic=0.6,
    )

    assert result == "trained"
    assert captured["cos_lr"] is False
    assert captured["close_mosaic"] == 3
    assert captured["box"] == 5.25
    assert captured["cls"] == 0.75
    assert captured["fliplr"] == 0.2
    assert captured["mosaic"] == 0.6
