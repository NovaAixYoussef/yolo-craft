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
