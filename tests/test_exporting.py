from yolo_craft.exporting import copy_artifact


def test_copy_artifact_file(tmp_path):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"model")
    destination = copy_artifact(source, tmp_path / "exports")
    assert destination is not None
    assert destination.read_bytes() == b"model"


def test_copy_artifact_directory(tmp_path):
    source = tmp_path / "model.mlpackage"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"weights")
    destination = copy_artifact(source, tmp_path / "exports")
    assert destination is not None
    assert (destination / "weights.bin").read_bytes() == b"weights"


def test_copy_missing_artifact(tmp_path):
    assert copy_artifact(tmp_path / "missing.onnx", tmp_path / "exports") is None
