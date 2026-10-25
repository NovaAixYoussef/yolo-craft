import zipfile

from yolo_craft.archiving import zip_dataset


def test_zip_dataset_creates_archive(tmp_path):
    source = tmp_path / "dataset"
    (source / "images").mkdir(parents=True)
    (source / "images" / "sample.jpg").write_bytes(b"image")
    output = tmp_path / "archive" / "dataset.zip"
    result = zip_dataset(source, output)
    assert result is not None
    assert result.exists()
    assert result.suffix == ".zip"
    with zipfile.ZipFile(result) as archive:
        assert "images/sample.jpg" in archive.namelist()


def test_zip_dataset_missing_source(tmp_path):
    result = zip_dataset(tmp_path / "missing", tmp_path / "out.zip")
    assert result is None
