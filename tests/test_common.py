from pathlib import Path

from yolo_craft.common import (
    is_stream,
    is_video_file,
    next_capture_path,
    normalize_names,
    parse_source,
    resolve_existing,
    source_label,
)


def test_parse_source_distinguishes_webcam_and_paths():
    assert parse_source("0") == 0
    assert parse_source(" rtsp://camera/live ") == "rtsp://camera/live"
    assert parse_source("video.mp4") == "video.mp4"


def test_stream_and_video_detection():
    assert is_stream("rtsp://camera/live")
    assert is_stream("https://example.com/feed")
    assert not is_stream("clip.mp4")
    assert is_video_file("clip.mp4")
    assert not is_video_file(0)


def test_source_label():
    assert source_label(0) == "webcam 0"
    assert source_label("clip.mp4") == "clip.mp4"


def test_next_capture_path_uses_max_index(tmp_path):
    (tmp_path / "uncertain_0001.jpg").write_bytes(b"a")
    (tmp_path / "uncertain_0007.jpg").write_bytes(b"b")
    (tmp_path / "manual_0042.jpg").write_bytes(b"c")
    result = next_capture_path(tmp_path, "uncertain")
    assert result == tmp_path / "uncertain_0008.jpg"


def test_normalize_names_sorts_mapping_and_accepts_list():
    assert normalize_names({"2": "button", "0": "joystick"}) == {
        0: "joystick",
        2: "button",
    }
    assert normalize_names(["joystick", "button"]) == {
        0: "joystick",
        1: "button",
    }
    assert normalize_names(None) == {}


def test_resolve_existing_requires_explicit_path(tmp_path):
    file_path = tmp_path / "weights.pt"
    file_path.write_bytes(b"model")
    assert resolve_existing(str(file_path), []) == file_path
    assert resolve_existing(str(tmp_path / "missing.pt"), [str(file_path)]) is None
    assert resolve_existing(None, [str(file_path)]) == file_path
    assert resolve_existing(None, [str(tmp_path / "missing.pt")]) is None


def test_common_paths_are_relative():
    assert Path("data/raw").name == "raw"
