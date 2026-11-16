import cv2
import numpy as np
import yaml

from yolo_craft.labeling import (
    HEADER_HEIGHT,
    HEADER_MIN_WIDTH,
    MAX_DISPLAY_HEIGHT,
    MAX_DISPLAY_WIDTH,
    compute_display_scale,
    display_to_image,
    label_images,
    load_classes,
    parse_class_override,
    read_image,
    render_frame,
    split_dataset,
    write_data_yaml,
    write_image,
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


def test_render_frame_keeps_small_image_below_header():
    image = np.zeros((28, 28, 3), dtype=np.uint8)
    image[:, :] = (20, 40, 60)

    canvas, scale = render_frame(
        image,
        boxes=[(2, 3, 20, 22, 0)],
        classes={0: "digit"},
        selected_id=0,
        index=0,
        total=1,
    )

    assert scale > 1
    assert canvas.shape[0] == HEADER_HEIGHT + int(round(28 * scale))
    assert canvas.shape[1] >= HEADER_MIN_WIDTH
    assert tuple(canvas[HEADER_HEIGHT + 14, 14]) == (20, 40, 60)


def test_display_to_image_maps_scaled_coordinates():
    assert display_to_image(6 * 12 + 5, HEADER_HEIGHT + 6 * 12 + 5, 12, HEADER_HEIGHT, 28, 28) == (6, 6)
    assert display_to_image(10, 10, 12, HEADER_HEIGHT, 28, 28) is None
    assert display_to_image(28 * 12, HEADER_HEIGHT + 5, 12, HEADER_HEIGHT, 28, 28) is None


def test_compute_display_scale_fits_large_images():
    scale = compute_display_scale(1920, 1080)
    assert scale < 1
    assert 1920 * scale <= MAX_DISPLAY_WIDTH + 1
    assert 1080 * scale <= MAX_DISPLAY_HEIGHT + 1


def test_read_and_write_image_with_unicode_path(tmp_path):
    path = tmp_path / "größe_ümlaut.png"
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)

    assert write_image(path, image)
    loaded = read_image(path)
    assert loaded is not None
    assert loaded.shape == (12, 16, 3)
    assert tuple(loaded[5, 5]) == (10, 20, 30)


def test_label_images_draws_and_saves_on_small_image(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    processed = tmp_path / "processed"
    background = tmp_path / "background"
    source.mkdir()
    make_image(source / "digit.png", width=28, height=28)

    callback = {}
    shown = {}

    def fake_named_window(name):
        return None

    def fake_set_mouse_callback(name, func):
        callback["func"] = func

    def fake_imshow(name, canvas):
        shown["canvas"] = canvas

    def fake_wait_key(delay):
        func = callback["func"]
        func(cv2.EVENT_LBUTTONDOWN, 2 * 12, 88 + 3 * 12, 0, None)
        func(cv2.EVENT_MOUSEMOVE, 20 * 12, 88 + 22 * 12, 0, None)
        func(cv2.EVENT_LBUTTONUP, 20 * 12, 88 + 22 * 12, 0, None)
        return ord("s")

    monkeypatch.setattr(cv2, "namedWindow", fake_named_window)
    monkeypatch.setattr(cv2, "setMouseCallback", fake_set_mouse_callback)
    monkeypatch.setattr(cv2, "imshow", fake_imshow)
    monkeypatch.setattr(cv2, "waitKey", fake_wait_key)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    stats = label_images(
        source_dir=source,
        processed_dir=processed,
        background_dir=background,
        classes={0: "digit"},
    )

    assert stats == {"saved": 1, "background": 0, "skipped": 0}
    assert (processed / "digit.png").exists()
    label = (processed / "digit.txt").read_text(encoding="utf-8").strip()
    assert label == "0 0.392857 0.446429 0.642857 0.678571"
    assert shown["canvas"].shape[0] > 28


def test_label_images_background_writes_empty_label(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    processed = tmp_path / "processed"
    background = tmp_path / "background"
    source.mkdir()
    make_image(source / "frame.png", width=28, height=28)

    monkeypatch.setattr(cv2, "namedWindow", lambda name: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda name, func: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, canvas: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("b"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    stats = label_images(
        source_dir=source,
        processed_dir=processed,
        background_dir=background,
        classes={0: "digit"},
    )

    assert stats == {"saved": 0, "background": 1, "skipped": 0}
    assert (background / "frame.png").exists()
    assert (processed / "frame.png").exists()
    assert (processed / "frame.txt").read_text(encoding="utf-8") == ""


def test_label_images_skips_frame_without_writing(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    processed = tmp_path / "processed"
    background = tmp_path / "background"
    source.mkdir()
    make_image(source / "frame.png", width=28, height=28)

    monkeypatch.setattr(cv2, "namedWindow", lambda name: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda name, func: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, canvas: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("n"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    stats = label_images(
        source_dir=source,
        processed_dir=processed,
        background_dir=background,
        classes={0: "digit"},
    )

    assert stats == {"saved": 0, "background": 0, "skipped": 1}
    assert not (processed / "frame.png").exists()
    assert not (processed / "frame.txt").exists()


def test_label_images_undoes_last_box(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    processed = tmp_path / "processed"
    background = tmp_path / "background"
    source.mkdir()
    make_image(source / "frame.png", width=28, height=28)

    callback = {}
    calls = {"count": 0}

    def fake_set_mouse_callback(name, func):
        callback["func"] = func

    def fake_wait_key(delay):
        calls["count"] += 1
        if calls["count"] == 1:
            func = callback["func"]
            func(cv2.EVENT_LBUTTONDOWN, 2 * 12, 88 + 3 * 12, 0, None)
            func(cv2.EVENT_MOUSEMOVE, 20 * 12, 88 + 22 * 12, 0, None)
            func(cv2.EVENT_LBUTTONUP, 20 * 12, 88 + 22 * 12, 0, None)
            return 255
        if calls["count"] == 2:
            return ord("c")
        return ord("s")

    monkeypatch.setattr(cv2, "namedWindow", lambda name: None)
    monkeypatch.setattr(cv2, "setMouseCallback", fake_set_mouse_callback)
    monkeypatch.setattr(cv2, "imshow", lambda name, canvas: None)
    monkeypatch.setattr(cv2, "waitKey", fake_wait_key)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda name, prop: 1.0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    stats = label_images(
        source_dir=source,
        processed_dir=processed,
        background_dir=background,
        classes={0: "digit"},
    )

    assert stats == {"saved": 1, "background": 0, "skipped": 0}
    assert (processed / "frame.txt").read_text(encoding="utf-8") == ""
