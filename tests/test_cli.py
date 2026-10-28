import pytest

from yolo_craft import __version__
from yolo_craft.cli import build_parser, main


def test_cli_exposes_all_commands():
    parser = build_parser()
    args = parser.parse_args(["capture", "--source", "0"])
    assert args.command == "capture"
    assert callable(args.func)


def test_cli_version():
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0


def test_cli_requires_command():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code != 0


def test_cli_label_defaults():
    parser = build_parser()
    args = parser.parse_args(["label", "--classes", "joystick,button"])
    assert args.classes == "joystick,button"
    assert args.command == "label"


def test_cli_test_defaults():
    parser = build_parser()
    args = parser.parse_args(["test", "--source", "image.jpg"])
    assert args.imgsz == 640
    assert args.conf == 0.25
    assert args.augment is False
    assert args.classes is None
    assert args.device == ""
    assert args.save_txt is None
    assert args.save_json is None
    assert args.no_labels is False
    assert args.no_conf is False
    assert args.line_width is None
    assert args.no_display is False


def test_cli_train_hyperparameters():
    parser = build_parser()
    args = parser.parse_args(["train"])
    assert args.cos_lr is True
    assert args.close_mosaic == 10
    assert args.box == 7.5
    assert args.cls == 0.5
    assert args.fliplr == 0.5
    assert args.mosaic == 1.0

    custom = parser.parse_args(
        [
            "train",
            "--no-cos-lr",
            "--close-mosaic",
            "4",
            "--box",
            "6.5",
            "--cls",
            "0.9",
            "--fliplr",
            "0.1",
            "--mosaic",
            "0.4",
        ]
    )
    assert custom.cos_lr is False
    assert custom.close_mosaic == 4
    assert custom.box == 6.5
    assert custom.cls == 0.9
    assert custom.fliplr == 0.1
    assert custom.mosaic == 0.4


def test_version_value():
    assert __version__ == "1.0.0"
