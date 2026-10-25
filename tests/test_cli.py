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


def test_version_value():
    assert __version__ == "1.0.0"
