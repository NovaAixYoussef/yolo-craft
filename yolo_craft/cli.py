import argparse

from . import (
    active,
    archiving,
    augmentation,
    benchmarking,
    capture,
    exporting,
    inference,
    labeling,
    reporting,
    training,
)
from . import __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="yolo-craft",
        description=(
            "Capture hard frames, label them, train, evaluate, and export "
            "YOLO models."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture.add_parser(subparsers)
    labeling.add_parser(subparsers)
    training.add_parser(subparsers)
    inference.add_parser(subparsers)
    active.add_parser(subparsers)
    augmentation.add_parser(subparsers)
    exporting.add_parser(subparsers)
    reporting.add_parser(subparsers)
    benchmarking.add_parser(subparsers)
    archiving.add_parser(subparsers)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
