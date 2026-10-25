import shutil
from pathlib import Path

from .common import DATASET_DIR, OUTPUTS_DIR


def zip_dataset(folder_to_zip=None, output_filename=None):
    source = Path(folder_to_zip) if folder_to_zip else DATASET_DIR
    if not source.exists():
        print(f"Error: source folder '{source}' does not exist.")
        return None

    output = Path(output_filename) if output_filename else OUTPUTS_DIR / "dataset.zip"
    if output.suffix.lower() == ".zip":
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating archive from '{source}'.")
    archive = shutil.make_archive(str(output), "zip", str(source))
    archive_path = Path(archive)
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Archive created: {archive_path} ({size_mb:.2f} MB)")
    return archive_path


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "archive",
        help="Archive the training dataset as a zip file.",
    )
    parser.add_argument("--source", default=str(DATASET_DIR))
    parser.add_argument("--output", default=str(OUTPUTS_DIR / "dataset.zip"))
    parser.set_defaults(func=run)
    return parser


def run(args):
    result = zip_dataset(args.source, args.output)
    return 0 if result is not None else 1
