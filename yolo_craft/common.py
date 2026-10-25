import re
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RAW_DIR = Path("data") / "raw"
PROCESSED_DIR = Path("data") / "processed"
DATASET_DIR = Path("data") / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
BACKGROUND_DIR = DATASET_DIR / "background"
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
RUNS_DIR = OUTPUTS_DIR / "runs"
TRAIN_DIR = RUNS_DIR / "train"
EXPORTS_DIR = OUTPUTS_DIR / "exports"
BENCHMARKS_DIR = OUTPUTS_DIR / "benchmarks"
REPORTS_DIR = OUTPUTS_DIR / "reports"


def parse_source(value):
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def is_stream(source):
    return isinstance(source, str) and source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )


def is_video_file(source):
    return isinstance(source, str) and not is_stream(source)


def source_label(source):
    if isinstance(source, int):
        return f"webcam {source}"
    return str(source)


def collect_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def next_capture_path(folder, prefix, suffix=".jpg"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    indices = []
    for path in folder.glob(f"{prefix}_*"):
        match = pattern.match(path.stem)
        if match:
            indices.append(int(match.group(1)))
    next_index = max(indices, default=0) + 1
    return folder / f"{prefix}_{next_index:04d}{suffix}"


def load_yaml(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return {}
    return payload or {}


def normalize_names(names):
    if isinstance(names, dict):
        parsed = {}
        for key, value in names.items():
            try:
                parsed[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return dict(sorted(parsed.items()))
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def resolve_existing(explicit, candidates):
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path
        return None
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    return None


def ensure_runtime_dirs():
    for folder in (
        RAW_DIR,
        PROCESSED_DIR,
        DATASET_DIR,
        BACKGROUND_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        RUNS_DIR,
        EXPORTS_DIR,
        BENCHMARKS_DIR,
        REPORTS_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)
