import csv
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
WEIGHTS = DEMO / "model" / "best.pt"
IMAGES = DEMO / "images"
DETECTIONS = DEMO / "detections"
MEDIA = DEMO / "media"
RUNS = DEMO / "runs"
POSTER = DEMO / "poster.png"


def font(size, bold=False):
    names = (
        ("comicbd.ttf", "arialbd.ttf", "segoeui.ttf", "arial.ttf")
        if bold
        else ("comic.ttf", "arial.ttf", "segoeui.ttf", "calibri.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def bevel(draw, box, raised=True, fill=(192, 192, 192)):
    x1, y1, x2, y2 = box
    light = (255, 255, 255)
    dark = (90, 90, 90)
    draw.rectangle(box, fill=fill, outline=(0, 0, 0))
    if raised:
        draw.line((x1 + 2, y1 + 2, x2 - 2, y1 + 2), fill=light, width=2)
        draw.line((x1 + 2, y1 + 2, x1 + 2, y2 - 2), fill=light, width=2)
        draw.line((x1 + 2, y2 - 2, x2 - 2, y2 - 2), fill=dark, width=2)
        draw.line((x2 - 2, y1 + 2, x2 - 2, y2 - 2), fill=dark, width=2)
    else:
        draw.line((x1 + 2, y1 + 2, x2 - 2, y1 + 2), fill=dark, width=2)
        draw.line((x1 + 2, y1 + 2, x1 + 2, y2 - 2), fill=dark, width=2)
        draw.line((x1 + 2, y2 - 2, x2 - 2, y2 - 2), fill=light, width=2)
        draw.line((x2 - 2, y1 + 2, x2 - 2, y2 - 2), fill=light, width=2)


def fit(path, width, height):
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, height), Image.LANCZOS)
    return image


def metrics():
    with (RUNS / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    best = max(rows, key=lambda row: float(row["metrics/mAP50(B)"]))
    return {
        "epoch": int(best["epoch"]),
        "epochs": len(rows),
        "minutes": float(rows[-1]["time"]) / 60,
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "map50": float(best["metrics/mAP50(B)"]),
        "map5095": float(best["metrics/mAP50-95(B)"]),
    }


def annotated(model, path, size=640):
    result = model.predict(str(path), conf=0.25, imgsz=640, verbose=False)[0]
    frame = result.plot()
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)


def make_video(model):
    apples = sorted(IMAGES.glob("apple_*.jpg"))
    bananas = sorted(IMAGES.glob("banana_*.jpg"))
    files = [path for pair in zip(apples, bananas) for path in pair]
    frames = []
    gif_frames = []
    for index, path in enumerate(files):
        frame = annotated(model, path)
        frames.append(frame)
        if index % 2 == 0:
            rgb = cv2.cvtColor(cv2.resize(frame, (360, 360)), cv2.COLOR_BGR2RGB)
            gif_frames.append(Image.fromarray(rgb).quantize(colors=96, method=Image.ADAPTIVE))
    writer = cv2.VideoWriter(
        str(MEDIA / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 640)
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
    gif_frames[0].save(
        MEDIA / "demo.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=100,
        loop=0,
    )
    print("Video frames:", len(frames), "GIF frames:", len(gif_frames))


def make_poster():
    data = metrics()
    canvas = Image.new("RGB", (1600, 1100), (0, 128, 128))
    draw = ImageDraw.Draw(canvas)
    bevel(draw, (40, 40, 1560, 1050))
    draw.rectangle((46, 46, 1554, 100), fill=(0, 0, 128))
    draw.text(
        (66, 56),
        "TRAINED WITH JUST 100 IMAGES IN A FEW MINUTES",
        fill=(255, 255, 255),
        font=font(32, True),
    )
    for x in (1434, 1474, 1514):
        draw.rectangle((x, 54, x + 30, 92), fill=(192, 192, 192))
        draw.line((x + 2, 56, x + 28, 56), fill=(255, 255, 255), width=2)
        draw.line((x + 2, 56, x + 2, 90), fill=(255, 255, 255), width=2)
        draw.line((x + 2, 90, x + 28, 90), fill=(90, 90, 90), width=2)
        draw.line((x + 28, 56, x + 28, 90), fill=(90, 90, 90), width=2)
    draw.text(
        (80, 124),
        "YOLO CRAFT - Apple vs Banana - 100 epochs at 640 px in "
        + f"{data['minutes']:.1f} minutes",
        fill=(0, 0, 0),
        font=font(22),
    )

    cards = [
        ("PRECISION", f"{data['precision']:.3f}", (0, 0, 160)),
        ("RECALL", f"{data['recall']:.3f}", (0, 128, 0)),
        ("mAP50", f"{data['map50']:.3f}", (128, 0, 128)),
        ("mAP50-95", f"{data['map5095']:.3f}", (192, 0, 0)),
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 80 + index * 365
        bevel(draw, (x, 175, x + 345, 305), fill=(255, 255, 255))
        draw.text((x + 18, 190), label, fill=(0, 0, 128), font=font(20, True))
        draw.text((x + 18, 224), value, fill=color, font=font(44, True))

    for x, path, caption in (
        (80, DETECTIONS / "apple_detected.jpg", "APPLE  0.94"),
        (680, DETECTIONS / "banana_detected.jpg", "BANANA  0.87"),
    ):
        bevel(draw, (x, 340, x + 560, 900), raised=False, fill=(255, 255, 255))
        image = Image.open(path).convert("RGB").resize((540, 540), Image.LANCZOS)
        canvas.paste(image, (x + 10, 350))
        draw.text((x + 10, 910), caption, fill=(0, 0, 128), font=font(26, True))

    bevel(draw, (1270, 340, 1520, 900), fill=(255, 255, 255))
    curves = fit(RUNS / "results.png", 240, 240)
    canvas.paste(curves, (1275, 350))
    facts = [
        "DATASET  100 images",
        "CLASSES  Apple / Banana",
        "EPOCHS  " + str(data["epochs"]),
        "BEST EPOCH  " + str(data["epoch"]),
        "TIME  " + f"{data['minutes']:.1f} min",
    ]
    for index, line in enumerate(facts):
        draw.text((1280, 610 + index * 34), line, fill=(0, 0, 0), font=font(17))

    bevel(draw, (46, 1005, 1554, 1044))
    draw.text(
        (60, 1014),
        "YOLO CRAFT 1.0  |  Apple vs Banana  |  Ready",
        fill=(0, 0, 0),
        font=font(16),
    )
    canvas.save(POSTER)
    print("Poster:", POSTER)


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    make_video(model)
    make_poster()


if __name__ == "__main__":
    main()
