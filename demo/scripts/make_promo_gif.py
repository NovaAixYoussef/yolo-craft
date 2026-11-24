from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "demo" / "app" / "YOLO_CRAFT_Promo_1080p.mp4"
OUT = ROOT / "demo" / "media" / "promo.gif"
WIDTH = 480
FPS = 6
COLORS = 64


def main():
    capture = cv2.VideoCapture(str(SOURCE))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or source_fps <= 0:
        raise SystemExit("Cannot read promo video.")

    ratio = WIDTH / capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * ratio))
    step = max(1, int(round(source_fps / FPS)))
    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            resized = cv2.resize(frame, (WIDTH, height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb).quantize(colors=COLORS, method=Image.ADAPTIVE))
        index += 1
    capture.release()
    if not frames:
        raise SystemExit("No frames extracted.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=2,
    )
    print("source_frames", total, "source_fps", source_fps)
    print("gif_frames", len(frames), "size", (WIDTH, height))
    print("output", OUT, round(OUT.stat().st_size / 1048576, 2), "MB")


if __name__ == "__main__":
    main()
