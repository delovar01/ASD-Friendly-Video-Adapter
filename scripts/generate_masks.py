"""Generate placeholder cartoon animal masks (512x512 RGBA).

Run: python scripts/generate_masks.py
Writes PNGs to app/static/masks/.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "app" / "static" / "masks"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 512


def _new() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _face(draw: ImageDraw.ImageDraw, face_color: tuple[int, int, int],
          ear_color: tuple[int, int, int] | None = None) -> None:
    if ear_color is None:
        ear_color = face_color
    # Face
    draw.ellipse((56, 96, 456, 496), fill=face_color + (255,))
    # Inner face (snout area, slightly lighter)
    lighter = tuple(min(255, c + 28) for c in face_color)
    draw.ellipse((156, 280, 356, 456), fill=lighter + (255,))


def _eyes(draw: ImageDraw.ImageDraw) -> None:
    # Whites
    draw.ellipse((150, 220, 230, 300), fill=(255, 255, 255, 255))
    draw.ellipse((282, 220, 362, 300), fill=(255, 255, 255, 255))
    # Pupils
    draw.ellipse((176, 246, 212, 282), fill=(30, 30, 30, 255))
    draw.ellipse((308, 246, 344, 282), fill=(30, 30, 30, 255))
    # Highlights
    draw.ellipse((198, 252, 208, 262), fill=(255, 255, 255, 255))
    draw.ellipse((330, 252, 340, 262), fill=(255, 255, 255, 255))


def _nose_mouth(draw: ImageDraw.ImageDraw, nose_color=(30, 30, 30)) -> None:
    draw.ellipse((232, 340, 280, 380), fill=nose_color + (255,))
    draw.arc((200, 370, 312, 440), start=20, end=160, fill=(30, 30, 30, 255), width=6)


def fox() -> Image.Image:
    img, d = _new()
    # Triangular ears
    orange = (230, 126, 48)
    d.polygon([(90, 160), (60, 40), (200, 150)], fill=orange + (255,))
    d.polygon([(312, 150), (452, 40), (422, 160)], fill=orange + (255,))
    d.polygon([(110, 150), (95, 80), (175, 145)], fill=(255, 240, 230, 255))
    d.polygon([(337, 145), (417, 80), (402, 150)], fill=(255, 240, 230, 255))
    _face(d, orange)
    # White cheek
    d.ellipse((130, 310, 382, 470), fill=(255, 245, 230, 255))
    _eyes(d)
    _nose_mouth(d)
    return img


def bear() -> Image.Image:
    img, d = _new()
    brown = (139, 94, 60)
    d.ellipse((60, 90, 180, 210), fill=brown + (255,))
    d.ellipse((332, 90, 452, 210), fill=brown + (255,))
    d.ellipse((85, 115, 155, 185), fill=(210, 170, 130, 255))
    d.ellipse((357, 115, 427, 185), fill=(210, 170, 130, 255))
    _face(d, brown)
    d.ellipse((160, 310, 352, 460), fill=(222, 190, 150, 255))
    _eyes(d)
    _nose_mouth(d)
    return img


def rabbit() -> Image.Image:
    img, d = _new()
    gray = (230, 225, 220)
    # Long ears
    d.ellipse((140, 0, 210, 220), fill=gray + (255,))
    d.ellipse((302, 0, 372, 220), fill=gray + (255,))
    d.ellipse((155, 20, 195, 200), fill=(255, 190, 195, 255))
    d.ellipse((317, 20, 357, 200), fill=(255, 190, 195, 255))
    _face(d, gray)
    _eyes(d)
    _nose_mouth(d, nose_color=(230, 120, 140))
    # Buck teeth
    d.rectangle((242, 400, 258, 440), fill=(255, 255, 255, 255))
    d.rectangle((258, 400, 274, 440), fill=(255, 255, 255, 255))
    return img


def panda() -> Image.Image:
    img, d = _new()
    # Black ears
    d.ellipse((60, 70, 180, 190), fill=(30, 30, 30, 255))
    d.ellipse((332, 70, 452, 190), fill=(30, 30, 30, 255))
    _face(d, (250, 250, 250))
    # Black eye patches
    d.ellipse((130, 200, 240, 320), fill=(30, 30, 30, 255))
    d.ellipse((272, 200, 382, 320), fill=(30, 30, 30, 255))
    # Eyes (small, on patches)
    d.ellipse((170, 240, 208, 278), fill=(255, 255, 255, 255))
    d.ellipse((304, 240, 342, 278), fill=(255, 255, 255, 255))
    d.ellipse((182, 252, 200, 270), fill=(30, 30, 30, 255))
    d.ellipse((316, 252, 334, 270), fill=(30, 30, 30, 255))
    _nose_mouth(d)
    return img


def cat() -> Image.Image:
    img, d = _new()
    gray = (180, 180, 190)
    # Triangle ears
    d.polygon([(100, 180), (80, 40), (210, 160)], fill=gray + (255,))
    d.polygon([(302, 160), (432, 40), (412, 180)], fill=gray + (255,))
    d.polygon([(120, 170), (110, 75), (185, 155)], fill=(255, 200, 210, 255))
    d.polygon([(327, 155), (402, 75), (392, 170)], fill=(255, 200, 210, 255))
    _face(d, gray)
    _eyes(d)
    # Pink nose
    d.polygon([(232, 345), (280, 345), (256, 380)], fill=(230, 120, 140, 255))
    d.arc((200, 370, 312, 440), start=20, end=160, fill=(30, 30, 30, 255), width=6)
    # Whiskers
    for y in (370, 390, 410):
        d.line((100, y, 210, y - 10), fill=(30, 30, 30, 180), width=3)
        d.line((302, y - 10, 412, y), fill=(30, 30, 30, 180), width=3)
    return img


def main() -> None:
    animals = {
        "fox.png": fox(),
        "bear.png": bear(),
        "rabbit.png": rabbit(),
        "panda.png": panda(),
        "cat.png": cat(),
    }
    for name, img in animals.items():
        p = OUT / name
        img.save(p, "PNG")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
