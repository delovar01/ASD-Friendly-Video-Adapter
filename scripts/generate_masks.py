"""Generate placeholder cartoon animal masks (512x512 RGBA).

The masks are drawn around fixed anchor points so that `app/pipeline/faces.py`
can warp them onto detected faces by eye position:

    eye_left  = (0.35 * SIZE, 0.48 * SIZE)
    eye_right = (0.65 * SIZE, 0.48 * SIZE)

The mask extends well beyond the detected eye region (forehead to below chin,
including ears) so the warp fully covers the underlying human face. The outer
alpha is Gaussian-feathered so there is no hard rectangular edge when blending.

Run: python scripts/generate_masks.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "app" / "static" / "masks"
OUT.mkdir(parents=True, exist_ok=True)

SIZE = 512

# Anchor points — MUST match MASK_EYE_L / MASK_EYE_R in app/pipeline/faces.py.
EYE_L = (int(SIZE * 0.35), int(SIZE * 0.48))
EYE_R = (int(SIZE * 0.65), int(SIZE * 0.48))
NOSE = (int(SIZE * 0.50), int(SIZE * 0.62))
MOUTH = (int(SIZE * 0.50), int(SIZE * 0.74))

# Head shape (ellipse bounding box) — full coverage from above forehead to below chin.
HEAD_BOX = (int(SIZE * 0.10), int(SIZE * 0.14),
            int(SIZE * 0.90), int(SIZE * 0.94))

FEATHER_RADIUS = 14  # px of outer-alpha Gaussian blur for a soft, non-sticker edge.


def _new() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _head(draw: ImageDraw.ImageDraw, face_color: tuple[int, int, int]) -> None:
    draw.ellipse(HEAD_BOX, fill=face_color + (255,))
    # Slightly lighter muzzle area around nose/mouth.
    lighter = tuple(min(255, c + 30) for c in face_color)
    mz = (int(SIZE * 0.30), int(SIZE * 0.56),
          int(SIZE * 0.70), int(SIZE * 0.90))
    draw.ellipse(mz, fill=lighter + (255,))


def _eyes(draw: ImageDraw.ImageDraw) -> None:
    eye_r = int(SIZE * 0.065)
    pupil_r = int(SIZE * 0.030)
    hl_r = int(SIZE * 0.010)
    for cx, cy in (EYE_L, EYE_R):
        # White of eye.
        draw.ellipse((cx - eye_r, cy - eye_r, cx + eye_r, cy + eye_r),
                     fill=(255, 255, 255, 255))
        # Pupil (slightly down-right to give it some charm).
        px, py = cx + 3, cy + 3
        draw.ellipse((px - pupil_r, py - pupil_r, px + pupil_r, py + pupil_r),
                     fill=(30, 30, 30, 255))
        # Highlight.
        hx, hy = cx + 6, cy - 3
        draw.ellipse((hx - hl_r, hy - hl_r, hx + hl_r, hy + hl_r),
                     fill=(255, 255, 255, 255))


def _nose(draw: ImageDraw.ImageDraw,
          color: tuple[int, int, int] = (30, 30, 30)) -> None:
    nx, ny = NOSE
    rx, ry = int(SIZE * 0.045), int(SIZE * 0.038)
    draw.ellipse((nx - rx, ny - ry, nx + rx, ny + ry), fill=color + (255,))


def _mouth(draw: ImageDraw.ImageDraw) -> None:
    mx, my = MOUTH
    draw.arc((mx - 55, my - 20, mx + 55, my + 30),
             start=20, end=160, fill=(30, 30, 30, 255), width=6)


def fox() -> Image.Image:
    img, d = _new()
    orange = (230, 126, 48)
    # Triangular ears at the top of the head (outer + inner).
    d.polygon([(95, 160), (55, 30), (210, 150)], fill=orange + (255,))
    d.polygon([(302, 150), (457, 30), (417, 160)], fill=orange + (255,))
    d.polygon([(115, 150), (95, 70), (180, 140)], fill=(255, 240, 230, 255))
    d.polygon([(332, 140), (417, 70), (397, 150)], fill=(255, 240, 230, 255))
    _head(d, orange)
    # Off-white cheek patch.
    d.ellipse((140, 310, 372, 470), fill=(255, 245, 230, 255))
    _eyes(d)
    _nose(d)
    _mouth(d)
    return img


def bear() -> Image.Image:
    img, d = _new()
    brown = (139, 94, 60)
    # Round ears.
    d.ellipse((55, 80, 175, 200), fill=brown + (255,))
    d.ellipse((337, 80, 457, 200), fill=brown + (255,))
    d.ellipse((80, 105, 150, 175), fill=(210, 170, 130, 255))
    d.ellipse((362, 105, 432, 175), fill=(210, 170, 130, 255))
    _head(d, brown)
    _eyes(d)
    _nose(d)
    _mouth(d)
    return img


def rabbit() -> Image.Image:
    img, d = _new()
    gray = (230, 225, 220)
    # Long ears above head.
    d.ellipse((150, 0, 210, 210), fill=gray + (255,))
    d.ellipse((302, 0, 362, 210), fill=gray + (255,))
    d.ellipse((163, 18, 197, 195), fill=(255, 190, 195, 255))
    d.ellipse((315, 18, 349, 195), fill=(255, 190, 195, 255))
    _head(d, gray)
    _eyes(d)
    _nose(d, color=(230, 120, 140))
    _mouth(d)
    # Buck teeth under mouth.
    d.rectangle((244, 395, 258, 435), fill=(255, 255, 255, 255))
    d.rectangle((258, 395, 272, 435), fill=(255, 255, 255, 255))
    return img


def panda() -> Image.Image:
    img, d = _new()
    # Black ears.
    d.ellipse((55, 60, 175, 180), fill=(30, 30, 30, 255))
    d.ellipse((337, 60, 457, 180), fill=(30, 30, 30, 255))
    _head(d, (250, 250, 250))
    # Black eye patches — drawn around the eye anchors so the eyes sit on them.
    ex, ey = EYE_L
    d.ellipse((ex - 55, ey - 60, ex + 55, ey + 60), fill=(30, 30, 30, 255))
    ex, ey = EYE_R
    d.ellipse((ex - 55, ey - 60, ex + 55, ey + 60), fill=(30, 30, 30, 255))
    _eyes(d)
    _nose(d)
    _mouth(d)
    return img


def cat() -> Image.Image:
    img, d = _new()
    gray = (180, 180, 190)
    # Triangular ears.
    d.polygon([(105, 175), (75, 30), (215, 155)], fill=gray + (255,))
    d.polygon([(297, 155), (437, 30), (407, 175)], fill=gray + (255,))
    d.polygon([(125, 165), (115, 70), (190, 150)], fill=(255, 200, 210, 255))
    d.polygon([(322, 150), (397, 70), (387, 165)], fill=(255, 200, 210, 255))
    _head(d, gray)
    _eyes(d)
    # Pink triangle nose.
    nx, ny = NOSE
    d.polygon([(nx - 22, ny - 15), (nx + 22, ny - 15), (nx, ny + 18)],
              fill=(230, 120, 140, 255))
    _mouth(d)
    # Whiskers.
    for y in (370, 390, 410):
        d.line((95, y, 215, y - 10), fill=(30, 30, 30, 180), width=3)
        d.line((297, y - 10, 417, y), fill=(30, 30, 30, 180), width=3)
    return img


def _feather(img: Image.Image, radius: int = FEATHER_RADIUS) -> Image.Image:
    """Blur the alpha channel at the mask perimeter so edges dissolve into the frame.

    Without this, the warped mask leaves a hard cartoon-on-photo seam — the exact
    'glued-on smiley' look we're trying to avoid.
    """
    r, g, b, a = img.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=radius))
    return Image.merge("RGBA", (r, g, b, a))


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
        _feather(img).save(p, "PNG")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
