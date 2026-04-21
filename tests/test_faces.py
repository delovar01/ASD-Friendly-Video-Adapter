from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.pipeline.faces import _iou, _overlay_mask


def test_iou_known_values():
    # Same box -> 1.0
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    # Disjoint -> 0.0
    assert _iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0
    # Half overlap
    assert _iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_overlay_mask_changes_pixels(tmp_path: Path):
    # Solid-gray frame.
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    # Opaque red mask.
    mask = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    out = _overlay_mask(frame, mask, (100, 60, 64, 64), pad=0.0)
    assert out.shape == frame.shape
    # Some pixel inside the bbox must have changed (red channel high).
    region = out[60:124, 100:164]
    assert region[:, :, 2].max() >= 200  # BGR: red is channel 2
    # Corners outside the bbox still gray.
    assert tuple(out[0, 0]) == (128, 128, 128)
