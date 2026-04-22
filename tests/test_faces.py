from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.pipeline.faces import (
    MASK_EYE_L,
    MASK_EYE_R,
    MASK_SIZE,
    _affine_eyes,
    _iou,
    _overlay_mask,
    _warp_and_blend,
)


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


def test_affine_eyes_maps_anchors_to_targets():
    # Source anchors (mask space) -> target anchors (frame space).
    dst_l = np.array([100.0, 120.0], dtype=np.float32)
    dst_r = np.array([200.0, 140.0], dtype=np.float32)  # slight roll
    M = _affine_eyes(MASK_EYE_L, MASK_EYE_R, dst_l, dst_r)
    ones = np.array([1.0], dtype=np.float32)
    mapped_l = M @ np.concatenate([np.array(MASK_EYE_L, dtype=np.float32), ones])
    mapped_r = M @ np.concatenate([np.array(MASK_EYE_R, dtype=np.float32), ones])
    np.testing.assert_allclose(mapped_l, dst_l, atol=1e-3)
    np.testing.assert_allclose(mapped_r, dst_r, atol=1e-3)


def test_warp_and_blend_replaces_face_region():
    # Solid-gray frame.
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    # Mask: green square over the eye anchor area, transparent elsewhere.
    mask_rgba = np.zeros((MASK_SIZE, MASK_SIZE, 4), dtype=np.uint8)
    mask_rgba[:, :, 1] = 255  # green
    mask_rgba[:, :, 3] = 255  # fully opaque

    # Place eyes roughly centered and horizontal — no roll.
    eye_l = np.array([140.0, 120.0], dtype=np.float32)
    eye_r = np.array([180.0, 120.0], dtype=np.float32)
    out = _warp_and_blend(frame, mask_rgba, eye_l, eye_r)

    assert out.shape == frame.shape
    # The pixel midway between the eyes should now be green, not gray.
    mid = out[120, 160]
    assert mid[1] >= 200  # BGR: green is channel 1
    assert mid[0] < 50 and mid[2] < 50
