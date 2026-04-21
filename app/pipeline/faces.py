from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from ..config import MASKS_DIR, MASK_FILES, TARGET_FPS

mp_face = mp.solutions.face_detection


@dataclass
class Track:
    id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    mask_idx: int
    missed: int = 0
    history: list[tuple[int, int, int, int]] = field(default_factory=list)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _load_masks() -> list[Image.Image]:
    masks: list[Image.Image] = []
    for name in MASK_FILES:
        p = MASKS_DIR / name
        if p.exists():
            masks.append(Image.open(p).convert("RGBA"))
    if not masks:
        raise FileNotFoundError(f"No masks found in {MASKS_DIR}")
    return masks


def _overlay_mask(frame_bgr: np.ndarray, mask: Image.Image,
                  bbox: tuple[int, int, int, int], pad: float = 0.15) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox
    # Expand.
    px = int(bw * pad)
    py = int(bh * pad)
    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(w, x + bw + px)
    y1 = min(h, y + bh + py)
    tw, th = x1 - x0, y1 - y0
    if tw <= 0 or th <= 0:
        return frame_bgr

    scaled = mask.resize((tw, th), Image.LANCZOS)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(rgb)
    base.paste(scaled, (x0, y0), scaled)
    return cv2.cvtColor(np.array(base), cv2.COLOR_RGB2BGR)


def process_video(
    src: Path,
    dst: Path,
    *,
    progress: Callable[[float], None] | None = None,
) -> None:
    """Read src, replace faces with animal masks, write silent video to dst."""
    masks = _load_masks()
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {src}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = min(src_fps, float(TARGET_FPS))
    frame_stride = max(1, int(round(src_fps / out_fps)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, out_fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for {dst}")

    tracks: list[Track] = []
    next_track_id = 0
    frame_idx = 0
    kept_idx = 0

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = det.process(rgb)

            current_boxes: list[tuple[int, int, int, int]] = []
            if res.detections:
                for d in res.detections:
                    rb = d.location_data.relative_bounding_box
                    bx = max(0, int(rb.xmin * w))
                    by = max(0, int(rb.ymin * h))
                    bw = max(1, int(rb.width * w))
                    bh = max(1, int(rb.height * h))
                    current_boxes.append((bx, by, bw, bh))

            # Match current boxes to existing tracks (greedy IoU).
            assigned_tracks: set[int] = set()
            matches: dict[int, int] = {}  # box_idx -> track_idx
            for bi, box in enumerate(current_boxes):
                best_j, best_iou = -1, 0.3
                for tj, tr in enumerate(tracks):
                    if tj in assigned_tracks:
                        continue
                    score = _iou(box, tr.bbox)
                    if score > best_iou:
                        best_iou = score
                        best_j = tj
                if best_j >= 0:
                    assigned_tracks.add(best_j)
                    matches[bi] = best_j

            new_tracks: list[Track] = []
            used_box_ids: set[int] = set()
            for bi, box in enumerate(current_boxes):
                if bi in matches:
                    tr = tracks[matches[bi]]
                    tr.bbox = box
                    tr.missed = 0
                    tr.history.append(box)
                    new_tracks.append(tr)
                    used_box_ids.add(bi)
                else:
                    tr = Track(
                        id=next_track_id,
                        bbox=box,
                        mask_idx=hash(next_track_id) % len(masks),
                        history=[box],
                    )
                    next_track_id += 1
                    new_tracks.append(tr)
                    used_box_ids.add(bi)

            # Keep unmatched tracks briefly (interpolate up to 2 frames).
            for tj, tr in enumerate(tracks):
                if tj in assigned_tracks:
                    continue
                tr.missed += 1
                if tr.missed <= 2:
                    new_tracks.append(tr)

            tracks = new_tracks

            # Apply masks.
            out = frame
            for tr in tracks:
                out = _overlay_mask(out, masks[tr.mask_idx], tr.bbox)

            writer.write(out)
            kept_idx += 1
            frame_idx += 1

            if progress and total > 0 and frame_idx % 30 == 0:
                progress(min(1.0, frame_idx / total))

    cap.release()
    writer.release()
    if progress:
        progress(1.0)
