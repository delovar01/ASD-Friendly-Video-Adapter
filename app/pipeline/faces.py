from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from ..config import MASKS_DIR, MASK_FILES, TARGET_FPS

# Legacy detector kept importable — tests and older callers may reference it.
mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Mask anchor points (fractions of mask size) — must match scripts/generate_masks.py.
MASK_SIZE = 512
MASK_EYE_L = (MASK_SIZE * 0.35, MASK_SIZE * 0.48)
MASK_EYE_R = (MASK_SIZE * 0.65, MASK_SIZE * 0.48)

# Face Mesh landmark indices (MediaPipe's 468-point topology).
LEYE_IDX = (33, 133, 159, 145)   # left-eye outer/inner/top/bottom
REYE_IDX = (362, 263, 386, 374)  # right-eye outer/inner/top/bottom
CHEEK_L_IDX = 234
CHEEK_R_IDX = 454
FOREHEAD_IDX = 10
CHIN_IDX = 152

EMA_ALPHA = 0.45  # higher = more smoothing (less jitter, slightly laggier)


@dataclass
class Track:
    id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h — used for cross-frame matching
    mask_idx: int
    missed: int = 0
    # EMA state for landmark-derived anchors.
    eye_l: np.ndarray | None = None
    eye_r: np.ndarray | None = None
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
            img = Image.open(p).convert("RGBA")
            if img.size != (MASK_SIZE, MASK_SIZE):
                img = img.resize((MASK_SIZE, MASK_SIZE), Image.LANCZOS)
            masks.append(img)
    if not masks:
        raise FileNotFoundError(f"No masks found in {MASKS_DIR}")
    return masks


def _overlay_mask(frame_bgr: np.ndarray, mask: Image.Image,
                  bbox: tuple[int, int, int, int], pad: float = 0.15) -> np.ndarray:
    """Legacy bbox-based paste. Retained as a fallback when landmarks are unavailable."""
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox
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


def _landmark_xy(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _mean_landmarks(landmarks, idxs, w: int, h: int) -> np.ndarray:
    pts = np.stack([_landmark_xy(landmarks, i, w, h) for i in idxs])
    return pts.mean(axis=0)


def _bbox_from_landmarks(landmarks, w: int, h: int) -> tuple[int, int, int, int]:
    xs = np.array([lm.x for lm in landmarks], dtype=np.float32) * w
    ys = np.array([lm.y for lm in landmarks], dtype=np.float32) * h
    x0, y0 = int(max(0, xs.min())), int(max(0, ys.min()))
    x1, y1 = int(min(w, xs.max())), int(min(h, ys.max()))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _affine_eyes(src_l: tuple[float, float], src_r: tuple[float, float],
                 dst_l: np.ndarray, dst_r: np.ndarray) -> np.ndarray:
    """2x3 similarity transform that maps mask eye anchors onto detected eye centers."""
    svec = np.array([src_r[0] - src_l[0], src_r[1] - src_l[1]], dtype=np.float32)
    dvec = (dst_r - dst_l).astype(np.float32)
    scale = float(np.linalg.norm(dvec) / max(1e-6, np.linalg.norm(svec)))
    angle = float(np.arctan2(dvec[1], dvec[0]) - np.arctan2(svec[1], svec[0]))
    c, s = np.cos(angle), np.sin(angle)
    R = scale * np.array([[c, -s], [s, c]], dtype=np.float32)
    t = dst_l.astype(np.float32) - R @ np.array(src_l, dtype=np.float32)
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = R
    M[:, 2] = t
    return M


def _warp_and_blend(frame_bgr: np.ndarray, mask_rgba: np.ndarray,
                    eye_l: np.ndarray, eye_r: np.ndarray) -> np.ndarray:
    """Warp animal mask onto the frame so that its eye anchors align with detected eyes,
    then alpha-blend. The mask alpha channel is expected to be pre-feathered (see
    scripts/generate_masks.py) so the blend has no hard rectangular edges.
    """
    h, w = frame_bgr.shape[:2]
    M = _affine_eyes(MASK_EYE_L, MASK_EYE_R, eye_l, eye_r)
    warped = cv2.warpAffine(
        mask_rgba, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
    mask_bgr = cv2.cvtColor(warped[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    frame_f = frame_bgr.astype(np.float32)
    out = alpha * mask_bgr + (1.0 - alpha) * frame_f
    return np.clip(out, 0, 255).astype(np.uint8)


def _ema(prev: np.ndarray | None, curr: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return curr
    return (alpha * prev + (1.0 - alpha) * curr).astype(np.float32)


def process_video(
    src: Path,
    dst: Path,
    *,
    progress: Callable[[float], None] | None = None,
) -> None:
    """Replace human faces with cartoon animal faces using MediaPipe Face Mesh.

    The mask is warped (rotate + scale) so its eye anchors land on the detected eyes,
    then alpha-blended. Combined with pre-feathered mask edges and temporal smoothing,
    this gives a genuine face-replacement look rather than a sticker on a bounding box.
    """
    masks_pil = _load_masks()
    masks_rgba = [np.array(m) for m in masks_pil]  # uint8 RGBA, MASK_SIZE x MASK_SIZE

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

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)

            detections: list[dict] = []
            if res.multi_face_landmarks:
                for face in res.multi_face_landmarks:
                    lms = face.landmark
                    bbox = _bbox_from_landmarks(lms, w, h)
                    eye_l = _mean_landmarks(lms, LEYE_IDX, w, h)
                    eye_r = _mean_landmarks(lms, REYE_IDX, w, h)
                    detections.append({"bbox": bbox, "eye_l": eye_l, "eye_r": eye_r})

            # Greedy IoU matching of detections to existing tracks so each person
            # keeps the same animal mask across frames.
            assigned_tracks: set[int] = set()
            matches: dict[int, int] = {}  # det_idx -> track_idx
            for di, det in enumerate(detections):
                best_j, best_iou = -1, 0.2
                for tj, tr in enumerate(tracks):
                    if tj in assigned_tracks:
                        continue
                    score = _iou(det["bbox"], tr.bbox)
                    if score > best_iou:
                        best_iou = score
                        best_j = tj
                if best_j >= 0:
                    assigned_tracks.add(best_j)
                    matches[di] = best_j

            new_tracks: list[Track] = []
            for di, det in enumerate(detections):
                if di in matches:
                    tr = tracks[matches[di]]
                    tr.bbox = det["bbox"]
                    tr.missed = 0
                    tr.eye_l = _ema(tr.eye_l, det["eye_l"], EMA_ALPHA)
                    tr.eye_r = _ema(tr.eye_r, det["eye_r"], EMA_ALPHA)
                    tr.history.append(det["bbox"])
                    new_tracks.append(tr)
                else:
                    tr = Track(
                        id=next_track_id,
                        bbox=det["bbox"],
                        mask_idx=next_track_id % len(masks_rgba),
                        eye_l=det["eye_l"].astype(np.float32),
                        eye_r=det["eye_r"].astype(np.float32),
                        history=[det["bbox"]],
                    )
                    next_track_id += 1
                    new_tracks.append(tr)

            # Keep recently-lost tracks for a couple of frames so brief detection
            # misses don't strip the mask off (causes a "flash of real face").
            for tj, tr in enumerate(tracks):
                if tj in assigned_tracks:
                    continue
                tr.missed += 1
                if tr.missed <= 2 and tr.eye_l is not None and tr.eye_r is not None:
                    new_tracks.append(tr)

            tracks = new_tracks

            out = frame
            for tr in tracks:
                if tr.eye_l is None or tr.eye_r is None:
                    continue
                out = _warp_and_blend(out, masks_rgba[tr.mask_idx], tr.eye_l, tr.eye_r)

            writer.write(out)
            frame_idx += 1

            if progress and total > 0 and frame_idx % 30 == 0:
                progress(min(1.0, frame_idx / total))

    cap.release()
    writer.release()
    if progress:
        progress(1.0)
