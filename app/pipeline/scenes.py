from __future__ import annotations

from pathlib import Path

from scenedetect import ContentDetector, open_video, SceneManager

from ..config import CROSSFADE_DUR, HARD_CUT_MAX_GAP, SCENE_THRESHOLD
from .ffmpeg_utils import probe_duration, run_ffmpeg


def detect_scenes(src: Path, threshold: float = SCENE_THRESHOLD) -> list[tuple[float, float]]:
    """Return list of (start_sec, end_sec) scenes."""
    video = open_video(str(src))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video=video, show_progress=False)
    scenes = sm.get_scene_list()
    out: list[tuple[float, float]] = []
    for s in scenes:
        out.append((s[0].get_seconds(), s[1].get_seconds()))
    if not out:
        # Single-scene fallback.
        dur = probe_duration(src)
        out = [(0.0, dur)]
    return out


def find_hard_cuts(scenes: list[tuple[float, float]],
                   max_gap: float = HARD_CUT_MAX_GAP) -> list[float]:
    """Return list of scene-boundary timestamps marked as hard cuts.

    A cut is hard if the scene it starts is shorter than `max_gap`.
    """
    cuts: list[float] = []
    for i in range(1, len(scenes)):
        start_i, end_i = scenes[i]
        if (end_i - start_i) < max_gap:
            cuts.append(start_i)
    return cuts


def apply_crossfades(src: Path, dst: Path, cuts: list[float],
                     dur: float = CROSSFADE_DUR) -> None:
    """If there are several hard cuts, re-encode with xfade transitions.

    For MVP we keep this simple: if fewer than 3 hard cuts, just copy.
    Otherwise, we split at each cut and xfade successively.
    """
    if len(cuts) < 3:
        run_ffmpeg(["-i", str(src), "-c", "copy", str(dst)])
        return

    # Build segments.
    total = probe_duration(src)
    boundaries = [0.0] + sorted(cuts) + [total]
    segments: list[tuple[float, float]] = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s > 0.05:
            segments.append((s, e))

    if len(segments) < 2:
        run_ffmpeg(["-i", str(src), "-c", "copy", str(dst)])
        return

    # Build filter_complex: trim each segment, then xfade sequentially.
    inputs: list[str] = []
    for s, e in segments:
        inputs.extend([
            "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
            "-i", str(src),
        ])

    filter_parts: list[str] = []
    for i in range(len(segments)):
        filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")

    prev = "v0"
    offset = segments[0][1] - segments[0][0]
    for i in range(1, len(segments)):
        out_label = f"x{i}"
        filter_parts.append(
            f"[{prev}][v{i}]xfade=transition=fade:duration={dur}:offset={max(0.0, offset - dur):.3f}[{out_label}]"
        )
        prev = out_label
        offset += (segments[i][1] - segments[i][0]) - dur

    filter_complex = ";".join(filter_parts)
    run_ffmpeg([
        *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{prev}]",
        "-an",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p",
        str(dst),
    ])
