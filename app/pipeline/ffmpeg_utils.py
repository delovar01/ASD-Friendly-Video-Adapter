from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FFmpegError(
            "FFmpeg/ffprobe not found in PATH. Install from https://ffmpeg.org/"
        )


def run_ffmpeg(args: list[str], *, overwrite: bool = True) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if overwrite:
        cmd.append("-y")
    cmd.extend(args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed ({proc.returncode}):\n{proc.stderr.strip()}"
        )


def probe(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe", "-hide_banner", "-loglevel", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def probe_duration(path: Path) -> float:
    data = probe(path)
    fmt = data.get("format", {})
    dur = fmt.get("duration")
    if dur is None:
        for s in data.get("streams", []):
            if s.get("duration"):
                dur = s["duration"]
                break
    return float(dur) if dur else 0.0


def probe_video_stream(path: Path) -> dict:
    for s in probe(path).get("streams", []):
        if s.get("codec_type") == "video":
            return s
    raise FFmpegError(f"No video stream in {path}")


def extract_audio(src: Path, dst: Path, sample_rate: int = 16000) -> None:
    run_ffmpeg([
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "wav", str(dst),
    ])


def mux_av(video: Path, audio: Path, dst: Path, *, crf: int = 23, abr: str = "128k") -> None:
    run_ffmpeg([
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", abr,
        "-shortest",
        str(dst),
    ])


def burn_subtitles(video_in: Path, srt: Path, dst: Path) -> None:
    # Escape path for subtitles filter (Windows-safe).
    srt_str = str(srt).replace("\\", "/").replace(":", "\\:")
    style = (
        "FontName=PT Sans,FontSize=24,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,"
        "Alignment=2,MarginV=30"
    )
    vf = f"subtitles='{srt_str}':force_style='{style}'"
    run_ffmpeg([
        "-i", str(video_in),
        "-vf", vf,
        "-c:a", "copy",
        "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p",
        str(dst),
    ])


def make_gif(src: Path, dst: Path, *, start: float, duration: float = 5.0,
             width: int = 480, fps: int = 12) -> None:
    vf = (
        f"fps={fps},scale={width}:-1:flags=lanczos,"
        "split[a][b];[a]palettegen[p];[b][p]paletteuse"
    )
    run_ffmpeg([
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(src),
        "-filter_complex", vf,
        "-loop", "0",
        str(dst),
    ])
