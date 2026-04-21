import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))


def _ffmpeg_ok() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.fixture(scope="session")
def sample_video() -> Path:
    """A 10-second synthetic MP4 with one moving "face" (circle) + a tone.

    Generated with ffmpeg if not present. Skips if ffmpeg is unavailable.
    """
    dst = FIXTURES / "sample_10s.mp4"
    if dst.exists():
        return dst
    if not _ffmpeg_ok():
        pytest.skip("ffmpeg not available")
    FIXTURES.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=gray:s=320x240:d=10:r=24",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-vf", "drawbox=x='50+20*t':y=60:w=120:h=120:color=pink@1.0:t=fill",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


@pytest.fixture(scope="session")
def sample_wav(tmp_path_factory) -> Path:
    if not _ffmpeg_ok():
        pytest.skip("ffmpeg not available")
    dst = tmp_path_factory.mktemp("audio") / "tone.wav"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        "sine=frequency=440:duration=3, anoisesrc=color=white:duration=3:amplitude=0.1",
        "-filter_complex", "[0:a][1:a]amix=inputs=2",
        "-ac", "1", "-ar", "16000",
        str(dst),
    ], check=True)
    return dst
