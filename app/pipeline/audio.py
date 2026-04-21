from __future__ import annotations

from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .ffmpeg_utils import run_ffmpeg
from ..config import TARGET_LUFS


def denoise_and_normalize(src_wav: Path, dst_wav: Path, *, target_lufs: float = TARGET_LUFS) -> None:
    """Denoise + de-ess + loudness-normalize the input WAV."""
    y, sr = librosa.load(str(src_wav), sr=None, mono=True)

    if y.size == 0:
        sf.write(str(dst_wav), y, sr)
        return

    y_dn = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.85)

    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(y_dn)
    except ValueError:
        loudness = target_lufs
    if np.isfinite(loudness):
        y_norm = pyln.normalize.loudness(y_dn, loudness, target_lufs)
    else:
        y_norm = y_dn

    peak = float(np.max(np.abs(y_norm))) if y_norm.size else 0.0
    if peak > 0.99:
        y_norm = y_norm * (0.99 / peak)

    tmp = dst_wav.with_suffix(".pre.wav")
    sf.write(str(tmp), y_norm.astype(np.float32), sr, subtype="PCM_16")

    # De-esser + soft limiter via ffmpeg.
    run_ffmpeg([
        "-i", str(tmp),
        "-af", "equalizer=f=7000:width_type=h:width=4000:g=-3,"
               "acompressor=threshold=-14dB:ratio=4:attack=5:release=50",
        str(dst_wav),
    ])
    tmp.unlink(missing_ok=True)


def measure_lufs(wav: Path) -> float:
    y, sr = librosa.load(str(wav), sr=None, mono=True)
    if y.size == 0:
        return float("-inf")
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(y))
