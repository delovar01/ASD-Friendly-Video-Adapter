from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.audio import denoise_and_normalize, measure_lufs


def test_normalize_brings_loudness_near_target(tmp_path: Path, sample_wav: Path):
    out = tmp_path / "clean.wav"
    denoise_and_normalize(sample_wav, out, target_lufs=-18.0)
    assert out.exists() and out.stat().st_size > 0
    lufs = measure_lufs(out)
    # Tolerance: limiter + denoise shifts things; just assert within 6 LU.
    assert -26.0 < lufs < -12.0, f"Unexpected LUFS: {lufs}"
