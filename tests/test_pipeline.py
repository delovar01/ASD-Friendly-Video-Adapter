from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.config import STORAGE_DIR
from app.models import JobStatus, create_job, get_job, init_db
from app.pipeline.ffmpeg_utils import probe_duration
from app.pipeline.orchestrator import run_pipeline


@pytest.mark.slow
def test_end_to_end_short(sample_video: Path, monkeypatch, tmp_path: Path):
    # Isolate storage under tmp.
    from app import config as cfg
    monkeypatch.setattr(cfg, "STORAGE_DIR", tmp_path / "storage")
    (tmp_path / "storage").mkdir()

    init_db()
    job_id = "test0000abcd"
    dirs = cfg.ensure_job_dirs(job_id)
    shutil.copyfile(sample_video, dirs["input"] / "source.mp4")
    create_job(job_id, "sample_10s.mp4")

    run_pipeline(job_id)

    job = get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.done, f"status={job.status}, err={job.error}"
    out = Path(job.output_path)
    assert out.exists() and out.stat().st_size > 0
    # Output duration should be close to source (~10s).
    dur = probe_duration(out)
    assert 5.0 <= dur <= 15.0
