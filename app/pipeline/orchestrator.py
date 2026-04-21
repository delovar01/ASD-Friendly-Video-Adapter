from __future__ import annotations

import logging
import shutil
import traceback
from pathlib import Path

from ..config import ensure_job_dirs
from ..models import JobStatus, update_job
from . import audio as audio_mod
from . import faces as faces_mod
from . import scenes as scenes_mod
from . import subtitles as subs_mod
from .ffmpeg_utils import (
    burn_subtitles,
    extract_audio,
    make_gif,
    mux_av,
    probe_duration,
)

log = logging.getLogger("pipeline")


def _set(job_id: str, status: JobStatus) -> None:
    log.info("[%s] -> %s", job_id, status)
    update_job(job_id, status=status)


def run_pipeline(job_id: str) -> None:
    dirs = ensure_job_dirs(job_id)
    src = dirs["input"] / "source.mp4"
    work = dirs["work"]
    out_dir = dirs["output"]

    try:
        # 2. Extract audio.
        _set(job_id, JobStatus.extracting_audio)
        audio_raw = work / "audio_raw.wav"
        extract_audio(src, audio_raw)

        # 3. Denoise + normalize.
        _set(job_id, JobStatus.denoising)
        audio_clean = work / "audio_clean.wav"
        audio_mod.denoise_and_normalize(audio_raw, audio_clean)

        # 4. Scenes.
        _set(job_id, JobStatus.detecting_scenes)
        scene_list = scenes_mod.detect_scenes(src)
        hard_cuts = scenes_mod.find_hard_cuts(scene_list)

        # 5. Replace faces -> silent video.
        _set(job_id, JobStatus.replacing_faces)
        video_faces = work / "video_faces.mp4"
        faces_mod.process_video(src, video_faces)

        # 6. Crossfades on hard cuts.
        _set(job_id, JobStatus.smoothing_cuts)
        video_smooth = work / "video_smooth.mp4"
        scenes_mod.apply_crossfades(video_faces, video_smooth, hard_cuts)

        # 7. Subtitles.
        _set(job_id, JobStatus.adding_subtitles)
        srt = work / "captions.srt"
        try:
            subs_mod.transcribe_to_srt(audio_clean, srt)
            has_subs = srt.exists() and srt.stat().st_size > 0
        except Exception as e:  # whisper can OOM or miss models
            log.warning("[%s] subtitles failed: %s", job_id, e)
            has_subs = False

        # 8. Mux + burn subs.
        _set(job_id, JobStatus.finalizing)
        muxed = work / "muxed.mp4"
        mux_av(video_smooth, audio_clean, muxed)

        output = out_dir / "adapted.mp4"
        if has_subs:
            burn_subtitles(muxed, srt, output)
        else:
            shutil.copyfile(muxed, output)

        # 9. Preview GIFs (before / after).
        try:
            dur = probe_duration(src)
            mid = max(0.0, dur / 2 - 2.5)
            before = out_dir / "preview_before.gif"
            after = out_dir / "preview_after.gif"
            make_gif(src, before, start=mid, duration=min(5.0, max(1.0, dur)))
            make_gif(output, after, start=mid, duration=min(5.0, max(1.0, dur)))
        except Exception as e:
            log.warning("[%s] preview gif failed: %s", job_id, e)
            before = None
            after = None

        # 10. Cleanup work dir, update job.
        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass

        update_job(
            job_id,
            status=JobStatus.done,
            output_path=str(output),
            preview_before=str(before) if before and before.exists() else None,
            preview_after=str(after) if after and after.exists() else None,
        )
        log.info("[%s] done", job_id)

    except Exception as e:
        log.error("[%s] failed: %s\n%s", job_id, e, traceback.format_exc())
        update_job(job_id, status=JobStatus.failed, error=str(e)[:500])
