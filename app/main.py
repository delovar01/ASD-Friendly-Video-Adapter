from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import (
    ALLOWED_EXT,
    MAX_DURATION_SEC,
    MAX_UPLOAD_MB,
    STATIC_DIR,
    STORAGE_DIR,
    TEMPLATES_DIR,
    ensure_job_dirs,
)
from .models import STEP_LABEL_RU, STEP_PROGRESS, JobStatus, create_job, get_job, init_db
from .pipeline.ffmpeg_utils import FFmpegError, ensure_ffmpeg, probe_duration
from .pipeline.orchestrator import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="ASD-Friendly Video Adapter")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _startup() -> None:
    try:
        ensure_ffmpeg()
    except FFmpegError as e:
        logging.error(str(e))
        raise
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/jobs")
async def create_job_route(
    background: BackgroundTasks,
    file: UploadFile = File(...),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Неподдерживаемый формат: {ext}. Разрешены: {sorted(ALLOWED_EXT)}")

    job_id = uuid.uuid4().hex[:12]
    dirs = ensure_job_dirs(job_id)
    src = dirs["input"] / "source.mp4"

    size = 0
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    with src.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > max_bytes:
                f.close()
                shutil.rmtree(dirs["root"], ignore_errors=True)
                raise HTTPException(413, f"Файл больше {MAX_UPLOAD_MB} МБ")
            f.write(chunk)

    try:
        dur = probe_duration(src)
    except FFmpegError as e:
        shutil.rmtree(dirs["root"], ignore_errors=True)
        raise HTTPException(400, f"Не удалось прочитать видео: {e}")

    if dur <= 0:
        shutil.rmtree(dirs["root"], ignore_errors=True)
        raise HTTPException(400, "В файле не обнаружено видео")
    if dur > MAX_DURATION_SEC:
        shutil.rmtree(dirs["root"], ignore_errors=True)
        raise HTTPException(400, f"Ролик длиннее {MAX_DURATION_SEC // 60} минут")

    create_job(job_id, original_name=file.filename or "video.mp4")
    background.add_task(run_pipeline, job_id)

    return JSONResponse({"job_id": job_id, "redirect": f"/jobs/{job_id}"})


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(job_id: str, request: Request):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return templates.TemplateResponse("job.html", {"request": request, "job": job})


@app.get("/jobs/{job_id}/status", response_class=HTMLResponse)
def job_status_fragment(job_id: str, request: Request):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    progress = STEP_PROGRESS.get(job.status, 0)
    label = STEP_LABEL_RU.get(job.status, str(job.status))
    response = templates.TemplateResponse(
        "_status.html",
        {"request": request, "job": job, "progress": progress, "label": label},
    )
    # HTTP 286 tells HTMX to stop polling once the work is finished.
    if job.status in (JobStatus.done, JobStatus.failed):
        response.status_code = 286
    return response


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    job = get_job(job_id)
    if not job or job.status != JobStatus.done or not job.output_path:
        raise HTTPException(404, "Результат не готов")
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(404, "Файл отсутствует")
    return FileResponse(path, filename="adapted.mp4", media_type="video/mp4")


@app.get("/jobs/{job_id}/preview/{kind}")
def preview(job_id: str, kind: str):
    if kind not in {"before", "after"}:
        raise HTTPException(404)
    job = get_job(job_id)
    if not job:
        raise HTTPException(404)
    path_str = job.preview_before if kind == "before" else job.preview_after
    if not path_str:
        raise HTTPException(404)
    path = Path(path_str)
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/gif")


@app.get("/jobs/{job_id}/video/{kind}")
def video_file(job_id: str, kind: str):
    if kind not in {"source", "adapted"}:
        raise HTTPException(404)
    job = get_job(job_id)
    if not job:
        raise HTTPException(404)
    if kind == "source":
        path = STORAGE_DIR / job_id / "input" / "source.mp4"
    else:
        path = Path(job.output_path) if job.output_path else None
    if not path or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4")
