from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from .config import DB_PATH


class JobStatus(str, Enum):
    queued = "queued"
    extracting_audio = "extracting_audio"
    denoising = "denoising"
    detecting_scenes = "detecting_scenes"
    replacing_faces = "replacing_faces"
    smoothing_cuts = "smoothing_cuts"
    adding_subtitles = "adding_subtitles"
    finalizing = "finalizing"
    done = "done"
    failed = "failed"


STEP_PROGRESS = {
    JobStatus.queued: 0,
    JobStatus.extracting_audio: 5,
    JobStatus.denoising: 15,
    JobStatus.detecting_scenes: 25,
    JobStatus.replacing_faces: 55,
    JobStatus.smoothing_cuts: 70,
    JobStatus.adding_subtitles: 85,
    JobStatus.finalizing: 95,
    JobStatus.done: 100,
    JobStatus.failed: 100,
}

STEP_LABEL_RU = {
    JobStatus.queued: "В очереди",
    JobStatus.extracting_audio: "Извлекаем аудио",
    JobStatus.denoising: "Чистим звук",
    JobStatus.detecting_scenes: "Ищем сцены",
    JobStatus.replacing_faces: "Меняем лица на животных",
    JobStatus.smoothing_cuts: "Сглаживаем склейки",
    JobStatus.adding_subtitles: "Добавляем субтитры",
    JobStatus.finalizing: "Собираем финал",
    JobStatus.done: "Готово",
    JobStatus.failed: "Ошибка",
}


class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)
    original_name: str
    status: JobStatus = Field(default=JobStatus.queued)
    error: Optional[str] = None
    output_path: Optional[str] = None
    preview_before: Optional[str] = None
    preview_after: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def create_job(job_id: str, original_name: str) -> Job:
    with Session(engine) as s:
        job = Job(id=job_id, original_name=original_name)
        s.add(job)
        s.commit()
        s.refresh(job)
        return job


def get_job(job_id: str) -> Job | None:
    with Session(engine) as s:
        return s.get(Job, job_id)


def update_job(job_id: str, **fields) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if not job:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        job.updated_at = datetime.utcnow()
        s.add(job)
        s.commit()


def list_jobs(limit: int = 20) -> list[Job]:
    with Session(engine) as s:
        q = select(Job).order_by(Job.created_at.desc()).limit(limit)
        return list(s.exec(q))
