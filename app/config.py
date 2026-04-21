from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage"
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATES_DIR = BASE_DIR / "app" / "templates"
MASKS_DIR = STATIC_DIR / "masks"
DB_PATH = BASE_DIR / "jobs.db"

MAX_UPLOAD_MB = 200
MAX_DURATION_SEC = 10 * 60
ALLOWED_EXT = {".mp4", ".mov"}

TARGET_LUFS = -18.0
SCENE_THRESHOLD = 27.0
HARD_CUT_MAX_GAP = 0.4
CROSSFADE_DUR = 0.5
TARGET_FPS = 24

MASK_FILES = ["fox.png", "bear.png", "rabbit.png", "panda.png", "cat.png"]


def job_dir(job_id: str) -> Path:
    return STORAGE_DIR / job_id


def ensure_job_dirs(job_id: str) -> dict[str, Path]:
    root = job_dir(job_id)
    dirs = {
        "root": root,
        "input": root / "input",
        "work": root / "work",
        "output": root / "output",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs
