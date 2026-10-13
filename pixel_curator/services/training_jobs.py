"""Shared service - persistent training job store (read by ui AND gui)."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_JOBS = REPO_ROOT / "projects" / "training_jobs.json"

TRAINING_TYPES = ("plan_only", "lora", "dreambooth", "unknown")
JOB_STATUSES = ("created", "planned", "queued", "running", "completed",
                "failed", "stopped", "unavailable")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class TrainingJob:
    job_id: str
    name: str = ""
    training_type: str = "plan_only"
    model_architecture: str = "unknown"
    device: str = "auto"
    dataset_path: str = ""
    package_path: str = ""
    output_path: str = ""
    base_model_path: str = ""
    base_model_profile: str = ""
    command: list = field(default_factory=list)
    status: str = "created"
    dry_run: bool = True
    confirmed: bool = False
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    log_path: str | None = None
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrainingJob:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def default_jobs_path(output_base: str | None = None) -> Path:
    if output_base:
        try:
            p = Path(output_base).expanduser()
            if p.is_dir() or p.parent.is_dir():
                return p / "training_jobs.json"
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
    return FALLBACK_JOBS


@dataclass
class TrainingJobStore:
    path: str = str(FALLBACK_JOBS)

    def _read(self) -> list:
        try:
            p = Path(self.path)
            if p.is_file():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("jobs"), list):
                    return data["jobs"]
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
        return []

    def _write(self, jobs: list) -> None:
        try:
            p = Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"jobs": jobs}, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass

    def list_jobs(self) -> list:
        return [TrainingJob.from_dict(j) for j in self._read()]

    def get_job(self, job_id: str) -> TrainingJob | None:
        for j in self._read():
            if j.get("job_id") == job_id:
                return TrainingJob.from_dict(j)
        return None

    def save_job(self, job: TrainingJob) -> None:
        jobs = [j for j in self._read() if j.get("job_id") != job.job_id]
        jobs.append(job.to_dict())
        self._write(jobs)

    def update_job(self, job_id: str, **updates) -> TrainingJob | None:
        jobs = self._read()
        for j in jobs:
            if j.get("job_id") == job_id:
                j.update({k: v for k, v in updates.items() if v is not None})
                self._write(jobs)
                return TrainingJob.from_dict(j)
        return None

    def delete_job_record(self, job_id: str) -> bool:
        jobs = self._read()
        kept = [j for j in jobs if j.get("job_id") != job_id]
        self._write(kept)
        return len(kept) < len(jobs)

    def as_rows(self) -> list:
        return [[j.job_id, j.training_type, j.model_architecture, j.device,
                 j.status, j.started_at or ""] for j in self.list_jobs()]


__all__ = ["TrainingJob", "TrainingJobStore", "default_jobs_path",
           "TRAINING_TYPES", "JOB_STATUSES"]
