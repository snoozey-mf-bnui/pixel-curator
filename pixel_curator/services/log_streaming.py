"""Shared service - live log tailing + background process manager.

Canonical Phase 3 process manager used by both UI surfaces. Provides the
Phase 3 API (``start_process`` / ``stop_process`` / ``get_status`` /
``tail_job_log`` / ``list_processes``) plus Phase 2 aliases (``start`` /
``stop`` / ``status`` / ``tail`` / ``list_jobs``) so existing callers keep
working. stdlib-only, thread-safe, import-safe (no Gradio/torch).
"""
from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


def new_job_id(prefix: str = "job") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"


@dataclass
class LogTailResult:
    log_path: str
    exists: bool
    text: str
    line_count: int
    truncated: bool = False
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"log_path": self.log_path, "exists": self.exists,
                "text": self.text, "line_count": self.line_count,
                "truncated": self.truncated, "warnings": list(self.warnings)}


def tail_log_file(log_path, max_lines: int = 200) -> LogTailResult:
    p = Path(log_path)
    if not p.is_file():
        return LogTailResult(str(p), False, f"(no log at {p})", 0)
    try:
        all_lines = p.read_text(encoding="utf-8",
                                errors="replace").splitlines()
        tail = all_lines[-max_lines:]
        return LogTailResult(str(p), True, "\n".join(tail), len(all_lines),
                             truncated=len(all_lines) > len(tail))
    except Exception as exc:
        return LogTailResult(str(p), True, f"(could not read log: {exc})", 0,
                             warnings=[str(exc)])


def discover_recent_logs(root, limit: int = 50) -> list:
    out = []
    try:
        r = Path(root)
        if not r.is_dir():
            return out
        logs = sorted((p for p in r.rglob("*.log") if p.is_file()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for p in logs[:limit]:
            try:
                st = p.stat()
                out.append({"path": str(p), "name": p.name,
                            "size": st.st_size,
                            "modified": datetime.fromtimestamp(
                                st.st_mtime).isoformat(timespec="seconds")})
            except Exception:
                continue
    except Exception:
        pass
    return out


@dataclass
class ManagedProcess:
    job_id: str
    command: list
    cwd: str
    log_path: str
    status: str = "pending"   # pending|running|completed|failed|stopped
    returncode: Optional[int] = None
    created_at: str = field(default_factory=lambda:
                            datetime.now().isoformat(timespec="seconds"))
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "command": list(self.command),
                "cwd": self.cwd, "log_path": self.log_path,
                "status": self.status, "returncode": self.returncode,
                "created_at": self.created_at, "started_at": self.started_at,
                "finished_at": self.finished_at}


class ProcessLogManager:
    """Runs + tracks background jobs; tees combined output to a log file."""

    def __init__(self):
        self._lock = threading.Lock()
        self._procs: dict = {}
        self._popens: dict = {}

    # -- Phase 3 API --------------------------------------------------------
    def start_process(self, command: list, cwd, log_path, *,
                      env: dict | None = None, job_id: str | None = None
                      ) -> dict:
        return self._start(command, cwd, log_path, env=env,
                           job_id=job_id).to_dict()

    def stop_process(self, job_id: str) -> bool:
        return self.stop(job_id)

    def get_status(self, job_id: str) -> dict:
        return self.status(job_id)

    def tail_job_log(self, job_id: str, max_lines: int = 200) -> LogTailResult:
        with self._lock:
            mp = self._procs.get(job_id)
        if not mp:
            return LogTailResult("", False, f"(no job {job_id})", 0)
        return tail_log_file(mp.log_path, max_lines)

    def list_processes(self) -> list:
        return self.list_jobs()

    # -- Phase 2-compatible aliases ----------------------------------------
    def start(self, command: list, cwd, log_path, *, env=None, job_id=None
              ) -> ManagedProcess:
        return self._start(command, cwd, log_path, env=env, job_id=job_id)

    def tail(self, job_id: str, max_lines: int = 200) -> str:
        return self.tail_job_log(job_id, max_lines).text

    def status(self, job_id: str) -> dict:
        with self._lock:
            mp = self._procs.get(job_id)
        return mp.to_dict() if mp else {"job_id": job_id, "status": "unknown"}

    def list_jobs(self) -> list:
        with self._lock:
            return [mp.to_dict() for mp in sorted(
                self._procs.values(), key=lambda m: m.created_at, reverse=True)]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for m in self._procs.values() if m.status == "running")

    # -- internals ----------------------------------------------------------
    def _start(self, command, cwd, log_path, *, env=None, job_id=None
               ) -> ManagedProcess:
        job_id = job_id or new_job_id()
        log_path = Path(log_path)
        mp = ManagedProcess(job_id=job_id, command=[str(c) for c in command],
                            cwd=str(cwd), log_path=str(log_path))
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                f"$ {' '.join(mp.command)}\n[{datetime.now().isoformat()}]\n",
                encoding="utf-8")
            proc = subprocess.Popen(
                mp.command, cwd=str(cwd), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            mp.status = "running"
            mp.started_at = datetime.now().isoformat(timespec="seconds")
            with self._lock:
                self._procs[job_id] = mp
                self._popens[job_id] = proc
            threading.Thread(target=self._pump, args=(job_id, proc, log_path),
                             daemon=True).start()
        except Exception as exc:
            mp.status = "failed"
            mp.finished_at = datetime.now().isoformat(timespec="seconds")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n[spawn error] {type(exc).__name__}: {exc}\n")
            except Exception:
                pass
            with self._lock:
                self._procs[job_id] = mp
        return mp

    def _pump(self, job_id, proc, log_path) -> None:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                for line in iter(proc.stdout.readline, ""):
                    f.write(line)
                    f.flush()
            proc.wait()
        except Exception:                                 # pragma: no cover
            pass
        with self._lock:
            mp = self._procs.get(job_id)
            if mp:
                mp.returncode = getattr(proc, "returncode", None)
                if mp.status != "stopped":
                    mp.status = "completed" if mp.returncode == 0 else "failed"
                mp.finished_at = datetime.now().isoformat(timespec="seconds")
            self._popens.pop(job_id, None)

    def stop(self, job_id: str) -> bool:
        with self._lock:
            proc = self._popens.get(job_id)
            mp = self._procs.get(job_id)
        if not proc:
            return False
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            if mp:
                mp.status = "stopped"
                mp.finished_at = datetime.now().isoformat(timespec="seconds")
            return True
        except Exception:
            return False


DEFAULT_MANAGER = ProcessLogManager()

__all__ = ["LogTailResult", "tail_log_file", "discover_recent_logs",
           "ManagedProcess", "ProcessLogManager", "DEFAULT_MANAGER",
           "new_job_id"]
