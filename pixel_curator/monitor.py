"""Run monitoring: disk guard, live status, checkpoint watch, notifications.

Born from RUN #2/#3: a full boot volume starves macOS swap and MPSGraph
temp files, so the OS SIGKILLs training mid-epoch with nothing saved.
The guard refuses to launch below `min_free_gb` on the BOOT volume (the
output volume is checked too, but boot is the one that kills runs).

`pixel-curator status` reads the same manifest tables the trainer already
updates every ~1.5 s, so it works for a run started before this module
existed - no log parsing, no interference with the training process.

`pixel-curator status --follow N` polls, prints a compact line, and fires
a macOS notification (osascript) when an epoch checkpoint lands, when
the run finishes/fails, or when boot disk drops below the warn level.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import manifest
from .config import Project

BOOT_PATH = "/"
DEFAULT_MIN_FREE_GB = 20.0     # refuse to launch training below this (boot)
WARN_FREE_GB = 8.0             # mid-run warning threshold (boot)
STALL_SECONDS = 900.0          # 'running' but no new step in 15 min -> stalled


# -----------------------------
# Disk
# -----------------------------

def free_gb(path: str | Path = BOOT_PATH) -> float | None:
    try:
        return shutil.disk_usage(str(path)).free / 1024 ** 3
    except OSError:
        return None


def disk_volumes(prj: Project, output_base: Path | None = None) -> list[dict]:
    """Boot volume + every distinct volume the project writes to."""
    candidates: list[tuple[str, Path]] = [("boot (Macintosh HD)", Path(BOOT_PATH))]
    lora_out = Path(prj.lora_output_dir
                    or (Path(prj.output_base) / "LORA_OUTPUT")).expanduser()
    candidates.append(("lora_output", lora_out))
    candidates.append(("output_base", output_base or prj.output_path))

    seen: set[int] = set()
    out: list[dict] = []
    for label, p in candidates:
        probe = p
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            dev = probe.stat().st_dev
        except OSError:
            continue
        if dev in seen:
            continue
        seen.add(dev)
        gb = free_gb(probe)
        out.append({"label": label, "path": str(p),
                    "free_gb": round(gb, 1) if gb is not None else None})
    return out


def disk_guard(prj: Project, min_free_gb: float = DEFAULT_MIN_FREE_GB,
               output_base: Path | None = None) -> tuple[bool, str]:
    """(ok, report). Boot volume below the floor -> hard fail with fix tips.
    A cramped output volume only warns - checkpoints are small."""
    if min_free_gb <= 0:
        return True, "disk guard disabled (--min-free-gb 0)"
    lines: list[str] = []
    ok = True
    for vol in disk_volumes(prj, output_base):
        gb = vol["free_gb"]
        if gb is None:
            continue
        flag = ""
        if vol["label"].startswith("boot") and gb < min_free_gb:
            ok = False
            flag = f"  << BELOW {min_free_gb:g} GB floor"
        elif gb < 5:
            flag = "  << low"
        lines.append(f"  {vol['label']:<22} {gb:>8.1f} GB free{flag}")
    report = "Disk:\n" + "\n".join(lines)
    if not ok:
        report += (
            "\nFATAL: boot volume too full to train safely. macOS swap and "
            "MPSGraph temp files live there; running out mid-epoch gets the "
            "process SIGKILLed (this killed RUN #2/#3).\n"
            "Free space (move zips/caches to the SSD, empty Trash, "
            "`conda clean --all`), verify with `df -h /`, then retry.\n"
            "Override at your own risk: --min-free-gb 0"
        )
    return ok, report


# -----------------------------
# Notifications (macOS; silently a no-op elsewhere)
# -----------------------------

def notify_macos(title: str, message: str, sound: str = "Glass") -> bool:
    if sys.platform != "darwin" or shutil.which("osascript") is None:
        return False
    script = (f'display notification "{_esc(message)}" '
              f'with title "{_esc(title)}" sound name "{_esc(sound)}"')
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True,
                       timeout=10, check=False)
        return True
    except Exception:
        return False


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


# -----------------------------
# Status snapshot
# -----------------------------

def latest_run(conn) -> dict | None:
    """The running run if any, else the most recent one."""
    for clause in ("WHERE status='running'", ""):
        row = conn.execute(
            "SELECT run_id, name, dataset_version, preset, status, "
            "started_at, finished_at, current_step, total_steps, "
            "current_epoch, total_epochs, last_loss, error "
            f"FROM runs {clause} ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        if row:
            keys = ("run_id", "name", "dataset_version", "preset", "status",
                    "started_at", "finished_at", "step", "total_steps",
                    "epoch", "total_epochs", "last_loss", "error")
            return dict(zip(keys, row))
    return None


def last_progress_ts(conn, run_id: int) -> str | None:
    """ISO timestamp of the most recent recorded training step, or None."""
    row = conn.execute(
        "SELECT MAX(ts) FROM run_metrics WHERE run_id=?", (run_id,)).fetchone()
    return row[0] if row and row[0] else None


def stall_check(run: dict | None, last_ts: str | None, *,
                now: datetime | None = None,
                stall_seconds: float = STALL_SECONDS) -> dict:
    """Pure: is a 'running' run idle (no new step) beyond the threshold?

    A run can sit in status='running' while the trainer is actually hung, OOM,
    or hard-killed - the status never flips. We detect that by how long it has
    been since the last recorded step (falling back to started_at)."""
    if not run or run.get("status") != "running":
        return {"stalled": False, "idle_seconds": None}
    ref = last_ts or run.get("started_at")
    if not ref:
        return {"stalled": False, "idle_seconds": None}
    try:
        ref_dt = datetime.fromisoformat(ref)
    except (ValueError, TypeError):
        return {"stalled": False, "idle_seconds": None}
    idle = ((now or datetime.now()) - ref_dt).total_seconds()
    if idle < 0:
        idle = 0.0
    return {"stalled": idle > stall_seconds, "idle_seconds": round(idle, 1)}


def step_rate(conn, run_id: int, window: int = 60) -> float | None:
    """Seconds per step over the most recent metric window."""
    rows = conn.execute(
        "SELECT step, ts FROM run_metrics WHERE run_id=? "
        "ORDER BY step DESC LIMIT ?", (run_id, window),
    ).fetchall()
    if len(rows) < 2:
        return None
    (s1, t1), (s0, t0) = rows[0], rows[-1]
    try:
        dt = (datetime.fromisoformat(t1) - datetime.fromisoformat(t0)).total_seconds()
    except (ValueError, TypeError):
        return None
    return dt / (s1 - s0) if s1 > s0 and dt > 0 else None


def fmt_eta(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "-"
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h {rem // 60:02d}m"


def checkpoints_for(prj: Project, run_name: str) -> list[Path]:
    lora_dir = Path(prj.lora_output_dir
                    or (Path(prj.output_base) / "LORA_OUTPUT")).expanduser()
    if not lora_dir.exists():
        return []
    return sorted(lora_dir.glob(f"{run_name}*.safetensors"),
                  key=lambda p: p.stat().st_mtime)


def status_snapshot(prj: Project, output_base: Path, *,
                    stall_seconds: float = STALL_SECONDS) -> dict:
    conn = manifest.connect(output_base, readonly=True)
    run = latest_run(conn)
    snap: dict = {"run": run, "disks": disk_volumes(prj, output_base),
                  "checkpoints": [], "eta_seconds": None, "sec_per_it": None,
                  "stalled": False, "idle_seconds": None}
    if not run:
        return snap
    snap["checkpoints"] = [p.name for p in checkpoints_for(prj, run["name"])]
    st = stall_check(run, last_progress_ts(conn, run["run_id"]),
                     stall_seconds=stall_seconds)
    snap["stalled"] = st["stalled"]
    snap["idle_seconds"] = st["idle_seconds"]
    if run["status"] == "running" and run["total_steps"]:
        rate = step_rate(conn, run["run_id"])
        snap["sec_per_it"] = round(rate, 2) if rate else None
        if rate:
            remaining = run["total_steps"] - run["step"]
            snap["eta_seconds"] = remaining * rate
            # next epoch checkpoint ETA
            per_epoch = run["total_steps"] // max(run["total_epochs"], 1)
            if per_epoch:
                to_next = per_epoch - (run["step"] % per_epoch)
                snap["next_checkpoint_eta_seconds"] = to_next * rate
    return snap


def render_status(snap: dict) -> str:
    run = snap.get("run")
    if not run:
        return "No training runs recorded yet."
    lines = []
    pct = (100 * run["step"] / run["total_steps"]) if run["total_steps"] else 0
    lines.append(
        f"RUN #{run['run_id']} '{run['name']}'  [{run['status'].upper()}]")
    lines.append(
        f"  epoch {run['epoch']}/{run['total_epochs']} | "
        f"step {run['step']}/{run['total_steps']} ({pct:.1f}%) | "
        f"loss {run['last_loss']}")
    if run["status"] == "running":
        lines.append(
            f"  speed {snap.get('sec_per_it') or '-'} s/it | "
            f"ETA {fmt_eta(snap.get('eta_seconds'))} | "
            f"next checkpoint in {fmt_eta(snap.get('next_checkpoint_eta_seconds'))}")
        if snap.get("stalled"):
            lines.append(
                f"  ** STALLED: no new step in {fmt_eta(snap.get('idle_seconds'))} "
                "- process may be hung, OOM, or killed")
    elif run.get("error"):
        lines.append(f"  error: {run['error']}")
    cps = snap.get("checkpoints") or []
    lines.append(f"  checkpoints on disk: {len(cps)}"
                 + (f"  (latest: {cps[-1]})" if cps else ""))
    for vol in snap.get("disks", []):
        gb = vol["free_gb"]
        warn = "  << LOW" if gb is not None and gb < WARN_FREE_GB else ""
        lines.append(f"  disk {vol['label']:<14} {gb} GB free{warn}")
    return "\n".join(lines)


# -----------------------------
# Follow mode (poll + notify)
# -----------------------------

@dataclass
class FollowConfig:
    output_base: Path
    interval: float = 30.0
    notify: bool = True
    max_polls: int = 0          # 0 = until run leaves 'running' (test hook)
    stall_seconds: float = STALL_SECONDS


def follow_generator(prj: Project, cfg: FollowConfig) -> Generator[str, None, None]:
    known: set[str] | None = None
    warned_disk = False
    warned_stall = False
    last_status: str | None = None
    polls = 0

    while True:
        snap = status_snapshot(prj, cfg.output_base,
                               stall_seconds=cfg.stall_seconds)
        run = snap.get("run")
        yield render_status(snap)
        if not run:
            return

        cps = set(snap.get("checkpoints") or [])
        if known is None:                      # baseline - don't notify history
            known = cps
        new = sorted(cps - known)
        known |= cps
        for name in new:
            msg = f"Epoch checkpoint saved: {name}"
            yield f"*** {msg}"
            if cfg.notify:
                notify_macos("pixel-curator", msg)

        boot = next((v["free_gb"] for v in snap.get("disks", [])
                     if v["label"].startswith("boot")), None)
        if (boot is not None and boot < WARN_FREE_GB and not warned_disk):
            warned_disk = True
            msg = f"Boot disk low: {boot} GB free - training may be killed"
            yield f"*** WARNING: {msg}"
            if cfg.notify:
                notify_macos("pixel-curator", msg, sound="Basso")

        # stall: 'running' but no new step for a while -> alert once, re-arm
        # if progress resumes
        if snap.get("stalled") and not warned_stall:
            warned_stall = True
            msg = (f"RUN #{run['run_id']} STALLED - no new step in "
                   f"{fmt_eta(snap.get('idle_seconds'))} (hung, OOM, or killed?)")
            yield f"*** WARNING: {msg}"
            if cfg.notify:
                notify_macos("pixel-curator", msg, sound="Basso")
        elif not snap.get("stalled"):
            warned_stall = False

        if last_status == "running" and run["status"] != "running":
            msg = f"RUN #{run['run_id']} {run['status'].upper()}"
            yield f"*** {msg}"
            if cfg.notify:
                notify_macos("pixel-curator", msg)
            return
        last_status = run["status"]
        if run["status"] != "running" and polls > 0:
            return

        polls += 1
        if cfg.max_polls and polls >= cfg.max_polls:
            return
        time.sleep(cfg.interval)


def status_json(prj: Project, output_base: Path) -> str:
    return json.dumps(status_snapshot(prj, output_base), indent=2, default=str)
