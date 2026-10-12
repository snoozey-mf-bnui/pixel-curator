"""Shared utilities: logging, shell, hashing, discovery, ffmpeg capabilities."""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .config import LOG_NAME, PHOTO_EXTS, VIDEO_EXTS

try:
    from PIL import Image  # optional accelerator
    HAVE_PIL = True
except Exception:
    Image = None  # type: ignore
    HAVE_PIL = False

# -----------------------------
# Logging (file + stdout + in-memory buffer for the UI)
# -----------------------------

LOG_BUFFER: deque[str] = deque(maxlen=800)


class BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            LOG_BUFFER.append(self.format(record))
        except Exception:
            pass


def setup_logging(output_base: Path) -> None:
    output_base.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if getattr(root, "_pixel_curator_configured", False):
        return
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handlers = [
        logging.FileHandler(output_base / LOG_NAME, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
        BufferHandler(),
    ]
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    root.setLevel(logging.INFO)
    root._pixel_curator_configured = True  # type: ignore[attr-defined]


# -----------------------------
# Small helpers
# -----------------------------

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_cmd(
    cmd: list[str], *, check: bool = True, capture: bool = True, text: bool = True
) -> subprocess.CompletedProcess:
    logging.debug("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture, text=text)


def which(name: str) -> Optional[str]:
    return shutil.which(name)


def stable_id(path: Path) -> str:
    """Stable ID from absolute path + size + mtime. Collision-proof for
    repeated names like IMG_5662.mov vs IMG_5662.mp4."""
    try:
        st = path.stat()
        raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    except FileNotFoundError:
        raw = str(path.resolve())
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def safe_slug(text: str, max_len: int = 80) -> str:
    allowed: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            allowed.append(ch)
        elif ch in (" ", "(", ")", "[", "]", ","):
            allowed.append("_")
    slug = "".join(allowed)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:max_len] or "unnamed"


def parse_fps(rate: Optional[str]) -> Optional[float]:
    if not rate or rate == "0/0":
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        return float(rate)
    except Exception:
        return None


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# -----------------------------
# Discovery
# -----------------------------

def iter_files(root_dirs: Iterable[Path], exts: set[str]) -> Iterable[Path]:
    for root_dir in root_dirs:
        root_dir = Path(root_dir).expanduser()
        if not root_dir.exists():
            logging.warning("Missing directory: %s", root_dir)
            continue
        for root, _, files in os.walk(root_dir):
            for name in files:
                if name.startswith("."):
                    continue
                p = Path(root) / name
                if p.suffix.lower() in exts:
                    yield p


def discover_videos(video_dirs: list[str]) -> list[Path]:
    return sorted(iter_files([Path(d) for d in video_dirs], VIDEO_EXTS))


def discover_photos(photo_dir: str) -> list[Path]:
    p = Path(photo_dir).expanduser()
    if not p.exists():
        return []
    return sorted(iter_files([p], PHOTO_EXTS))


# -----------------------------
# ffmpeg capabilities (cached)
# -----------------------------

_FFMPEG_CAPS: dict[str, bool] = {}


def ffmpeg_has_videotoolbox() -> bool:
    if "vtb" not in _FFMPEG_CAPS:
        ok = False
        if which("ffmpeg"):
            try:
                proc = run_cmd(["ffmpeg", "-hide_banner", "-hwaccels"], check=False)
                ok = "videotoolbox" in ((proc.stdout or "") + (proc.stderr or "")).lower()
            except Exception:
                ok = False
        _FFMPEG_CAPS["vtb"] = ok
    return _FFMPEG_CAPS["vtb"]


def ffmpeg_has_filter(name: str) -> bool:
    key = f"filter:{name}"
    if key not in _FFMPEG_CAPS:
        ok = False
        if which("ffmpeg"):
            try:
                proc = run_cmd(["ffmpeg", "-hide_banner", "-filters"], check=False)
                ok = f" {name} " in ((proc.stdout or "") + (proc.stderr or ""))
            except Exception:
                ok = False
        _FFMPEG_CAPS[key] = ok
    return _FFMPEG_CAPS[key]


def ffmpeg_vsync_args() -> list[str]:
    """ffmpeg >= 5.1 uses -fps_mode; older builds use -vsync."""
    if "fps_mode" not in _FFMPEG_CAPS:
        ok = False
        if which("ffmpeg"):
            try:
                proc = run_cmd(["ffmpeg", "-hide_banner", "-h", "long"], check=False)
                ok = "fps_mode" in ((proc.stdout or "") + (proc.stderr or ""))
            except Exception:
                ok = False
        _FFMPEG_CAPS["fps_mode"] = ok
    return ["-fps_mode", "vfr"] if _FFMPEG_CAPS["fps_mode"] else ["-vsync", "vfr"]


def check_dependencies() -> dict:
    result = {
        "ffmpeg": bool(which("ffmpeg")),
        "ffprobe": bool(which("ffprobe")),
        "personfromvid": bool(which("personfromvid")),
        "pillow": HAVE_PIL,
        "videotoolbox": False,
        "zscale": False,
        "insightface": False,
        "torch_mps": None,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    if result["ffmpeg"]:
        result["videotoolbox"] = ffmpeg_has_videotoolbox()
        result["zscale"] = ffmpeg_has_filter("zscale")
    try:
        import insightface  # type: ignore  # noqa: F401
        result["insightface"] = True
    except Exception:
        pass
    try:
        import torch  # type: ignore
        result["torch_mps"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except Exception:
        result["torch_mps"] = None
    if not result["ffmpeg"]:
        logging.warning("ffmpeg missing. Install: brew install ffmpeg")
    return result


def project_data_plane(prj) -> dict:
    """Fail-loud readiness checks for the project's data plane.

    Silent "empty dashboard" states usually mean the output base is
    unreachable or the project file was never finished — doctor surfaces
    that here instead.  Returns ``{"ok": bool, "checks": {...}}`` where
    each check is ``{"level": "ok"|"warn"|"error", "detail": str}``.
    ``ok`` is False when any error-level check fails.
    """
    from pathlib import Path as _P
    import os

    from .config import DB_NAME

    checks: dict = {}

    def _add(name: str, level: str, detail: str) -> None:
        checks[name] = {"level": level, "detail": detail}

    out = prj.output_path
    if not str(out):
        _add("output_path", "error", "output_base is empty")
    elif out.exists():
        if out.is_dir() and os.access(out, os.W_OK):
            _add("output_path", "ok", f"{out} (writable)")
        else:
            _add("output_path", "error",
                 f"{out} exists but is not a writable directory")
    else:
        parent = out.parent
        if parent.exists() and os.access(parent, os.W_OK):
            _add("output_path", "warn",
                 f"{out} does not exist yet (will be created on first run)")
        else:
            _add("output_path", "error",
                 f"{out} unreachable and parent {parent} is not writable - "
                 f"check mounts (external SSD attached?) and output_base")

    token = str(getattr(prj, "trigger_token", "") or "").strip()
    if not token or token.upper() == "XXXX":
        _add("trigger_token", "error",
             "trigger_token is empty / still a placeholder")
    else:
        _add("trigger_token", "ok", token)

    vdirs = [d for d in (getattr(prj, "video_dirs", None) or []) if str(d).strip()]
    pdir = str(getattr(prj, "photos_dir", "") or "").strip()
    missing_src = []
    for d in vdirs:
        if not _P(d).expanduser().is_dir():
            missing_src.append(f"video_dir {d}")
    if pdir and not _P(pdir).expanduser().is_dir():
        missing_src.append(f"photos_dir {pdir}")
    if not vdirs and not pdir:
        _add("sources", "error",
             "no video_dirs or photos_dir configured - nothing to curate")
    elif missing_src:
        _add("sources", "warn",
             "missing (skipped if intentional): " + "; ".join(missing_src))
    else:
        _add("sources", "ok",
             f"{len(vdirs)} video dir(s)" + (f", photos {pdir}" if pdir else ""))

    if (out / DB_NAME).is_file():
        _add("manifest", "ok", f"{out / DB_NAME}")
    else:
        _add("manifest", "warn",
             "no manifest yet - run: extract / photos / curate")

    domain = str(getattr(prj, "domain", "human") or "human").lower()
    if domain in ("human", "object", "animal", "general"):
        _add("domain", "ok", domain)
    else:
        _add("domain", "error",
             f"unknown domain {domain!r} "
             f"(expected human | object | animal | general)")

    ok = not any(c["level"] == "error" for c in checks.values())
    return {"ok": ok, "domain": domain, "checks": checks}
