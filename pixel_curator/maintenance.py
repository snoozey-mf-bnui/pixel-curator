"""Phase 7: watch folders (auto-ingest) + disk GC + space dashboard.

watch  - polls the project's video/photo dirs; when new media appears,
         runs extraction (and optionally basic curation) automatically.
gc     - dry-run by default; reports orphaned thumbnails, superseded
         dataset builds, and dangling eval images. --apply deletes.
space  - per-area disk usage of the output volume.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from . import manifest
from .config import CurateConfig, ExtractConfig, Project

logger = logging.getLogger(__name__)
from .util import discover_photos, discover_videos, human_bytes, setup_logging, stable_id

# -----------------------------
# Watch / auto-ingest
# -----------------------------

@dataclass
class WatchConfig:
    output_base: Path
    interval: int = 300
    auto_curate: bool = False
    once: bool = False


def find_new_media(prj: Project, conn) -> tuple[list, list]:
    known = {r[0] for r in conn.execute("SELECT source_id FROM sources")}
    new_videos = [v for v in discover_videos(prj.video_dirs)
                  if stable_id(v) not in known]
    new_photos = [p for p in discover_photos(prj.photos_dir)
                  if stable_id(p) not in known]
    return new_videos, new_photos


def watch_generator(prj: Project, cfg: WatchConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)
    cycle = 0
    while True:
        cycle += 1
        new_videos, new_photos = find_new_media(prj, conn)
        if new_videos or new_photos:
            yield (f"[watch #{cycle}] NEW media: {len(new_videos)} videos, "
                   f"{len(new_photos)} photos - ingesting...")
            from .extract import pipeline_generator
            ecfg = ExtractConfig(output_base=cfg.output_base)
            for update in pipeline_generator(prj.video_dirs, prj.photos_dir, ecfg):
                yield f"[ingest] {update}"
            if cfg.auto_curate:
                from .curate import curate_generator
                for update in curate_generator(CurateConfig(output_base=cfg.output_base)):
                    yield f"[curate] {update}"
        else:
            yield (f"[watch #{cycle}] no new media "
                   f"({time.strftime('%H:%M:%S')}); next check in {cfg.interval}s")
        if cfg.once:
            return
        time.sleep(cfg.interval)


# -----------------------------
# GC
# -----------------------------

@dataclass
class GcConfig:
    output_base: Path
    apply: bool = False
    keep_builds: int = 2      # latest N builds kept per recipe


def gc_generator(prj: Project, cfg: GcConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)
    base = cfg.output_base
    mode = "APPLY" if cfg.apply else "DRY RUN (use --apply to delete)"
    yield f"GC [{mode}]\n"
    total_bytes = 0

    # 1) orphaned thumbnails
    thumbs = base / ".thumbs"
    orphans: list[Path] = []
    if thumbs.exists():
        frame_ids = {r[0] for r in conn.execute("SELECT frame_id FROM frames")}
        for t in thumbs.glob("*.jpg"):
            if t.stem not in frame_ids:
                orphans.append(t)
    sz = sum(t.stat().st_size for t in orphans)
    total_bytes += sz
    yield f"Orphan thumbnails: {len(orphans)} ({human_bytes(sz)})"
    if cfg.apply:
        for t in orphans:
            t.unlink(missing_ok=True)

    # 2) superseded dataset builds (keep newest N per recipe; manifest rows kept)
    by_recipe: dict[str, list] = {}
    for v, recipe, d in conn.execute(
        "SELECT version, recipe_name, dir FROM datasets ORDER BY version DESC"
    ):
        by_recipe.setdefault(recipe, []).append((v, Path(d)))
    superseded: list[tuple[int, Path]] = []
    for recipe, builds in by_recipe.items():
        superseded.extend(b for b in builds[cfg.keep_builds:] if b[1].exists())
    sz = 0
    for _v, d in superseded:
        sz += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    total_bytes += sz
    yield (f"Superseded builds (keep {cfg.keep_builds}/recipe): "
           f"{[f'v{v:03d}' for v, _ in superseded]} ({human_bytes(sz)})")
    if cfg.apply:
        for _v, d in superseded:
            shutil.rmtree(d, ignore_errors=True)

    # 3) dangling eval images (file exists, row deleted - or vice versa report)
    eval_rows = {r[0] for r in conn.execute("SELECT image_path FROM evals")}
    dangling: list[Path] = []
    evals_dir = base / "evals"
    if evals_dir.exists():
        for f in evals_dir.rglob("*.png"):
            if str(f) not in eval_rows:
                dangling.append(f)
    sz = sum(f.stat().st_size for f in dangling)
    total_bytes += sz
    yield f"Dangling eval images: {len(dangling)} ({human_bytes(sz)})"
    if cfg.apply:
        for f in dangling:
            f.unlink(missing_ok=True)

    yield (f"\nGC {'freed' if cfg.apply else 'would free'}: "
           f"{human_bytes(total_bytes)}")


# -----------------------------
# Space dashboard
# -----------------------------

def space_report(prj: Project, output_base: Path) -> list[dict]:
    areas = {
        "frames": output_base / "frames",
        "photos_imported": output_base / "photos_imported",
        "datasets": output_base / "DATASET",
        "evals": output_base / "evals",
        "thumbnails": output_base / ".thumbs",
        "runs(logs)": output_base / "runs",
        "lora_output": Path(prj.lora_output_dir
                            or (output_base / "LORA_OUTPUT")).expanduser(),
    }
    report = []
    for name, path in areas.items():
        size = n = 0
        if path.exists():
            for f in path.rglob("*"):
                if f.is_file():
                    n += 1
                    try:
                        size += f.stat().st_size
                    except OSError:
                        pass
        report.append({"area": name, "files": n, "bytes": size,
                       "human": human_bytes(size)})
    try:
        usage = shutil.disk_usage(output_base)
        report.append({"area": "VOLUME FREE", "files": "-",
                       "bytes": usage.free, "human": human_bytes(usage.free)})
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return report


def backup_project(prj, out_path=None):
    """One-command backup: manifest, presets, project file, LoRA sidecars
    (heavy media excluded).  Returns the archive path."""
    import tarfile
    import time
    from pathlib import Path

    from .config import DB_NAME
    out = Path(out_path) if out_path else (
        Path.home() / f"pixel_curator_backup_{time.strftime('%Y%m%d_%H%M')}.tar.gz")
    repo = Path(__file__).resolve().parent
    with tarfile.open(out, "w:gz") as tar:
        db = prj.output_path / DB_NAME
        if db.exists():
            tar.add(db, arcname=f"manifest/{DB_NAME}")
        presets = repo / "outputs" / "playground_presets.json"
        if presets.exists():
            tar.add(presets, arcname="outputs/playground_presets.json")
        for toml in repo.glob("*.toml"):
            tar.add(toml, arcname=f"project/{toml.name}")
        # LoRA sidecars (.concept.json / .json next to models)
        try:
            from .lora_explorer import lora_dirs
            for d in lora_dirs(prj):
                for sc in list(d.glob("*.concept.json")) + list(
                        d.glob("*.json")):
                    tar.add(sc, arcname=f"sidecars/{sc.name}")
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
    return out


def restore_project(archive, dest_dir):
    """Extract a backup archive into dest_dir for manual placement
    (never overwrites live files automatically)."""
    import tarfile
    from pathlib import Path
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    return dest
