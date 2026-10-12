"""DreamBooth dataset exporter — Colab Pro+ ready, from curated frames.

Turns the curated/identity-filtered frames already in the manifest into a
DreamBooth training folder you can zip to Google Drive and train on Colab
(A100). One export serves BOTH training modes:

  * full-checkpoint DreamBooth  (output: a ~6.5GB .safetensors base model)
  * DreamBooth-LoRA             (output: a portable LoRA)

…because the dataset is identical; only the trainer flag differs. The README
written into the export spells out both commands.

Selection: from `selected`/`packaged` frames that have an identity score, rank
by a composite of identity similarity and sharpness, then (by default) pick a
cluster-diverse round-robin so the set isn't 1,000 near-identical selfies.

Non-destructive: dry-run plans by default; a real run symlinks/copies frames
into the export (never moves or edits the source). Pure stdlib.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .. import manifest
from ..config import Project
from ..util import now_iso


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DreamBoothExportConfig:
    output_base: Path
    out_dir: Optional[Path] = None          # default: <output_base>/dreambooth_exports/<token>
    token: str = "spikbnu"
    class_word: str = "person"
    count: int = 1000                        # instance images to select
    repeats: int = 1                         # kohya repeat-prefix on the instance folder
    diversity: bool = True                   # cluster round-robin
    diverse: bool = False                    # full diversity curator (session+scene caps)
    quality_aware: bool = False              # rank with advanced-quality signals
    max_source_frac: float = 0.15            # used when diverse=True
    max_cluster_frac: float = 0.20
    reg: bool = False                        # build a regularization (class) set spec
    reg_count: int = 300
    link_mode: str = "symlink"               # symlink | hardlink | copy
    dry_run: bool = True
    base_model_name: str = "cyberrealisticPony_v180Coreshift.safetensors"
    training_config: bool = True             # Phase 10: write training_config.*
                                             # beside the export (suggestion only)
    # Lineage mode: reproduce the EXACT training set of a known LoRA. When set,
    # frames + captions come verbatim from that dataset build (no re-selection).
    dataset_version: Optional[int] = None
    from_run: Optional[str] = None           # resolve dataset_version from runs.name
    from_recipe: Optional[str] = None        # resolve dataset_version from datasets.recipe_name


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _composite_rank(rows: list[dict]) -> list[dict]:
    """Rank by 0.6*identity + 0.4*normalized-sharpness, best first."""
    sharps = [r["sharpness"] for r in rows if r["sharpness"] is not None]
    lo, hi = (min(sharps), max(sharps)) if sharps else (0.0, 1.0)
    span = (hi - lo) or 1.0
    for r in rows:
        ident = r["identity_sim"] if r["identity_sim"] is not None else 0.0
        sharp_norm = ((r["sharpness"] - lo) / span) if r["sharpness"] is not None else 0.0
        r["_score"] = round(0.6 * float(ident) + 0.4 * float(sharp_norm), 5)
    rows.sort(key=lambda r: r["_score"], reverse=True)
    return rows


def _parse_adv_signals(signals_json) -> dict:
    """Pull the ranking-relevant advanced-quality scores from a signals_json
    payload. Missing keys -> absent (treated neutrally by the scorer)."""
    import json as _json
    try:
        s = _json.loads(signals_json) if signals_json else {}
    except Exception:
        return {}
    out = {}
    for k in ("face_region_score", "region_balance_score",
              "subject_integrity_score", "figure_integrity_score",
              "aesthetic_score", "texture_preservation_score",
              "motion_blur_score", "face_occlusion_score",
              "exposure_outlier_score"):
        v = s.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    pd = s.get("pose_duplicate")
    if isinstance(pd, dict):
        out["pose_near_duplicate"] = bool(pd.get("near_duplicate")) and \
            not pd.get("representative", True)
    exo = s.get("exposure_outlier")
    if isinstance(exo, dict):
        out["exposure_bad"] = bool(exo.get("is_overexposed")
                                   or exo.get("is_underexposed"))
    return out


def _quality_composite_rank(rows: list[dict]) -> list[dict]:
    """Quality-aware rank: identity + sharpness stay the core (always present),
    blended with advanced signals when available and demoting blur / occlusion /
    bad exposure / pose near-duplicates. Frames lacking advanced signals fall
    back gracefully to the identity+sharpness core (never penalized for absence).
    """
    sharps = [r["sharpness"] for r in rows if r["sharpness"] is not None]
    lo, hi = (min(sharps), max(sharps)) if sharps else (0.0, 1.0)
    span = (hi - lo) or 1.0
    for r in rows:
        ident = float(r["identity_sim"]) if r["identity_sim"] is not None else 0.0
        sharp = ((r["sharpness"] - lo) / span) if r["sharpness"] is not None \
            else 0.0
        a = r.get("_adv") or {}
        # core (always available): identity dominant for a self-LoRA
        score = 0.45 * ident + 0.20 * sharp
        # quality bonuses (only counted when present; re-normalized weight)
        region = a.get("face_region_score")
        if region is None:
            region = a.get("region_balance_score")
        integ = a.get("subject_integrity_score") or \
            a.get("figure_integrity_score")
        aes = a.get("aesthetic_score")
        tex = a.get("texture_preservation_score")
        for val, w in ((region, 0.15), (integ, 0.10), (aes, 0.05), (tex, 0.05)):
            if isinstance(val, (int, float)):
                score += w * float(val)
        # penalties
        blur = a.get("motion_blur_score")
        if isinstance(blur, (int, float)):
            score -= 0.10 * float(blur)
        occ = a.get("face_occlusion_score")
        if isinstance(occ, (int, float)):
            score -= 0.08 * float(occ)
        if a.get("exposure_bad"):
            score -= 0.06
        if a.get("pose_near_duplicate"):
            score -= 0.12          # demote redundant poses
        r["_score"] = round(score, 5)
    rows.sort(key=lambda r: r["_score"], reverse=True)
    return rows


def _cluster_round_robin(ranked: list[dict], count: int) -> list[dict]:
    """Interleave clusters (preserving in-cluster rank) for a diverse pick."""
    buckets: dict = {}
    for r in ranked:                                  # ranked is already best-first
        buckets.setdefault(r.get("cluster_id"), []).append(r)
    order = sorted(buckets.keys(), key=lambda c: -len(buckets[c]))
    out: list[dict] = []
    i = 0
    while len(out) < count and any(buckets[c] for c in order):
        c = order[i % len(order)]
        if buckets[c]:
            out.append(buckets[c].pop(0))
        i += 1
    return out[:count]


def _has_advanced_quality(conn) -> bool:
    try:
        conn.execute("SELECT 1 FROM frame_advanced_quality LIMIT 1").fetchone()
        return True
    except Exception:
        return False


def select_instance_frames(conn, count: int, *, diversity: bool = True,
                           quality_aware: bool = False) -> list[dict]:
    """Pick the cream of the curated frames for DreamBooth instance images.

    With ``quality_aware`` and a populated ``frame_advanced_quality`` table, the
    ranking blends region readiness, subject integrity, aesthetic and texture,
    and demotes blur / occlusion / bad exposure / pose near-duplicates. Without
    it (or with no advanced signals), it falls back to the identity+sharpness
    composite - so behavior is unchanged for existing callers."""
    use_adv = quality_aware and _has_advanced_quality(conn)
    if use_adv:
        cur = conn.execute(
            "SELECT f.frame_id, f.path, f.sharpness, f.cluster_id, "
            "d.identity_sim, c.caption_text, a.signals_json "
            "FROM frames f JOIN detections d ON d.frame_id = f.frame_id "
            "LEFT JOIN captions c ON c.frame_id = f.frame_id "
            "LEFT JOIN frame_advanced_quality a ON a.frame_id = f.frame_id "
            "WHERE f.status IN ('selected','packaged') "
            "AND d.identity_sim IS NOT NULL")
        rows = [{"frame_id": r[0], "path": r[1], "sharpness": r[2],
                 "cluster_id": r[3], "identity_sim": r[4], "caption": r[5],
                 "_adv": _parse_adv_signals(r[6])} for r in cur.fetchall()]
    else:
        cur = conn.execute(
            "SELECT f.frame_id, f.path, f.sharpness, f.cluster_id, "
            "d.identity_sim, c.caption_text "
            "FROM frames f JOIN detections d ON d.frame_id = f.frame_id "
            "LEFT JOIN captions c ON c.frame_id = f.frame_id "
            "WHERE f.status IN ('selected','packaged') "
            "AND d.identity_sim IS NOT NULL")
        rows = [{"frame_id": r[0], "path": r[1], "sharpness": r[2],
                 "cluster_id": r[3], "identity_sim": r[4], "caption": r[5]}
                for r in cur.fetchall()]
    if not rows:
        return []
    ranked = (_quality_composite_rank(rows) if use_adv
              else _composite_rank(rows))
    if diversity:
        return _cluster_round_robin(ranked, count)
    return ranked[:count]


def resolve_run_dataset(conn, run_name: str) -> Optional[int]:
    """Map a training run name to the dataset version it trained on."""
    for sql, arg in (("SELECT dataset_version FROM runs WHERE name = ? "
                      "ORDER BY run_id DESC LIMIT 1", run_name),
                     ("SELECT dataset_version FROM runs WHERE name LIKE ? "
                      "ORDER BY run_id DESC LIMIT 1", f"{run_name}%")):
        try:
            row = conn.execute(sql, (arg,)).fetchone()
        except Exception:
            return None
        if row and row[0] is not None:
            return int(row[0])
    return None


def resolve_recipe_dataset(conn, recipe_name: str) -> Optional[int]:
    """Map a build recipe name to its most recent dataset version. Use this to
    export a built-but-not-yet-trained dataset (before any run exists)."""
    for sql, arg in (("SELECT version FROM datasets WHERE recipe_name = ? "
                      "ORDER BY version DESC LIMIT 1", recipe_name),
                     ("SELECT version FROM datasets WHERE recipe_name LIKE ? "
                      "ORDER BY version DESC LIMIT 1", f"{recipe_name}%")):
        try:
            row = conn.execute(sql, (arg,)).fetchone()
        except Exception:
            return None
        if row and row[0] is not None:
            return int(row[0])
    return None


def select_dataset_frames(conn, version: int) -> list[dict]:
    """Exact training frames + verbatim captions from a dataset build. This
    reproduces the precise set that trained a known-good LoRA."""
    cur = conn.execute(
        "SELECT df.frame_id, f.path, df.caption FROM dataset_frames df "
        "JOIN frames f ON f.frame_id = df.frame_id "
        "WHERE df.version = ? AND df.split = 'train' ORDER BY f.path",
        (version,))
    return [{"frame_id": r[0], "path": r[1], "caption": r[2],
             "cluster_id": None, "identity_sim": None, "verbatim": True}
            for r in cur.fetchall()]


def dataset_lineage(conn, version: int) -> dict:
    """Human-readable provenance for a dataset version (recipe, counts)."""
    try:
        row = conn.execute(
            "SELECT recipe_name, image_count, built_at, dir FROM datasets "
            "WHERE version = ?", (version,)).fetchone()
    except Exception:
        row = None
    runs = []
    try:
        runs = [r[0] for r in conn.execute(
            "SELECT name FROM runs WHERE dataset_version = ? ORDER BY run_id",
            (version,)).fetchall()]
    except Exception:
        pass
    return {"version": version,
            "recipe": row[0] if row else None,
            "image_count": row[1] if row else None,
            "built_at": row[2] if row else None,
            "dir": row[3] if row else None,
            "trained_runs": runs}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _instance_caption(row: dict, token: str, class_word: str) -> str:
    cap = (row.get("caption") or "").strip()
    trig = f"{token} {class_word}".strip()
    if not cap:
        return trig
    # ensure the trigger leads the caption (DreamBooth binds identity to it)
    return cap if trig.lower() in cap.lower() else f"{trig}, {cap}"


def _link(src: Path, dst: Path, mode: str, warnings: list[str]) -> bool:
    if dst.exists():
        return False
    try:
        if mode == "symlink":
            os.symlink(src, dst)
        elif mode == "hardlink":
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
        return True
    except Exception as exc:                              # pragma: no cover
        warnings.append(f"link failed for {src.name}: {exc}")
        return False


def plan_dreambooth_export(conn, prj: Project, cfg: DreamBoothExportConfig) -> dict:
    """Build the export plan (and write it on a real run). Read-only on source."""
    token = cfg.token or prj.trigger_token
    cls = cfg.class_word or prj.class_word
    out_root = Path(cfg.out_dir or
                    (Path(cfg.output_base) / "dreambooth_exports" /
                     token)).expanduser()
    instance_dirname = f"{cfg.repeats}_{token} {cls}"     # kohya repeat-prefix
    instance_dir = out_root / "instance" / instance_dirname

    warnings: list[str] = []
    # --- choose the source: lineage (exact training set) or fresh selection ---
    version = cfg.dataset_version
    if version is None and cfg.from_recipe:
        version = resolve_recipe_dataset(conn, cfg.from_recipe)
        if version is None:
            warnings.append(f"NO BUILT DATASET for recipe '{cfg.from_recipe}' "
                            "- run `build --recipe " + cfg.from_recipe + "` first. "
                            "Refusing to silently fall back to fresh selection.")
            return {"selected": 0, "samples": [], "warnings": warnings,
                    "status": "aborted", "source": "unresolved",
                    "instance_prompt": f"{token} {cls}", "requested": 0,
                    "clusters_covered": 0, "diversity": cfg.diversity,
                    "lineage": None, "captions": "-"}
    if version is None and cfg.from_run:
        version = resolve_run_dataset(conn, cfg.from_run)
        if version is None:
            warnings.append(f"could not resolve a dataset for run "
                            f"'{cfg.from_run}'; falling back to fresh selection")
    lineage = dataset_lineage(conn, version) if version is not None else None
    div_plan = None
    if version is not None:
        frames = select_dataset_frames(conn, version)
        exact = True
    elif cfg.diverse:
        # full diversity curator: cap per session + scene, balance framing
        from ..diverse import fetch_candidates, diverse_select
        cands = fetch_candidates(conn)
        try:
            caps = dict(conn.execute(
                "SELECT frame_id, caption_text FROM captions").fetchall())
        except Exception:
            caps = {}
        for f in cands:
            f["caption"] = caps.get(f["frame_id"])
        div_plan = diverse_select(cands, cfg.count,
                                  max_source_frac=cfg.max_source_frac,
                                  max_cluster_frac=cfg.max_cluster_frac)
        frames = div_plan["selected"]
        exact = False
        warnings.extend(div_plan.get("warnings", []))
    else:
        frames = select_instance_frames(conn, cfg.count, diversity=cfg.diversity,
                                        quality_aware=cfg.quality_aware)
        exact = False

    samples: list[dict] = []
    for i, row in enumerate(frames):
        src = Path(str(row["path"])).expanduser()
        stem = src.stem or f"img_{i:05d}"
        # lineage mode keeps the ORIGINAL captions verbatim (faithful repro);
        # fresh selection re-derives a trigger-led caption.
        cap = ((row.get("caption") or f"{token} {cls}") if exact
               else _instance_caption(row, token, cls))
        samples.append({
            "source_path": str(src),
            "image_target": f"instance/{instance_dirname}/{stem}{src.suffix or '.jpg'}",
            "caption_target": f"instance/{instance_dirname}/{stem}.txt",
            "caption": cap,
            "identity_sim": row.get("identity_sim"),
            "cluster_id": row.get("cluster_id"),
        })

    n_clusters = len({s["cluster_id"] for s in samples})
    plan = {
        "token": token, "class_word": cls,
        "instance_prompt": f"{token} {cls}",
        "class_prompt": cls,
        "selected": len(samples),
        "requested": (lineage["image_count"] if lineage else cfg.count),
        "source": ("dataset_version" if exact
                   else "diverse_curator" if div_plan else "fresh_selection"),
        "diversity_plan": ({"sources_used": div_plan["sources_used"],
                            "clusters_used": div_plan["clusters_used"],
                            "framing": div_plan["framing"],
                            "source_cap": div_plan["source_cap"],
                            "top_sources": div_plan["top_sources"]}
                           if div_plan else None),
        "dataset_version": version,
        "lineage": lineage,
        "captions": ("verbatim (original training captions)" if exact
                     else "re-derived (trigger-led)"),
        "clusters_covered": n_clusters,
        "diversity": cfg.diversity,
        "repeats": cfg.repeats,
        "regularization": bool(cfg.reg),
        "reg_count": cfg.reg_count if cfg.reg else 0,
        "base_model": cfg.base_model_name,
        "out_root": str(out_root),
        "instance_dir": str(instance_dir),
        "link_mode": cfg.link_mode,
        "dry_run": bool(cfg.dry_run),
        "samples": samples,
        "warnings": warnings,
        "created_at": now_iso(),
    }

    if cfg.dry_run:
        plan["status"] = "planned"
        try:
            plan_dir = out_root / "_plans"
            plan_dir.mkdir(parents=True, exist_ok=True)
            p = plan_dir / "dreambooth_plan.json"
            p.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            plan["plan_path"] = str(p)
        except Exception as exc:                          # pragma: no cover
            warnings.append(f"could not write plan: {exc}")
        return plan

    # --- real export ---
    instance_dir.mkdir(parents=True, exist_ok=True)
    copied = 0          # genuinely (re)written this run
    present = 0         # usable image present on disk after this run
    missing = 0         # selected frames whose source file is not on disk
    used_names: set[str] = set()
    for i, s in enumerate(samples):
        src = Path(s["source_path"])
        if not src.is_file():
            missing += 1
            warnings.append(f"source missing, skipped: {src.name}")
            continue
        img_target = s["image_target"]
        cap_target = s["caption_target"]
        # collision-proof: two distinct sources sharing a basename must not
        # overwrite each other (which silently drops a frame). Disambiguate.
        if img_target in used_names:
            stem, suffix = Path(img_target).stem, Path(img_target).suffix
            uniq = f"{stem}__{i:06d}"
            img_target = f"instance/{instance_dirname}/{uniq}{suffix}"
            cap_target = f"instance/{instance_dirname}/{uniq}.txt"
        used_names.add(img_target)
        dst = out_root / img_target
        # a stale symlink from a previous export must not block a real copy
        if dst.is_symlink() and cfg.link_mode != "symlink":
            try:
                dst.unlink()
            except Exception:
                pass
        if _link(src, dst, cfg.link_mode, warnings):
            copied += 1
            present += 1
        elif dst.exists():
            present += 1            # already correctly in place from a prior run
        (out_root / cap_target).write_text(s["caption"], encoding="utf-8")

    if cfg.reg:
        (out_root / "reg" / f"1_{cls}").mkdir(parents=True, exist_ok=True)

    _write_dataset_config(out_root, instance_dirname, cfg, token, cls)
    _write_readme(out_root, cfg, token, cls, len(samples))
    plan["copied_images"] = copied
    plan["images_present"] = present
    plan["missing_sources"] = missing
    plan["status"] = "exported"

    # Phase 10: ready-to-train trainer config beside the export (the
    # dataset_config.toml above defines the data; training_config.toml
    # defines the trainer args). Suggestion only - never fatal.
    if getattr(cfg, "training_config", True):
        from .export_config import generate_for_export
        result, msgs = generate_for_export(
            conn, out_root, dataset_version=version,
            frame_count=present or len(samples), model_type="dreambooth",
            base_model=cfg.base_model_name)
        plan["training_config_msgs"] = msgs
        if result:
            plan["training_config"] = result.get("paths")
    return plan


def _write_dataset_config(out_root: Path, instance_dirname: str,
                          cfg: DreamBoothExportConfig, token: str, cls: str) -> None:
    lines = [
        "# kohya sd-scripts dataset config (SDXL DreamBooth / DreamBooth-LoRA)",
        "[general]",
        "shuffle_caption = true",
        "caption_extension = '.txt'",
        "keep_tokens = 1",
        "",
        "[[datasets]]",
        "resolution = 1024",
        "batch_size = 1",
        "enable_bucket = true",
        "min_bucket_reso = 768",
        "max_bucket_reso = 1280",
        "",
        "  [[datasets.subsets]]",
        f"  image_dir = 'instance/{instance_dirname}'",
        f"  class_tokens = '{token} {cls}'",
        f"  num_repeats = {cfg.repeats}",
    ]
    if cfg.reg:
        lines += [
            "",
            "  [[datasets.subsets]]",
            f"  image_dir = 'reg/1_{cls}'",
            f"  class_tokens = '{cls}'",
            "  is_reg = true",
            "  num_repeats = 1",
        ]
    (out_root / "dataset_config.toml").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")


def _write_readme(out_root: Path, cfg: DreamBoothExportConfig,
                  token: str, cls: str, n: int) -> None:
    reg_note = ("A regularization folder (reg/1_%s) was created — drop ~%d "
                "class images there (generated 'photo of a %s' or curated "
                "generic shots) before training." % (cls, cfg.reg_count, cls)
                if cfg.reg else
                "No regularization set (recommended on a strong base like "
                "cyberrealisticPony with this many instance images). Add one "
                "later with --reg if you see class-forgetting.")
    txt = f"""# DreamBooth export — {token} {cls}

{n} instance images selected (identity × sharpness, cluster-diverse).
Trigger / instance prompt: `{token} {cls}`   |   class prompt: `{cls}`
Base model to train on: **{cfg.base_model_name}** (train ON Pony so the output
keeps the Pony aesthetic + your identity — do NOT train from raw SDXL base).

## 1. Upload
Zip this whole folder and upload to Google Drive. In Colab (request an **A100**
runtime on Pro+), mount Drive and unzip.

## 2a. Full-checkpoint DreamBooth (≈6.5GB .safetensors — strongest likeness)
kohya `sdxl_train.py` (full fine-tune). Key flags:
```
accelerate launch sdxl_train.py \\
  --pretrained_model_name_or_path "{cfg.base_model_name}" \\
  --dataset_config dataset_config.toml \\
  --output_dir /content/drive/MyDrive/db_out --output_name {token}_db_v001 \\
  --resolution 1024 --train_batch_size 1 --gradient_checkpointing \\
  --mixed_precision bf16 --save_precision fp16 \\
  --learning_rate 1e-6 --max_train_steps 1600 --optimizer_type AdamW8bit \\
  --save_every_n_epochs 1 --save_model_as safetensors
```

## 2b. DreamBooth-LoRA (portable LoRA — same dataset)
kohya `sdxl_train_network.py` (LoRA). Same dataset_config; add:
```
accelerate launch sdxl_train_network.py \\
  --pretrained_model_name_or_path "{cfg.base_model_name}" \\
  --dataset_config dataset_config.toml \\
  --network_module networks.lora --network_dim 32 --network_alpha 16 \\
  --output_dir /content/drive/MyDrive/db_lora --output_name {token}_dblora_v001 \\
  --resolution 1024 --train_batch_size 2 --gradient_checkpointing \\
  --mixed_precision bf16 --learning_rate 1e-4 --max_train_steps 2000 \\
  --save_every_n_epochs 1 --save_model_as safetensors
```

## 3. {('Regularization' )}
{reg_note}

## 4. Use the output at home
- **Full checkpoint** → drop into reForge `models/Stable-diffusion/`, select it as
  the checkpoint, generate. It IS your identity base — do NOT also load a
  character LoRA of yourself on top (double identity).
- **DreamBooth-LoRA** → drop into `models/Lora/`, load on top of cyberrealisticPony
  at weight ~0.8, prompt `{token} {cls}`.
- Either identity source stacks fine with NON-identity concept LoRAs
  (outfit / pose / style / detail) + civitai LoRAs.
"""
    (out_root / "README.md").write_text(txt, encoding="utf-8")


def render_plan(plan: dict) -> str:
    if plan.get("status") == "aborted":
        return ("\nDreamBooth export ABORTED\n" + "=" * 56 + "\n  "
                + "\n  ".join(plan.get("warnings", ["unresolved"])))
    lines = ["", "DreamBooth export plan", "=" * 56,
             "NOTE: the defined DreamBooth export path is now the selector "
             "(python -m pixel_curator.curate.dreambooth_nsfw_selector); this "
             "legacy exporter is kept for lineage reproduction.",
             f"instance prompt : {plan['instance_prompt']}",
             f"source          : {plan.get('source', 'fresh_selection')}",
             f"captions        : {plan.get('captions', '-')}",
             f"selected        : {plan['selected']} / {plan['requested']} "
             f"(across {plan['clusters_covered']} clusters, "
             f"diversity={plan['diversity']})",]
    lin = plan.get("lineage")
    if lin:
        lines.append(f"lineage         : dataset v{lin['version']} "
                     f"(recipe '{lin['recipe']}', {lin['image_count']} imgs, "
                     f"built {lin['built_at']})")
        if lin.get("trained_runs"):
            lines.append(f"  trained runs  : {', '.join(lin['trained_runs'])}")
    lines += [
             f"regularization  : {'yes (' + str(plan['reg_count']) + ')' if plan['regularization'] else 'no'}",
             f"link mode       : {plan['link_mode']}",
             f"out             : {plan['out_root']}",
             f"status          : {plan['status']}"]
    if plan.get("plan_path"):
        lines.append(f"plan written    : {plan['plan_path']}")
    if plan["status"] == "exported":
        lines.append(f"images written  : {plan.get('copied_images', 0)}")
        lines.append(f"images present  : {plan.get('images_present', 0)} "
                     f"/ {plan['selected']} (usable on disk)")
        if plan.get("missing_sources"):
            lines.append(f"missing sources : {plan['missing_sources']} "
                         f"(selected frames whose file is not on disk - stale paths)")
        for m in plan.get("training_config_msgs", []):
            lines.append(m)
        lines.append(f"next            : zip {plan['out_root']} -> Google Drive -> Colab (see README.md)")
    else:
        lines.append("dry-run: nothing written except the plan. Re-run with --apply to export.")
    for w in plan.get("warnings", [])[:8]:
        lines.append(f"  ! {w}")
    return "\n".join(lines)
