"""Phase 4: Dataset Composer - recipe-driven, versioned, reproducible builds.

A *recipe* ([recipes.NAME] in the project file) = filters + quotas + repeats
+ caption policy. A *build* materializes one recipe (or a multi-concept combo)
into DATASET/vNNN__name/ with a dataset.json snapshot and manifest rows.

Guarantees:
  - Deterministic: identical manifest state -> identical content hash.
  - Val split is stable per-frame (hash of frame_id), not random.
  - Non-destructive: hardlinks; sources and frames untouched.

CLI:
  pixel-curator build --recipe character_v1
  pixel-curator datasets
  pixel-curator diff 2 3
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from . import manifest
from .config import Project
from .packager import allocate_quota, parse_quota, round_robin_clusters
from .util import now_iso, safe_slug, setup_logging


@dataclass
class BuildConfig:
    output_base: Path
    recipe: str
    note: str = ""
    dataset_root: str = "DATASET"
    link_mode: str = "hardlink"
    parent: int | None = None    # explicit lineage: version this build
                                    # was derived/refined from (Phase 2)
    training_config: bool = True    # Phase 10: write training_config.*
                                    # beside the dataset (suggestion only)


RECIPE_DEFAULTS = {
    "type": "character",
    "repeats": 10,
    "max_total": 0,
    "max_per_video": 40,
    "quota": "",
    "framing": "",
    "include_clusters": "",
    "exclude_clusters": "",
    "exclude_caption_tags": "",   # drop frames whose caption has any of these
                                  # tags (1boy,2girls,...) - multi-person reject
    "shard": "",               # "N/M" -> take the Nth of M equal slices of the
                               # eligible pool (for multi-LoRA ensembles that
                               # each train on a different slice of the data)
    "min_identity": 0.0,
    "min_sharpness": 0.0,
    "val_fraction": 0.0,
    "token": "",
    "class_word": "",
    "caption_static": "",
    "smart_crop": False,       # subject-centered crop to SDXL buckets
    "bucket_base": 1024,       # 1024 standard, 768 for lighter runs
    # Study Intelligence Layer filters (v7) - blank/0 = inactive
    "study_primary": "",       # e.g. figure_study_candidate
    "study_min_confidence": 0.0,
    "study_export_only": False,  # require study_export_eligible = 1
}


def resolve_concepts(prj: Project, name: str) -> list[tuple[str, dict]]:
    """A recipe either IS a concept, or lists other recipes via `concepts`."""
    recipes = prj.recipes or {}
    if name not in recipes:
        raise KeyError(
            f"Recipe '{name}' not found. Available: {', '.join(sorted(recipes)) or '(none)'}"
        )
    r = {**RECIPE_DEFAULTS, **recipes[name]}
    concept_list = str(r.get("concepts", "")).strip()
    if concept_list:
        out = []
        for cname in (c.strip() for c in concept_list.split(",") if c.strip()):
            if cname not in recipes:
                raise KeyError(f"Concept recipe '{cname}' (in '{name}') not found")
            sub = {**RECIPE_DEFAULTS, **recipes[cname]}
            # combo-level val_fraction cascades unless concept overrides
            if not sub.get("val_fraction") and r.get("val_fraction"):
                sub["val_fraction"] = r["val_fraction"]
            out.append((cname, sub))
        return out
    return [(name, r)]


def _csv_set(spec) -> set[str]:
    return {s.strip() for s in str(spec or "").split(",") if s.strip()}


def stable_split(frame_id: str, val_fraction: float) -> str:
    """Deterministic train/val assignment from the frame id itself."""
    if val_fraction <= 0:
        return "train"
    bucket = int(hashlib.sha1(frame_id.encode()).hexdigest()[:6], 16) % 10000
    return "val" if bucket < int(val_fraction * 10000) else "train"


def _fetch_candidate_rows(conn) -> list:
    """All curated frames eligible for any recipe (12 columns)."""
    return conn.execute(
        "SELECT f.frame_id, f.source_id, f.path, f.sharpness, f.cluster_id, "
        "d.framing, c.caption_text, d.identity_sim, d.bbox, "
        "s.study_primary, s.study_confidence, s.study_export_eligible "
        "FROM frames f "
        "LEFT JOIN detections d ON d.frame_id = f.frame_id "
        "LEFT JOIN captions c ON c.frame_id = f.frame_id "
        "LEFT JOIN study_labels s ON s.frame_id = f.frame_id "
        "WHERE f.status IN ('selected','packaged') "
        "ORDER BY f.frame_id"
    ).fetchall()


def _filter_rows(rows: list, rcp: dict) -> list:
    """Apply a recipe's filters (no capping). Returns the 9-column rows that
    match: framing / clusters / identity / sharpness / study filters."""
    framing_allow = _csv_set(rcp["framing"])
    inc_clusters = _csv_set(rcp["include_clusters"])
    exc_clusters = _csv_set(rcp["exclude_clusters"])
    # Drop frames whose WD14 caption contains any of these tags (e.g. 1boy,
    # 2girls, multiple_girls) - the multi-person / other-people reject. Needs
    # captions to exist (run `caption` first); exact-token match, so "man"
    # won't match "woman".
    excl_cap_tags = {_norm_tag(t) for t in _csv_set(rcp.get("exclude_caption_tags", ""))}
    min_id = float(rcp["min_identity"] or 0)
    min_sharp = float(rcp["min_sharpness"] or 0)

    study_primary = str(rcp.get("study_primary") or "")
    study_min_conf = float(rcp.get("study_min_confidence") or 0)
    study_export_only = bool(rcp.get("study_export_only"))

    filtered = []
    for r in rows:
        (frame_id, sid, path, sharp, cluster, framing, caption, ident, _bbox,
         s_primary, s_conf, s_export) = r
        r = r[:9]   # downstream selection logic uses the original 9 columns
        if study_primary and (s_primary or "") != study_primary:
            continue
        if study_min_conf > 0 and (s_conf is None or s_conf < study_min_conf):
            continue
        if study_export_only and not s_export:
            continue
        if framing_allow and (framing or "none") not in framing_allow:
            continue
        cstr = str(cluster) if cluster is not None else ""
        if inc_clusters and cstr not in inc_clusters:
            continue
        if exc_clusters and cstr and cstr in exc_clusters:
            continue
        if min_id > 0 and (ident is None or ident < min_id):
            continue
        if min_sharp > 0 and (sharp or 0) < min_sharp:
            continue
        if excl_cap_tags and (_caption_tokens(caption) & excl_cap_tags):
            continue
        filtered.append(r)
    return filtered


def _norm_tag(t: str) -> str:
    return str(t or "").strip().lower().replace("_", " ")


def _caption_tokens(caption) -> set[str]:
    """Comma-separated WD14 caption -> normalized tag tokens (exact-match set)."""
    return {_norm_tag(t) for t in str(caption or "").split(",") if t.strip()}


def apply_shard(rows: list, shard_spec: str) -> list:
    """Take the Nth of M equal contiguous slices of ``rows`` (1-indexed "N/M").

    Slices are cut from a deterministic order (sharpness desc, then frame_id),
    so shard 1/3, 2/3, 3/3 partition the pool into disjoint thirds. Empty or
    malformed spec returns ``rows`` unchanged.
    """
    spec = str(shard_spec or "").strip()
    if "/" not in spec:
        return rows
    try:
        n_str, m_str = spec.split("/", 1)
        n, m = int(n_str), int(m_str)
    except ValueError:
        return rows
    if m <= 1 or not (1 <= n <= m):
        return rows
    # Round-robin (interleaved) over a sharpness-sorted order, so every shard
    # is a representative quality mix of the pool - not "shard 1 = all sharp,
    # shard 3 = all soft". Each shard is disjoint and balanced.
    ordered = sorted(rows, key=lambda r: (-(r[3] or 0.0), r[0]))
    return ordered[n - 1::m]


def select_for_recipe(conn, rcp: dict) -> list:
    """Returns rows (frame_id, source_id, path, sharpness, cluster_id, framing,
    caption) after filters + quota/diversity selection. Deterministic."""
    rows = _fetch_candidate_rows(conn)
    filtered = _filter_rows(rows, rcp)
    filtered = apply_shard(filtered, str(rcp.get("shard", "")))

    has_clusters = any(r[4] is not None for r in filtered)
    has_framing = any(r[5] for r in filtered)

    # per-video cap with cluster diversity
    per_source: dict[str, list] = defaultdict(list)
    for r in filtered:
        per_source[r[1]].append(r)
    capped: list = []
    max_pv = int(rcp["max_per_video"] or 0)
    for sid in sorted(per_source):
        frames = sorted(per_source[sid], key=lambda r: (-(r[3] or 0.0), r[0]))
        cap = max_pv if max_pv > 0 else len(frames)
        capped.extend(round_robin_clusters(frames, cap) if has_clusters
                      else frames[:cap])

    max_total = int(rcp["max_total"] or 0)
    quota = parse_quota(str(rcp["quota"])) if (rcp["quota"] and has_framing) else {}
    if quota and max_total > 0:
        buckets: dict[str, list] = defaultdict(list)
        for r in capped:
            buckets[r[5] or "none"].append(r)
        alloc = allocate_quota({b: len(fs) for b, fs in buckets.items()},
                               min(max_total, len(capped)), quota)
        chosen = []
        for b in sorted(alloc):
            frames = sorted(buckets.get(b, []), key=lambda r: (-(r[3] or 0.0), r[0]))
            n = alloc[b]
            chosen.extend(round_robin_clusters(frames, n) if has_clusters
                          else frames[:n])
    elif max_total > 0:
        capped.sort(key=lambda r: (-(r[3] or 0.0), r[0]))
        chosen = capped[:max_total]
    else:
        chosen = capped

    chosen.sort(key=lambda r: r[0])  # deterministic order for hashing
    return chosen


def recipe_plan(conn, name: str, rcp: dict) -> dict:
    """Preview one recipe without writing anything: how many frames MATCH its
    filters (available) vs how many are SELECTED after caps/quota, plus the
    framing breakdown of the selection."""
    rows = _fetch_candidate_rows(conn)
    filtered = _filter_rows(rows, rcp)
    selected = select_for_recipe(conn, rcp)

    def _framing_counts(rs):
        out: dict[str, int] = defaultdict(int)
        for r in rs:
            out[r[5] or "none"] += 1
        return dict(sorted(out.items()))

    return {
        "recipe": name,
        "type": str(rcp.get("type") or ""),
        "filter": (str(rcp.get("framing") or "any framing")
                   + (f", id>={rcp['min_identity']}" if float(rcp.get("min_identity") or 0) else "")
                   + (f", sharp>={rcp['min_sharpness']}" if float(rcp.get("min_sharpness") or 0) else "")
                   + (f", clusters={rcp['include_clusters']}" if str(rcp.get("include_clusters") or "") else "")),
        "available": len(filtered),
        "selected": len(selected),
        "cap": int(rcp.get("max_total") or 0),
        "repeats": int(rcp.get("repeats") or 1),
        "selected_framing": _framing_counts(selected),
    }


def plan_generator(prj: Project, output_base: Path,
                   names: list[str] | None = None) -> Generator[str, None, None]:
    """Print an availability/selection table for every (non-combo) recipe.
    Non-destructive: reads the manifest only, writes no files."""
    setup_logging(output_base)
    conn = manifest.connect(output_base)

    recipes = prj.recipes or {}
    if not recipes:
        yield "No recipes defined in the project file."
        return

    # default: every recipe that is a concept (skip combos that just list others)
    if names:
        targets = [n for n in names if n in recipes]
    else:
        targets = [n for n in sorted(recipes)
                   if not str(recipes[n].get("concepts", "")).strip()]

    total_selected = 0
    yield (f"{'LoRA type (recipe)':24} {'available':>10} {'selected':>9} "
           f"{'cap':>6} {'repeats':>8}  framing of selection")
    yield "-" * 92
    for name in targets:
        rcp = {**RECIPE_DEFAULTS, **recipes[name]}
        p = recipe_plan(conn, name, rcp)
        total_selected += p["selected"]
        fr = ", ".join(f"{k}:{v}" for k, v in p["selected_framing"].items()) or "-"
        yield (f"{name:24} {p['available']:>10} {p['selected']:>9} "
               f"{(p['cap'] or '-'):>6} {p['repeats']:>8}  {fr}")
    yield "-" * 92
    yield (f"{'TOTAL selected (overlaps allowed)':24} {'':>10} {total_selected:>9}")
    yield ("\nNote: a frame can be selected by several recipes (e.g. a sharp "
           "close-up feeds character, detail and style). 'available' = frames "
           "matching that type's filters; 'selected' = kept after its cap/quota.\n"
           "Nothing was written. Run  build --recipe <name>  to materialize one.")


def content_hash(records: list[dict], concepts: list[tuple[str, dict]]) -> str:
    """Hash of everything that defines the dataset content (not time/version)."""
    payload = {
        "records": [
            {k: rec[k] for k in ("frame_id", "concept", "split", "caption", "name")}
            for rec in records
        ],
        "concepts": [
            {"name": n, "recipe": {k: rcp[k] for k in sorted(RECIPE_DEFAULTS)}}
            for n, rcp in concepts
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def _place(src: Path, dest: Path, link_mode: str) -> None:
    if dest.exists():
        return
    if link_mode == "hardlink":
        try:
            os.link(src, dest)
            return
        except OSError:
            pass
    shutil.copy2(src, dest)


def build_generator(prj: Project, cfg: BuildConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)

    try:
        concepts = resolve_concepts(prj, cfg.recipe)
    except KeyError as exc:
        yield f"FATAL: {exc}"
        return

    # assemble records
    records: list[dict] = []
    for cname, rcp in concepts:
        token = str(rcp["token"] or prj.trigger_token)
        class_word = str(rcp["class_word"] or prj.class_word)
        static = str(rcp["caption_static"] or f"{token} {class_word}".strip())
        chosen = select_for_recipe(conn, rcp)
        for frame_id, sid, path, _sh, _cl, _fr, caption, _id, bbox in chosen:
            records.append({
                "frame_id": frame_id, "concept": cname,
                "path": path, "name": f"{sid}_{Path(path).name}",
                "caption": caption or static,
                "split": stable_split(frame_id, float(rcp["val_fraction"] or 0)),
                "folder": f"{int(rcp['repeats'])}_{safe_slug(token)} {safe_slug(class_word)}".strip(),
                "bbox": bbox,
                "smart_crop": bool(rcp["smart_crop"]),
                "bucket_base": int(rcp["bucket_base"] or 1024),
            })
        yield f"Concept '{cname}': {len(chosen)} frames selected"

    if not records:
        yield "FATAL: no frames matched the recipe filters. Run curate first or relax filters."
        return

    chash = content_hash(records, concepts)

    # already built identically?
    prior = conn.execute(
        "SELECT version, dir FROM datasets WHERE content_hash = ?", (chash,)
    ).fetchone()
    if prior:
        yield (f"\nIdentical dataset already exists: v{prior[0]:03d} ({prior[1]})\n"
               f"Content hash {chash} - nothing new to build.")
        return

    # Phase 2 versioning: explicit lineage + curation snapshot (additive)
    from . import dataset_versioning as dsv
    parent_version = cfg.parent
    parent_warning = dsv.validate_parent(conn, parent_version)
    if parent_warning:
        yield f"WARNING: {parent_warning}"
        parent_version = None
    curation_snapshot = dsv.capture_curation_snapshot(conn)

    version = dsv.next_version(conn)
    ds_dir = cfg.output_base / cfg.dataset_root / f"v{version:03d}__{safe_slug(cfg.recipe)}"

    n_train = n_val = errors = 0
    for rec in records:
        src = Path(rec["path"])
        if not src.exists():
            errors += 1
            continue
        sub = "img" if rec["split"] == "train" else "val_img"
        dest_dir = ds_dir / sub / rec["folder"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / rec["name"]
        try:
            if rec.get("smart_crop"):
                from .crop import smart_crop_image
                if not dest.exists():
                    smart_crop_image(src, dest, rec.get("bbox"),
                                     base=rec.get("bucket_base", 1024))
            else:
                _place(src, dest, cfg.link_mode)
            dest.with_suffix(".txt").write_text(rec["caption"], encoding="utf-8")
            if rec["split"] == "train":
                n_train += 1
            else:
                n_val += 1
        except Exception as exc:
            errors += 1
            logging.error("build place failed %s: %s", src, exc)

    # snapshot
    snapshot = {
        "version": version,
        "recipe": cfg.recipe,
        "concepts": {n: r for n, r in concepts},
        "content_hash": chash,
        "built_at": now_iso(),
        "note": cfg.note,
        "counts": {"train": n_train, "val": n_val, "errors": errors},
        "frames": [
            {k: rec[k] for k in ("frame_id", "concept", "split", "name", "caption")}
            for rec in records
        ],
    }
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "dataset.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # single registration path (Phase 2): row insert + latest-version meta
    record = dsv.register_version(
        conn, kind=dsv.KIND_BUILD, name=cfg.recipe, directory=ds_dir,
        note=cfg.note, parent_version=parent_version,
        params={n: r for n, r in concepts},
        image_count=n_train, val_count=n_val,
        curation=curation_snapshot, version=version, content_hash=chash)
    conn.executemany(
        "INSERT OR REPLACE INTO dataset_frames(version, frame_id, concept, split, caption) "
        "VALUES (?,?,?,?,?)",
        [(version, r["frame_id"], r["concept"], r["split"], r["caption"])
         for r in records],
    )
    conn.commit()

    # version.json sidecar: provenance that travels with the dataset folder
    dsv.write_version_sidecar(ds_dir, record)

    # Phase 8: prefilled experiment template - travels with the dataset so
    # an external (e.g. Colab) training can be registered afterwards with
    # `pixel-curator exp register --from-file ... --model ...`.
    from .experiments import write_experiment_template
    write_experiment_template(
        ds_dir, name=f"{cfg.recipe}_v{version:03d}",
        dataset_version=version, dataset_dir=str(ds_dir),
        params={"recipe": cfg.recipe, "image_count": n_train})

    # Phase 10: ready-to-train kohya config + launch script beside the
    # dataset. Suggestion layer only - failure never breaks the build.
    tc_lines = ""
    if cfg.training_config:
        from .train.export_config import generate_for_export
        rtype = str(concepts[0][1].get("type") or "character") \
            if concepts else "character"
        _, tc_msgs = generate_for_export(
            conn, ds_dir, dataset_version=version, frame_count=n_train,
            model_type="lora", preset=rtype if rtype in ("character",
            "style", "outfit", "pose", "detail") else None,
            output_name=f"{cfg.recipe}_v{version:03d}")
        tc_lines = "".join(f"{m}\n" for m in tc_msgs)

    lineage_line = (f"Parent: v{parent_version:03d}\n"
                    if parent_version is not None else "")
    yield (
        f"\nBUILD v{version:03d} DONE.\n"
        f"Recipe: {cfg.recipe} ({len(concepts)} concept(s))\n"
        f"{lineage_line}"
        f"Train: {n_train} | Val: {n_val} | Errors: {errors}\n"
        f"Content hash: {chash}\n"
        f"Dir: {ds_dir}\n"
        f"Snapshot: {ds_dir / 'dataset.json'}\n"
        f"\nkohya 'Image folder': {ds_dir / 'img'}"
        + (f"\n{tc_lines.rstrip()}" if tc_lines else "")
    )

    # Phase 4: dataset-aware training suggestion at the export point
    # (advisory only; any failure degrades to a pointer at suggest-preset)
    try:
        from .train.param_advisor import recommend_training_params, render_params
        params = recommend_training_params(conn, version,
                                           recipe_name=cfg.recipe)
        yield render_params(params)
    except Exception:
        yield (f"\nNext: pixel-curator suggest-preset --dataset {version}  "
               "(training preset + parameter recommendations)")


# -----------------------------
# Listing & diff
# -----------------------------

def list_datasets(conn) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT version, recipe_name, content_hash, built_at, image_count, "
            "val_count, dir, note, kind, parent_version "
            "FROM datasets ORDER BY version").fetchall()
    except Exception:   # pre-v18 manifest opened read-only during training
        rows = [tuple(r) + (None, None) for r in conn.execute(
            "SELECT version, recipe_name, content_hash, built_at, image_count, "
            "val_count, dir, note FROM datasets ORDER BY version")]
    return [
        {"version": r[0], "recipe": r[1], "hash": r[2], "built_at": r[3],
         "train": r[4], "val": r[5], "dir": r[6], "note": r[7] or "",
         "kind": r[8] or "build", "parent": r[9]}
        for r in rows
    ]


def diff_datasets(conn, va: int, vb: int) -> dict:
    def load(v: int) -> dict[tuple, dict]:
        rows = conn.execute(
            "SELECT frame_id, concept, split, caption FROM dataset_frames "
            "WHERE version = ?", (v,),
        ).fetchall()
        if not rows and not conn.execute(
            "SELECT 1 FROM datasets WHERE version=?", (v,)
        ).fetchone():
            raise KeyError(f"dataset v{v} not found")
        return {(r[0], r[1]): {"split": r[2], "caption": r[3]} for r in rows}

    A, B = load(va), load(vb)
    added = sorted(k for k in B if k not in A)
    removed = sorted(k for k in A if k not in B)
    common = [k for k in A if k in B]
    caption_changed = sorted(k for k in common if A[k]["caption"] != B[k]["caption"])
    split_changed = sorted(k for k in common if A[k]["split"] != B[k]["split"])

    def fmix(d: dict) -> dict:
        mix: dict[str, int] = defaultdict(int)
        for (fid, concept) in d:
            mix[concept] += 1
        return dict(mix)

    return {
        "a": va, "b": vb,
        "added": [list(k) for k in added],
        "removed": [list(k) for k in removed],
        "caption_changed": [list(k) for k in caption_changed],
        "split_changed": [list(k) for k in split_changed],
        "concept_mix_a": fmix(A), "concept_mix_b": fmix(B),
        "summary": (f"v{va} -> v{vb}: +{len(added)} added, -{len(removed)} removed, "
                    f"{len(caption_changed)} captions changed, "
                    f"{len(split_changed)} split changes"),
    }
