"""Dataset insights - health report, caption lint, cluster naming,
orientation check.  Pure stdlib SQL over the manifest; pre-build QA."""
from __future__ import annotations

import json
from collections import Counter


def dataset_health(conn) -> dict:
    """One score (0-100) with the reasons it isn't 100."""
    score, reasons = 100, []
    sel = conn.execute("SELECT COUNT(*) FROM frames "
                       "WHERE status IN ('selected','packaged')").fetchone()[0]
    if sel == 0:
        return {"score": 0, "selected": 0,
                "reasons": ["no curated frames - run curate first"]}
    # framing balance
    fm = dict(conn.execute(
        "SELECT d.framing, COUNT(*) FROM frames f JOIN detections d "
        "ON d.frame_id=f.frame_id WHERE f.status IN ('selected','packaged') "
        "AND d.framing IS NOT NULL GROUP BY d.framing"))
    fb = fm.get("full_body", 0) / max(1, sum(fm.values()))
    if fb < 0.08:
        score -= 15; reasons.append(
            f"full-body coverage thin ({fb:.0%}) - pose flexibility suffers")
    # identity quality
    row = conn.execute(
        "SELECT AVG(identity_sim), COUNT(*) FROM detections d JOIN frames f "
        "ON f.frame_id=d.frame_id WHERE f.status IN ('selected','packaged') "
        "AND identity_sim IS NOT NULL").fetchone()
    if row[0] and row[0] < 0.45:
        score -= 15; reasons.append(
            f"mean identity similarity low ({row[0]:.2f})")
    # caption coverage
    cap = conn.execute(
        "SELECT COUNT(*) FROM frames f JOIN captions c ON "
        "c.frame_id=f.frame_id WHERE f.status IN ('selected','packaged') "
        "AND c.caption_text != ''").fetchone()[0]
    cov = cap / sel
    if cov < 0.95:
        score -= 20; reasons.append(f"caption coverage {cov:.0%} - "
                                    "finish captioning before build")
    # source dominance (temporal bias)
    dom = conn.execute(
        "SELECT source_id, COUNT(*) c FROM frames WHERE status IN "
        "('selected','packaged') GROUP BY source_id ORDER BY c DESC "
        "LIMIT 1").fetchone()
    if dom and dom[1] / sel > 0.25:
        score -= 10; reasons.append(
            f"one source dominates ({dom[1]}/{sel} frames) - "
            "max_per_video in the recipe will cap it at build time")
    # multi-person contamination
    multi = conn.execute(
        "SELECT COUNT(*) FROM frames f JOIN captions c ON "
        "c.frame_id=f.frame_id WHERE f.status IN ('selected','packaged') "
        "AND c.caption_text LIKE '%1boy%'").fetchone()[0]
    if multi / sel > 0.15:
        score -= 10; reasons.append(
            f"{multi} frames include a second person - review before build")
    return {"score": max(0, score), "selected": sel,
            "framing_mix": fm, "caption_coverage": round(cov, 2),
            "reasons": reasons or ["dataset is build-ready"]}


def caption_lint(conn, limit: int = 200) -> list[dict]:
    """Frames whose tags contradict detections."""
    out = []
    for fid, cap, faces in conn.execute(
            "SELECT f.frame_id, c.caption_text, d.face_count FROM frames f "
            "JOIN captions c ON c.frame_id=f.frame_id "
            "JOIN detections d ON d.frame_id=f.frame_id "
            "WHERE f.status IN ('selected','packaged')"):
        cap = cap or ""
        issues = []
        if "solo" in cap and (faces or 0) >= 2:
            issues.append("tagged solo but 2+ faces detected")
        if "no_humans" in cap and (faces or 0) >= 1:
            issues.append("tagged no_humans but a face was detected")
        if issues:
            out.append({"frame_id": fid, "issues": issues})
            if len(out) >= limit:
                break
    return out


def name_clusters(conn, top_n: int = 4) -> dict:
    """Auto-name clusters from their most distinctive frequent tags;
    stored in meta as cluster_names for the Review UI."""
    rows = conn.execute(
        "SELECT f.cluster_id, c.caption_text FROM frames f JOIN captions c "
        "ON c.frame_id=f.frame_id WHERE f.status IN ('selected','packaged') "
        "AND f.cluster_id IS NOT NULL").fetchall()
    global_c, per = Counter(), {}
    for cid, cap in rows:
        toks = [t.strip() for t in (cap or "").split(",")
                if t.strip() and "score_" not in t]
        per.setdefault(cid, Counter()).update(toks)
        global_c.update(toks)
    names = {}
    for cid, cnt in per.items():
        total = sum(cnt.values()) or 1
        # distinctiveness: local share vs global share
        ranked = sorted(cnt.items(), key=lambda kv: -(
            kv[1] / total - global_c[kv[0]] / max(1, sum(global_c.values()))))
        names[str(cid)] = ", ".join(t for t, _ in ranked[:top_n])
    # Cache names for the Review UI. Best-effort: `insights clusters` opens the
    # manifest read-only (safe during training), so a write may not be allowed —
    # the computed names are still returned either way.
    try:
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES "
                     "('cluster_names', ?)", (json.dumps(names),))
        conn.commit()
    except Exception:  # noqa: BLE001 - read-only conn or locked DB; caching is optional
        pass
    return names


def check_orientation(image_path):
    """EXIF-safe, confidence-gated upright check. Returns CCW degrees
    (0/90/180/270) or None when no confident face. Now delegates the decision
    to ``orient.decide_rotation`` - the single orientation brain - so it no
    longer misfires on EXIF-tagged or tilted faces the way the old cv2 path did."""
    try:
        import numpy as np
        from PIL import Image, ImageOps
        from insightface.app import FaceAnalysis
        from .identity_integration import _SWAPPER
        from .orient import decide_rotation, OrientConfig
        if _SWAPPER["app"] is None:
            _SWAPPER["app"] = FaceAnalysis(name="buffalo_l", providers=_rt_providers())
            _SWAPPER["app"].prepare(ctx_id=0, det_size=(640, 640))
        with Image.open(image_path) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")   # EXIF-safe
        faces = _SWAPPER["app"].get(np.asarray(rgb)[:, :, ::-1])
        if not faces:
            return None
        f = max(faces, key=lambda x: getattr(x, "det_score", 0.0))
        rot, _ = decide_rotation(f.kps, float(getattr(f, "det_score", 1.0)),
                                 OrientConfig())
        return rot
    except Exception:
        return None


def global_search(conn, prj, q: str, limit: int = 6) -> dict:
    """One query across frames, LoRAs, presets, and variation batches."""
    like = f"%{q}%"
    out = {"frames": [], "loras": [], "presets": [], "batches": []}
    try:
        out["frames"] = [
            {"frame_id": r[0], "caption": (r[1] or "")[:90]}
            for r in conn.execute(
                "SELECT f.frame_id, c.caption_text FROM frames f JOIN "
                "captions c ON c.frame_id=f.frame_id WHERE f.status IN "
                "('selected','packaged') AND c.caption_text LIKE ? LIMIT ?",
                (like, limit))]
        out["presets"] = [r[0] for r in conn.execute(
            "SELECT name FROM concept_control_presets WHERE name LIKE ? "
            "AND kind != 'stack_history' LIMIT ?", (like, limit))]
        out["batches"] = [r[0] for r in conn.execute(
            "SELECT batch_id FROM variation_batches WHERE batch_id LIKE ? "
            "LIMIT ?", (like, limit))]
    except Exception:
        pass
    try:
        from .lora_explorer import filter_cards, scan_loras_cached
        out["loras"] = [c.lora_id for c in filter_cards(
            scan_loras_cached(prj), search=q)][:limit]
    except Exception:
        pass
    return out


def log_stack_history(conn, summary: dict, keep: int = 100) -> None:
    """Append a resolved-stack snapshot to the timeline (kind
    stack_history in the presets store), pruned to the newest `keep`."""
    import time
    import uuid
    from .concept_control import save_preset
    save_preset(conn, f"hist_{time.strftime('%Y%m%d_%H%M%S')}_"
                      f"{uuid.uuid4().hex[:6]}",
                "stack_history", summary)
    conn.execute(
        "DELETE FROM concept_control_presets WHERE kind='stack_history' "
        "AND name NOT IN (SELECT name FROM concept_control_presets WHERE "
        "kind='stack_history' ORDER BY name DESC LIMIT ?)", (keep,))
    conn.commit()


def _hamming(a: str, b: str) -> int:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except Exception:
        return 64


def aesthetic_proxy(row: dict) -> float:
    """Lightweight aesthetic score (0-1) from technical signals: sharpness,
    exposure band, framing readability.  Registered as a study SIGNAL_HOOK
    and used as the duplicate-collapse tiebreak.  No ML required."""
    sharp = min(float(row.get("sharpness") or 0), 150.0) / 150.0
    b = row.get("brightness")
    expo = 1.0 - min(abs((float(b) if b is not None else 110) - 110) / 110,
                     1.0)
    framing = {"full_body": 1.0, "upper_body": 0.9, "portrait": 0.8,
               "closeup": 0.7}.get(row.get("framing") or "", 0.5)
    return round(0.5 * sharp + 0.3 * expo + 0.2 * framing, 3)


try:                                    # register as an optional probe
    from .study import SIGNAL_HOOKS as _SH
    _SH.setdefault("aesthetic_proxy",
                   lambda row: {"aesthetic_score": aesthetic_proxy(row)})
except Exception:
    pass


def collapse_near_duplicates(conn, max_hamming: int = 6,
                             apply: bool = False) -> dict:
    """Duplicate-pose collapse: within each source, consecutive frames
    whose dHash differs by <= max_hamming bits are near-identical poses -
    keep the highest aesthetic score, reject the rest (reversible:
    status rejected_neardup)."""
    rows = conn.execute(
        "SELECT f.frame_id, f.source_id, f.path, f.dhash, f.sharpness, "
        "f.brightness, d.framing FROM frames f LEFT JOIN detections d ON "
        "d.frame_id=f.frame_id WHERE f.status IN ('selected','packaged') "
        "AND f.dhash IS NOT NULL ORDER BY f.source_id, f.path").fetchall()
    rejected, kept_groups = [], 0
    group: list = []

    def flush():
        nonlocal kept_groups
        if len(group) > 1:
            kept_groups += 1
            best = max(group, key=lambda r: aesthetic_proxy(
                {"sharpness": r[4], "brightness": r[5], "framing": r[6]}))
            rejected.extend(r[0] for r in group if r[0] != best[0])
        group.clear()

    prev = None
    for r in rows:
        if (prev is not None and r[1] == prev[1]
                and _hamming(r[3], prev[3]) <= max_hamming):
            if not group:
                group.append(prev)
            group.append(r)
        else:
            flush()
        prev = r
    flush()
    if apply and rejected:
        conn.executemany(
            "UPDATE frames SET status='rejected_neardup' WHERE frame_id=?",
            [(fid,) for fid in rejected])
        conn.commit()
    return {"groups": kept_groups, "would_reject": len(rejected),
            "applied": bool(apply and rejected)}


def fix_rotations(conn, prj, apply: bool = False,
                  limit: int = 0):
    """Orientation pass over curated frames: detect upside-down faces via
    landmarks and rotate the extracted frame file 180 degrees in place
    (source media untouched).  Yields progress; resumable by nature
    (correct frames are no-ops)."""
    rows = conn.execute(
        "SELECT f.frame_id, f.path FROM frames f JOIN detections d ON "
        "d.frame_id=f.frame_id WHERE f.status IN ('selected','packaged') "
        "AND d.face_count >= 1").fetchall()
    if limit:
        rows = rows[:limit]
    yield f"Orientation check: {len(rows)} frames"
    flipped = 0
    for i, (fid, path) in enumerate(rows, 1):
        rot = check_orientation(path)
        if rot:
            flipped += 1
            if apply:
                try:
                    from PIL import Image
                    img = Image.open(path).rotate(rot, expand=True)
                    img.save(path)
                    yield f"  rotated {fid} by {rot} deg"
                except Exception as exc:
                    yield f"  {fid}: rotate failed ({exc})"
            else:
                yield f"  {fid}: needs {rot} deg rotation (dry run)"
        if i % 500 == 0:
            yield f"  {i}/{len(rows)} checked..."
    yield (f"Done: {flipped} upside-down frame(s) "
           f"{'rotated' if apply else 'found (use --apply to fix)'}")


def _rt_providers():
    from .runtime import onnx_providers
    return onnx_providers()


# ---------------------------------------------------------------------------
# Advanced Curation Phase 1: Quality Metrics Dashboard (additive)
# ---------------------------------------------------------------------------

# Warning thresholds (aligned with curation defaults; conservative)
QUALITY_THRESHOLDS = {
    "low_sharpness": 35.0,        # below curation min_sharpness default
    "soft_sharpness": 60.0,       # readable but soft
    "low_light": 60.0,            # brightness 0-255
    "overexposed": 200.0,
    "low_identity_confidence": 0.45,
    "small_face": 0.04,           # face area ratio
}


def _median(values: list) -> float:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    return float(vals[mid] if len(vals) % 2 else
                 (vals[mid - 1] + vals[mid]) / 2)


def _bins(values: list, edges: list[float]) -> dict:
    """Binned distribution {'<=edge': n, ..., '>last': n} for histograms."""
    out = {f"<={e:g}": 0 for e in edges}
    out[f">{edges[-1]:g}"] = 0
    for v in values:
        if v is None:
            continue
        for e in edges:
            if v <= e:
                out[f"<={e:g}"] += 1
                break
        else:
            out[f">{edges[-1]:g}"] += 1
    return out


def frame_quality(row: dict) -> dict:
    """Per-image quality payload + warning reason codes (pure)."""
    t = QUALITY_THRESHOLDS
    warnings: list[str] = []
    sharp, bright = row.get("sharpness"), row.get("brightness")
    ident, farea = row.get("identity_sim"), row.get("face_area")
    if sharp is not None:
        if sharp < t["low_sharpness"]:
            warnings.append("low_sharpness")
        elif sharp < t["soft_sharpness"]:
            warnings.append("minor_blur")
    if bright is not None:
        if bright < t["low_light"]:
            warnings.append("low_light")
        elif bright > t["overexposed"]:
            warnings.append("overexposed")
    if ident is not None and ident < t["low_identity_confidence"]:
        warnings.append("low_identity_confidence")
    if farea is not None and farea < t["small_face"]:
        warnings.append("small_face")
    blur_score = (round(1.0 - min(float(sharp), 150.0) / 150.0, 3)
                  if sharp is not None else None)
    return {
        "sharpness": sharp, "blur_score": blur_score,
        "brightness": (round(bright / 255.0, 3)
                       if bright is not None else None),
        "identity_confidence": ident,
        "face_area_ratio": farea,
        "framing": row.get("framing"),
        "aesthetic_score": aesthetic_proxy(row),
        "warnings": warnings,
    }


def _dashboard_rows(conn, dataset: int = 0) -> list[dict]:
    join_ds = ("JOIN dataset_frames df ON df.frame_id = f.frame_id "
               f"AND df.version = {int(dataset)} " if dataset else "")
    where = ("" if dataset
             else "WHERE f.status NOT IN ('new')")
    try:
        rows = conn.execute(
            f"SELECT f.frame_id, f.path, f.status, f.sharpness, "
            f"f.brightness, d.framing, d.identity_sim, d.face_area, "
            f"s.study_primary, s.study_tags, s.study_review_status, "
            f"f.dup_group_id, f.dup_role, f.dup_score, f.dup_method "
            f"FROM frames f {join_ds}"
            f"LEFT JOIN detections d ON d.frame_id = f.frame_id "
            f"LEFT JOIN study_labels s ON s.frame_id = f.frame_id "
            f"{where} ORDER BY f.path").fetchall()
    except Exception:                       # pre-v12 manifest: no dup columns
        rows = conn.execute(
            f"SELECT f.frame_id, f.path, f.status, f.sharpness, "
            f"f.brightness, d.framing, d.identity_sim, d.face_area, "
            f"s.study_primary, s.study_tags, s.study_review_status, "
            f"NULL, NULL, NULL, NULL "
            f"FROM frames f {join_ds}"
            f"LEFT JOIN detections d ON d.frame_id = f.frame_id "
            f"LEFT JOIN study_labels s ON s.frame_id = f.frame_id "
            f"{where} ORDER BY f.path").fetchall()
    keys = ("frame_id", "path", "status", "sharpness", "brightness",
            "framing", "identity_sim", "face_area", "study_primary",
            "study_tags", "study_review_status", "dup_group_id",
            "dup_role", "dup_score", "dup_method")
    return [dict(zip(keys, r)) for r in rows]


def quality_dashboard(conn, dataset: int = 0,
                      per_image: bool = False) -> dict:
    """Dataset-level quality insights: counts, medians, binned
    distributions, warning summaries and duplicate stats.  JSON-safe for
    the UI; `per_image` adds the full per-frame metric payloads."""
    rows = _dashboard_rows(conn, dataset)
    selected = [r for r in rows if r["status"] in ("selected", "packaged")]
    rejected = [r for r in rows if str(r["status"]).startswith("rejected")
                or r["status"] == "duplicate"]
    needs_review = [r for r in rows
                    if r["study_review_status"] == "needs_review"
                    or r["status"] == "scored"]

    quality = [dict(r, **{"q": frame_quality(r)}) for r in rows]
    warn_counts: Counter = Counter()
    for r in quality:
        warn_counts.update(r["q"]["warnings"])

    sharp_vals = [r["sharpness"] for r in rows if r["sharpness"] is not None]
    bright_vals = [r["brightness"] for r in rows
                   if r["brightness"] is not None]
    id_vals = [r["identity_sim"] for r in rows
               if r["identity_sim"] is not None]
    aes_vals = [r["q"]["aesthetic_score"] for r in quality]

    framing: Counter = Counter(r["framing"] or "unscored" for r in rows)
    study: Counter = Counter(r["study_primary"] for r in rows
                             if r["study_primary"])

    dup_groups = {r["dup_group_id"] for r in rows if r["dup_group_id"]}
    dup_candidates = sum(1 for r in rows if r["dup_role"] == "candidate")
    dup_exact = {r["dup_group_id"] for r in rows
                 if r["dup_group_id"] and "phash" == (r["dup_method"] or "")
                 and (r["dup_score"] or 0) >= 0.999}

    out = {
        "dataset": dataset or None,
        "images_analyzed": len(rows),
        "counts": {
            "selected": len(selected),
            "needs_review": len(needs_review),
            "rejected": len(rejected),
        },
        "quality": {
            "sharpness": {
                "median": round(_median(sharp_vals), 1),
                "distribution": _bins(sharp_vals, [35, 60, 100, 150]),
                "low_warnings": warn_counts.get("low_sharpness", 0)
                + warn_counts.get("minor_blur", 0),
            },
            "brightness": {
                "median": round(_median(bright_vals), 1),
                "distribution": _bins(bright_vals, [60, 110, 160, 200]),
                "low_light_warnings": warn_counts.get("low_light", 0),
                "overexposed_warnings": warn_counts.get("overexposed", 0),
            },
            "identity_confidence": {
                "median": round(_median(id_vals), 3) if id_vals else None,
                "distribution": (_bins(id_vals, [0.45, 0.6, 0.75, 0.9])
                                 if id_vals else {}),
                "low_warnings": warn_counts.get(
                    "low_identity_confidence", 0),
            },
            "aesthetic_score": {
                "median": round(_median(aes_vals), 3) if aes_vals else None,
            },
            "framing_distribution": dict(framing.most_common()),
            "study_tag_distribution": dict(study.most_common()),
            "warning_counts": dict(warn_counts),
        },
        "duplicates": {
            "group_count": len(dup_groups),
            "exact_group_count": len(dup_exact),
            "review_candidates": dup_candidates,
        },
        "reason_codes": sorted(warn_counts),
        "thresholds": QUALITY_THRESHOLDS,
    }
    if per_image:
        out["items"] = [
            {"frame_id": r["frame_id"], "path": r["path"],
             "status": r["status"], "quality": r["q"],
             "study_primary": r["study_primary"],
             "duplicates": ({"group_id": r["dup_group_id"],
                             "score": r["dup_score"],
                             "method": r["dup_method"],
                             "role": r["dup_role"]}
                            if r["dup_group_id"] else None)}
            for r in quality]
    return out


def render_dashboard(d: dict) -> str:
    """Readable CLI rendering of quality_dashboard() output."""
    c = d["counts"]
    q = d["quality"]
    lines = [
        f"Dataset {d['dataset'] or '(all curated frames)'} Quality Insights",
        "",
        f"Images analyzed: {d['images_analyzed']:,}",
        f"Selected: {c['selected']:,}",
        f"Needs review: {c['needs_review']:,}",
        f"Rejected: {c['rejected']:,}",
        "",
        "Quality:",
        f"- Sharpness: median {q['sharpness']['median']}, "
        f"{q['sharpness']['low_warnings']} low-sharpness warnings",
        f"- Brightness: median {q['brightness']['median']}, "
        f"{q['brightness']['low_light_warnings']} low-light / "
        f"{q['brightness']['overexposed_warnings']} overexposed warnings",
    ]
    idc = q["identity_confidence"]
    if idc["median"] is not None:
        lines.append(f"- Identity confidence: median {idc['median']}, "
                     f"{idc['low_warnings']} low-confidence warnings")
    fr = q["framing_distribution"]
    total = max(1, sum(fr.values()))
    lines.append("- Framing: " + ", ".join(
        f"{n / total:.0%} {k}" for k, n in list(fr.items())[:5]))
    if q["study_tag_distribution"]:
        lines.append("- Study tags: " + ", ".join(
            f"{k} x{n}" for k, n in
            list(q["study_tag_distribution"].items())[:5]))
    dup = d["duplicates"]
    lines += [
        "",
        "Duplicates:",
        f"- Exact/near duplicate groups: {dup['group_count']} "
        f"(exact: {dup['exact_group_count']})",
        f"- Suggested review candidates: {dup['review_candidates']}",
    ]
    if not dup["group_count"]:
        lines.append("- Run `pixel-curator duplicates` to scan for "
                     "duplicate groups")
    return "\n".join(lines)

