"""Viz payload builder: scoped frames -> embeddings -> projection ->
clusters -> JSON for the CLI / UI scatter viewer.

Embeddings live in the manifest (`clip_embeds`) - the same cache the
diversity-clustering step fills, so nothing is computed twice.  Frames
missing an embedding are embedded on demand when the optional CLIP
stack is installed; otherwise the payload degrades to the frames that
have embeddings, with a clear professional note.

Projection payloads are cached per dataset version + method under
  {output_base}/viz/v{NNN}/projection_{method}.json
and reused unless --recompute is passed.
"""

from __future__ import annotations

import json
from array import array
from pathlib import Path

from .. import manifest  # noqa: F401  (kept for callers' convenience)
from ..insights import frame_quality
from .clustering import cluster_embeddings, cluster_summaries
from .projection import project_2d


def viz_cache_dir(output_base: Path, dataset: int) -> Path:
    return Path(output_base) / "viz" / (f"v{dataset:03d}" if dataset
                                        else "all")


def _scoped_frames(conn, dataset: int) -> list[dict]:
    if dataset:
        rows = conn.execute(
            "SELECT f.frame_id, f.path, f.status, f.sharpness, "
            "f.brightness, d.framing, d.identity_sim, d.face_area, "
            "s.study_primary, s.study_tags, f.dup_group_id "
            "FROM frames f JOIN dataset_frames df "
            "ON df.frame_id = f.frame_id AND df.version = ? "
            "LEFT JOIN detections d ON d.frame_id = f.frame_id "
            "LEFT JOIN study_labels s ON s.frame_id = f.frame_id "
            "ORDER BY f.path", (dataset,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT f.frame_id, f.path, f.status, f.sharpness, "
            "f.brightness, d.framing, d.identity_sim, d.face_area, "
            "s.study_primary, s.study_tags, f.dup_group_id "
            "FROM frames f "
            "LEFT JOIN detections d ON d.frame_id = f.frame_id "
            "LEFT JOIN study_labels s ON s.frame_id = f.frame_id "
            "WHERE f.status IN ('selected', 'packaged') "
            "ORDER BY f.path").fetchall()
    keys = ("frame_id", "path", "status", "sharpness", "brightness",
            "framing", "identity_sim", "face_area", "study_primary",
            "study_tags", "dup_group_id")
    return [dict(zip(keys, r)) for r in rows]


def _decode_embed(blob: bytes) -> list[float]:
    a = array("f")
    a.frombytes(blob)
    return list(a)


def _load_embeddings(conn, frame_ids: list[str]
                     ) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    marks = ",".join("?" for _ in frame_ids)
    if not frame_ids:
        return out
    for fid, blob in conn.execute(
            f"SELECT frame_id, embed FROM clip_embeds "
            f"WHERE frame_id IN ({marks})", frame_ids):
        if blob:
            out[fid] = _decode_embed(blob)
    return out


def _embed_missing(conn, frames: list[dict], missing: list[dict],
                   warnings: list[str]) -> dict[str, list[float]]:
    """Embed frames lacking vectors when the optional CLIP stack exists."""
    from ..curate.diversity import clip_available
    ok, backend = clip_available()
    if not ok:
        warnings.append(
            f"{len(missing)} frame(s) have no stored embedding and the "
            f"optional CLIP stack is unavailable ({backend}) - they are "
            "excluded from the map. Install 'pixel-curator[cluster]' or run "
            "the diversity clustering step.")
        return {}
    try:
        from ..curate.diversity import _load_clip
        from ..util import now_iso
        embed, name = _load_clip()
        out: dict[str, list[float]] = {}
        batch = 32
        for start in range(0, len(missing), batch):
            chunk = missing[start:start + batch]
            ok_idx, feats = embed([Path(m["path"]) for m in chunk])
            for local, vec in zip(ok_idx, feats):
                fid = chunk[local]["frame_id"]
                out[fid] = [float(x) for x in vec]
                conn.execute(
                    "INSERT OR REPLACE INTO clip_embeds"
                    "(frame_id, embed, embedded_at) VALUES (?,?,?)",
                    (fid, array("f", out[fid]).tobytes(), now_iso()))
        conn.commit()
        warnings.append(f"embedded {len(out)} new frame(s) via {name}")
        return out
    except Exception as exc:
        warnings.append(f"embedding generation unavailable ({exc}) - "
                        "using cached embeddings only")
        return {}


def _quality_score(f: dict) -> float:
    from ..insights import aesthetic_proxy
    return aesthetic_proxy(f)


def build_viz_payload(conn, output_base: Path, dataset: int = 0,
                      method: str = "auto", recompute: bool = False,
                      k: int = 0) -> dict:
    """Full visualization payload (items + clusters + meta), cached."""
    cache_dir = viz_cache_dir(output_base, dataset)
    cache_file = cache_dir / f"projection_{method or 'auto'}.json"
    if cache_file.exists() and not recompute:
        try:
            cached = json.loads(cache_file.read_text())
            cached["meta"]["cached"] = True
            return cached
        except Exception:
            pass                                    # fall through to rebuild

    warnings: list[str] = []
    frames = _scoped_frames(conn, dataset)
    embeds = _load_embeddings(conn, [f["frame_id"] for f in frames])
    missing = [f for f in frames if f["frame_id"] not in embeds]
    if missing:
        embeds.update(_embed_missing(conn, frames, missing, warnings))

    usable = [f for f in frames if f["frame_id"] in embeds]
    vectors = [embeds[f["frame_id"]] for f in usable]
    points, method_used, proj_warnings = project_2d(vectors, method)
    cluster_ids, cluster_method, cl_warnings = cluster_embeddings(
        vectors, k)
    warnings += proj_warnings + cl_warnings

    items = []
    for f, (x, y), cid in zip(usable, points, cluster_ids):
        q = frame_quality(f)
        tags = []
        if f.get("study_primary"):
            tags.append(f["study_primary"])
        raw = f.get("study_tags")
        if raw:
            try:
                extra = (json.loads(raw) if raw.strip().startswith("[")
                         else raw.split(","))
                tags += [t.strip() for t in extra if t and t.strip()]
            except Exception:
                pass
        items.append({
            "item_id": f["frame_id"], "x": x, "y": y,
            "cluster_id": cid,
            "quality_score": q["aesthetic_score"],
            "identity_confidence": f.get("identity_sim"),
            "framing": f.get("framing"),
            "study_tags": sorted(set(tags)),
            "duplicate_group_id": f.get("dup_group_id"),
            "status": f.get("status"),
            "quality_warnings": q["warnings"],
            "path": f.get("path"),
        })

    payload = {
        "dataset": dataset or None,
        "items": items,
        "clusters": cluster_summaries(items),
        "meta": {
            "frames_in_scope": len(frames),
            "frames_with_embeddings": len(usable),
            "projection_method": method_used,
            "cluster_method": cluster_method,
            "warnings": warnings,
            "cached": False,
        },
    }
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload, indent=1))
        payload["meta"]["cache_path"] = str(cache_file)
    except Exception:
        pass
    return payload


def render_viz_summary(payload: dict) -> str:
    m = payload["meta"]
    lines = [
        f"EMBEDDING MAP | dataset "
        f"{payload['dataset'] or '(all selected frames)'}",
        f"  frames in scope: {m['frames_in_scope']} | with embeddings: "
        f"{m['frames_with_embeddings']}",
        f"  projection: {m['projection_method']} | clustering: "
        f"{m['cluster_method']}",
    ]
    for w in m["warnings"]:
        lines.append(f"  NOTE: {w}")
    for c in payload["clusters"][:12]:
        lines.append(f"  {c['cluster_id']:<13} {c['cluster_summary']}")
    if len(payload["clusters"]) > 12:
        lines.append(f"  ... {len(payload['clusters']) - 12} more clusters")
    if m.get("cache_path"):
        lines.append(f"  cached -> {m['cache_path']}")
    lines.append("  Open the UI Viz tab for the interactive map.")
    return "\n".join(lines)
