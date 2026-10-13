"""Diversity curator — select frames for a FLEXIBLE base model.

The lesson: a DreamBooth bakes in whatever is *consistent and uncaptioned*. If
most of your good frames come from one session (one outfit, one room, one pose),
the model glues that outfit/room to your identity. Quality-only selection makes
this worse - it happily takes 1,000 near-variations of the same party.

This curator selects for *spread*, not just per-frame quality:

  * cap frames per SOURCE  - no single shoot/video dominates
  * cap frames per CLUSTER - no single visual scene dominates (CLIP cluster ~ scene)
  * balance FRAMING        - soft quota across closeup/portrait/upper/full body
  * gate on identity x sharpness, prefer the best within each bucket

It trades a little per-frame quality for dataset variety - which is exactly what
lets the trained base put you in *anything*. If the data simply isn't diverse
enough to fill the request under the caps, it returns fewer and says so, rather
than silently padding with the dominant session.

Pure-Python and deterministic so it's testable; reads only the manifest.
"""
from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict

logger = logging.getLogger(__name__)

DEFAULT_FRAMING_QUOTA = {"closeup": 0.20, "portrait": 0.30,
                         "upper_body": 0.30, "full_body": 0.20}


def fetch_candidates(conn: sqlite3.Connection) -> list[dict]:
    """All curated frames with the attributes the curator needs. When the
    Priority-2 quality pass has run, its integrity score is joined in so
    selection is quality-aware (face-region sharpness, detail retention,
    occlusion, composition) rather than whole-image-sharpness only."""
    has_fq = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='frame_quality'"
    ).fetchone() is not None
    fq_cols = (", q.integrity, q.face_sharpness, q.detail_retention, q.occlusion, "
               "q.frontalness, q.pose_bucket" if has_fq else "")
    fq_join = ("LEFT JOIN frame_quality q ON q.frame_id = f.frame_id"
               if has_fq else "")
    cur = conn.execute(
        f"SELECT f.frame_id, f.path, f.source_id, f.sharpness, f.cluster_id, "
        f"d.identity_sim, d.framing{fq_cols} "
        f"FROM frames f LEFT JOIN detections d ON d.frame_id = f.frame_id "
        f"{fq_join} WHERE f.status IN ('selected','packaged')")
    out = []
    for r in cur.fetchall():
        d = {"frame_id": r[0], "path": r[1], "source_id": r[2] or "?",
             "sharpness": r[3], "cluster_id": r[4],
             "identity_sim": r[5], "framing": r[6],
             "integrity": r[7] if has_fq else None,
             "face_sharpness": r[8] if has_fq else None,
             "detail_retention": r[9] if has_fq else None,
             "occlusion": r[10] if has_fq else None,
             "frontalness": r[11] if has_fq else None,
             "pose_bucket": r[12] if has_fq else None}
        out.append(d)
    return out


def pose_diversity_diagnostic(frames: list[dict]) -> dict | None:
    """Optional, fail-safe pose-diversity diagnostic over a frame set, using the
    pose_diversity module. Returns None when pose data is absent or the module
    is unavailable - never raises, never changes selection."""
    if not any(f.get("pose_bucket") or f.get("frontalness") is not None
               for f in frames):
        return None
    try:
        from .curate.pose_diversity import compute_pose_diversity
        scores = compute_pose_diversity(frames)
        return scores.to_dict()
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None


def _pose_bonus_for(frame: dict, seen_buckets: dict) -> float:
    """Deterministic rarity bonus in [0, 1]: a frame whose pose/body-angle
    bucket is under-represented so far scores higher. Returns 0 when no pose
    signal is present, so disabled/missing-signal behavior is unchanged."""
    bucket = frame.get("pose_bucket")
    if not bucket and frame.get("frontalness") is not None:
        fr = frame["frontalness"]
        bucket = ("frontal" if fr >= 0.75 else "three_quarter" if fr >= 0.5
                  else "profile" if fr >= 0.25 else "rear")
    if not bucket:
        return 0.0
    # rarer bucket (fewer already selected) -> closer to 1.0
    return 1.0 / (1.0 + float(seen_buckets.get(bucket, 0)))


def _score(frames: list[dict], *, pose_bonus: float = 0.0,
           pose_bonus_cap: float = 0.05,
           pose_duplicate_penalty: float = 0.0,
           pose_duplicate_penalty_cap: float = 0.08) -> None:
    """Composite quality score. Base = identity x whole-image sharpness. When
    Priority-2 quality signals are present, blend in `integrity` (face-region
    sharpness + detail retention + occlusion + composition) so the curator
    prefers genuinely good frames, not merely sharp backgrounds.

    When ``pose_bonus > 0`` a small, capped pose-rarity term is added so ties
    favor under-represented poses. The bonus is capped (default 0.05) so it can
    never outweigh identity/quality relevance; with ``pose_bonus == 0`` (the
    default) scoring is byte-for-byte unchanged."""
    sharps = [f["sharpness"] for f in frames if f["sharpness"] is not None]
    lo, hi = (min(sharps), max(sharps)) if sharps else (0.0, 1.0)
    span = (hi - lo) or 1.0
    have_quality = any(f.get("integrity") is not None for f in frames)
    # deterministic bucket prior: most-common buckets first, stable by name
    seen: dict = {}
    if pose_bonus > 0:
        from collections import Counter
        bc = Counter(f.get("pose_bucket") for f in frames if f.get("pose_bucket"))
        seen = dict(bc)
    cap = max(0.0, float(pose_bonus_cap))
    for f in frames:
        ident = f["identity_sim"] if f["identity_sim"] is not None else 0.5
        sn = ((f["sharpness"] - lo) / span) if f["sharpness"] is not None else 0.5
        base = 0.5 * float(ident) + 0.5 * float(sn)
        if have_quality:
            integ = f["integrity"] if f.get("integrity") is not None else 0.5
            # identity 0.40, sharpness 0.25, integrity 0.35
            score = (0.40 * float(ident) + 0.25 * float(sn)
                     + 0.35 * float(integ))
        else:
            score = base
        if pose_bonus > 0:
            score += min(cap, pose_bonus * _pose_bonus_for(f, seen))
        # optional, default-off down-rank for non-representative pose
        # near-duplicates; representatives are never penalized.
        if pose_duplicate_penalty > 0:
            pd = f.get("pose_duplicate")
            if isinstance(pd, dict) and pd.get("near_duplicate") \
                    and not pd.get("representative", True):
                score -= min(max(0.0, float(pose_duplicate_penalty_cap)),
                             float(pose_duplicate_penalty))
        f["_score"] = round(score, 5)


def diverse_select(frames: list[dict], count: int, *,
                   max_source_frac: float = 0.15,
                   max_cluster_frac: float = 0.20,
                   framing_quota: dict | None = None,
                   min_identity: float = 0.0,
                   min_sharpness: float = 0.0,
                   pose_bonus: float = 0.0,
                   pose_bonus_cap: float = 0.05,
                   pose_duplicate_penalty: float = 0.0,
                   pose_duplicate_penalty_cap: float = 0.08) -> dict:
    """Select up to `count` frames maximizing source/scene/framing spread.

    ``pose_bonus`` (default 0 = disabled) adds a small, capped pose-rarity term
    to the ranking; existing selection is unchanged when it is 0 or when no pose
    signals are present."""
    cands = [f for f in frames
             if (f.get("identity_sim") or 0.0) >= min_identity
             and (f.get("sharpness") or 0.0) >= min_sharpness]
    if not cands:
        return {"selected": [], "n": 0, "requested": count, "n_candidates": 0,
                "warnings": ["no candidates passed the identity/sharpness gate"],
                "sources_used": 0, "clusters_used": 0, "framing": {},
                "source_cap": 0, "cluster_cap": 0, "shortfall": count,
                "top_sources": []}
    _score(cands, pose_bonus=pose_bonus, pose_bonus_cap=pose_bonus_cap,
           pose_duplicate_penalty=pose_duplicate_penalty,
           pose_duplicate_penalty_cap=pose_duplicate_penalty_cap)
    n_target = min(count, len(cands))
    src_cap = max(1, int(n_target * max_source_frac))
    clu_cap = max(1, int(n_target * max_cluster_frac))
    quota = framing_quota or DEFAULT_FRAMING_QUOTA
    frm_target = {k: int(round(v * n_target)) for k, v in quota.items()}

    # group by cluster, best-first within each
    by_cluster: dict = defaultdict(list)
    for f in cands:
        by_cluster[f["cluster_id"]].append(f)
    for cid in by_cluster:
        by_cluster[cid].sort(key=lambda f: f["_score"], reverse=True)
    cluster_order = sorted(by_cluster.keys(),
                           key=lambda c: len(by_cluster[c]), reverse=True)

    selected: list[dict] = []
    src_count: dict = defaultdict(int)
    clu_count: dict = defaultdict(int)
    frm_count: dict = defaultdict(int)

    def _take_from(cid, respect_framing) -> bool:
        q = by_cluster[cid]
        if clu_count[cid] >= clu_cap:
            return False
        for idx, f in enumerate(q):
            if src_count[f["source_id"]] >= src_cap:
                continue
            fr = f.get("framing")
            if (respect_framing and fr in frm_target
                    and frm_count[fr] >= frm_target[fr]):
                continue
            q.pop(idx)
            selected.append(f)
            src_count[f["source_id"]] += 1
            clu_count[cid] += 1
            if fr:
                frm_count[fr] += 1
            return True
        return False

    # round-robin: first honoring framing quota, then a relaxed pass
    for respect in (True, False):
        i = 0
        stalls = 0
        while len(selected) < n_target and cluster_order and stalls < len(cluster_order):
            cid = cluster_order[i % len(cluster_order)]
            i += 1
            if _take_from(cid, respect):
                stalls = 0
            else:
                stalls += 1
        if len(selected) >= n_target:
            break

    top_sources = sorted(src_count.items(), key=lambda kv: -kv[1])[:6]
    warnings = []
    shortfall = count - len(selected)
    if shortfall > 0:
        warnings.append(
            f"only {len(selected)} diverse frames available under caps "
            f"(source<={src_cap}, cluster<={clu_cap}). Your data isn't varied "
            "enough to fill {count} without over-using one session - either "
            "lower --count, raise --max-source-frac, or add more varied shoots."
            .replace("{count}", str(count)))
    return {"selected": selected, "n": len(selected), "requested": count,
            "n_candidates": len(cands),
            "sources_used": len([s for s, c in src_count.items() if c]),
            "clusters_used": len([c for c, n in clu_count.items() if n]),
            "framing": dict(frm_count), "source_cap": src_cap,
            "cluster_cap": clu_cap, "shortfall": max(0, shortfall),
            "pose_diversity": pose_diversity_diagnostic(selected),
            "top_sources": top_sources, "warnings": warnings}


def render_plan(plan: dict) -> str:
    lines = ["", "Diversity selection plan", "=" * 56,
             f"selected     : {plan['n']} / {plan['requested']} "
             f"(from {plan['n_candidates']} candidates)",
             f"spread       : {plan['sources_used']} sources, "
             f"{plan['clusters_used']} clusters",
             f"caps         : <= {plan['source_cap']}/source, "
             f"<= {plan['cluster_cap']}/cluster",
             f"framing mix  : {plan['framing']}"]
    if plan["top_sources"]:
        top = ", ".join(f"{s}:{c}" for s, c in plan["top_sources"][:5])
        lines.append(f"top sources  : {top}")
    pdg = plan.get("pose_diversity")
    if pdg:
        lines.append(f"pose diversity: {pdg.get('diversity_score')} "
                     f"(keypoint var {pdg.get('keypoint_variance')}) "
                     f"{pdg.get('pose_distribution', {})}")
    for w in plan.get("warnings", []):
        lines.append(f"  ! {w}")
    if not plan.get("warnings"):
        lines.append("Good spread - no single session or scene dominates.")
    return "\n".join(lines)
