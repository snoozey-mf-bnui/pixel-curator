"""Preset Recommendation System - dataset-aware training preset advisor.

Inspects a built dataset version (manifest tables: datasets, dataset_frames,
frames, detections, captions, study_labels) and recommends the most suitable
training preset from train/presets.py with confidence, professional
reasoning, alternatives, warnings, raw signals and reason codes.

Design rules:
    - explainable: every recommendation carries reason codes + prose
    - conservative and identity-preserving: weak identity consistency or a
      small dataset always gates specialized presets down to a stable one
    - tolerant: missing Study Intelligence rows, captions or detections
      degrade gracefully (never raises on absent data)
    - dynamic: only presets that actually exist in PRESETS are recommended
"""
from __future__ import annotations

import json
from typing import Optional

from .presets import PRESETS

# study_primary values written by the Study Intelligence Layer
_STUDY_PRIMARIES = ("figure_study_candidate", "fashion_study_candidate",
                    "lingerie_fashion_candidate", "form_proportion_candidate")

# preference chains: conceptual target -> ordered real-preset fallbacks
_FALLBACK_CHAINS: dict[str, list[str]] = {
    "fine_art_figure_study": ["fine_art_figure_study", "figure_study",
                              "intimate_figure", "balanced_study",
                              "character"],
    "figure_study": ["figure_study", "fine_art_figure_study",
                     "intimate_figure", "balanced_study", "character"],
    "intimate_anatomy_study": ["intimate_anatomy_study", "intimate_figure",
                               "figure_study", "balanced_study", "character"],
    "lingerie_form_study": ["lingerie_form_study", "fashion_editorial",
                            "intimate_figure", "balanced_study", "outfit"],
    "fashion_editorial": ["fashion_editorial", "lingerie_form_study",
                          "outfit", "balanced_study"],
    "explicit_body_detail": ["explicit_body_detail",
                             "intimate_anatomy_study", "detail",
                             "intimate_figure", "balanced_study"],
    "intimate_figure": ["intimate_figure", "balanced_study", "figure_study",
                        "character"],
    "balanced_study": ["balanced_study", "intimate_figure", "character"],
    "character": ["character", "balanced_study"],
}

# dataset-size tiers (selected training frames)
_SIZE_SMALL = 150
_SIZE_SUFFICIENT = 300
_SIZE_STRONG = 700


def available_presets(presets: Optional[dict] = None) -> list[str]:
    return sorted(presets if presets is not None else PRESETS)


def _closest_available(target: str, presets: dict) -> Optional[str]:
    for cand in _FALLBACK_CHAINS.get(target, [target]):
        if cand in presets:
            return cand
    return sorted(presets)[0] if presets else None


# ---------------------------------------------------------------------------
# Signal extraction (manifest -> flat dict; tolerant of missing data)
# ---------------------------------------------------------------------------

def _q(conn, sql: str, params=()) -> list:
    try:
        return conn.execute(sql, params).fetchall()
    except Exception:
        return []


def collect_signals(conn, dataset_version: int) -> dict:
    """Dataset-version-scoped signals.  Raises ValueError only when the
    dataset version itself does not exist; everything else degrades."""
    row = _q(conn, "SELECT recipe_name, recipe_json, image_count, val_count, "
                   "dir, built_at FROM datasets WHERE version = ?",
             (dataset_version,))
    if not row:
        raise ValueError(f"dataset v{dataset_version} not found - "
                         "run a build first (Builds tab or `build --recipe`)")
    recipe_name, recipe_json, image_count, val_count, ds_dir, built_at = row[0]

    fids = sorted({r[0] for r in _q(
        conn, "SELECT frame_id FROM dataset_frames WHERE version = ? "
              "AND split = 'train'", (dataset_version,))})
    s: dict = {
        "dataset_version": dataset_version,
        "recipe_name": recipe_name or "",
        "selected_frames": len(fids) or int(image_count or 0),
        "val_frames": int(val_count or 0),
        "built_at": built_at,
    }
    try:
        rj = json.loads(recipe_json or "{}")
    except Exception:
        rj = {}
    s["recipe_type"] = str(rj.get("type") or "")
    s["recipe_study_primary"] = str(rj.get("study_primary") or "")

    if not fids:
        # legacy build without dataset_frames rows - global fallback scope
        fids = [r[0] for r in _q(
            conn, "SELECT frame_id FROM frames "
                  "WHERE status IN ('selected','packaged')")]
        s["scope"] = "global_fallback"
    else:
        s["scope"] = "dataset"

    marks = ",".join("?" * len(fids)) if fids else "''"

    # --- frames: sharpness ------------------------------------------------
    rows = _q(conn, f"SELECT AVG(sharpness) FROM frames "
                    f"WHERE frame_id IN ({marks})", fids)
    s["sharpness_mean"] = (round(rows[0][0], 1)
                           if rows and rows[0][0] is not None else None)

    # --- detections: identity + framing ------------------------------------
    det = _q(conn, f"SELECT identity_sim, framing, face_count FROM detections "
                   f"WHERE frame_id IN ({marks})", fids)
    sims = [r[0] for r in det if r[0] is not None]
    s["identity_strength"] = (round(sum(sims) / len(sims), 3)
                              if sims else None)
    if len(sims) > 1:
        mean = sum(sims) / len(sims)
        var = sum((x - mean) ** 2 for x in sims) / len(sims)
        s["identity_std"] = round(var ** 0.5, 3)
    else:
        s["identity_std"] = None
    s["face_rate"] = (round(sum(1 for r in det if (r[2] or 0) > 0)
                            / len(det), 3) if det else None)
    mix: dict[str, int] = {}
    for r in det:
        mix[r[1] or "none"] = mix.get(r[1] or "none", 0) + 1
    total = max(1, sum(mix.values()))
    s["framing_mix"] = {k: round(v / total, 3) for k, v in sorted(mix.items())}

    # --- study intelligence -------------------------------------------------
    st = _q(conn, f"SELECT study_primary, study_confidence, "
                  f"identity_lock_score FROM study_labels "
                  f"WHERE frame_id IN ({marks})", fids)
    s["study_classified"] = len(st)
    s["study_coverage"] = (round(len(st) / len(fids), 3) if fids else 0.0)
    prim: dict[str, int] = {}
    for r in st:
        if r[0]:
            prim[r[0]] = prim.get(r[0], 0) + 1
    n_st = max(1, len(st))
    s["figure_study_rate"] = round(
        (prim.get("figure_study_candidate", 0)
         + prim.get("form_proportion_candidate", 0)) / n_st, 3)
    s["fashion_study_rate"] = round(
        (prim.get("fashion_study_candidate", 0)
         + prim.get("lingerie_fashion_candidate", 0)) / n_st, 3)
    s["lingerie_fashion_rate"] = round(
        prim.get("lingerie_fashion_candidate", 0) / n_st, 3)
    s["form_proportion_rate"] = round(
        prim.get("form_proportion_candidate", 0) / n_st, 3)
    confs = [r[1] for r in st if r[1] is not None]
    s["study_confidence_mean"] = (round(sum(confs) / len(confs), 3)
                                  if confs else None)
    locks = [r[2] for r in st if r[2] is not None]
    s["identity_lock_mean"] = (round(sum(locks) / len(locks), 3)
                               if locks else None)

    # --- captions -----------------------------------------------------------
    cap = _q(conn, f"SELECT COUNT(*) FROM captions "
                   f"WHERE frame_id IN ({marks})", fids)
    s["caption_coverage"] = (round((cap[0][0] if cap else 0)
                                   / len(fids), 3) if fids else 0.0)

    # --- diversity ------------------------------------------------------------
    cl = _q(conn, f"SELECT COUNT(DISTINCT cluster_id) FROM frames "
                  f"WHERE frame_id IN ({marks}) AND cluster_id IS NOT NULL",
            fids)
    s["cluster_count"] = int(cl[0][0]) if cl else 0

    # --- composite quality score (0-1) ---------------------------------------
    parts = []
    if s["sharpness_mean"] is not None:
        parts.append(max(0.0, min(1.0, (s["sharpness_mean"] - 30) / 120)))
    if s["identity_std"] is not None:
        parts.append(max(0.0, min(1.0, 1.0 - s["identity_std"] * 4)))
    if s["caption_coverage"]:
        parts.append(min(1.0, s["caption_coverage"]))
    s["quality_score"] = (round(sum(parts) / len(parts), 3)
                          if parts else None)
    return s


# ---------------------------------------------------------------------------
# Scoring (explainable, conservative, identity-preserving)
# ---------------------------------------------------------------------------

def _score_candidates(s: dict, presets: dict,
                      recipe_name: str = "") -> tuple[dict, list, list]:
    """Returns ({conceptual_target: (score, reason)}, reason_codes,
    warnings)."""
    codes: list[str] = []
    warns: list[str] = []
    n = s.get("selected_frames") or 0
    idstr = s.get("identity_strength")
    quality = s.get("quality_score")
    fm = s.get("framing_mix") or {}
    full_body = fm.get("full_body", 0.0)
    upper = fm.get("upper_body", 0.0)
    close = fm.get("closeup", 0.0) + fm.get("portrait", 0.0)
    figure = s.get("figure_study_rate") or 0.0
    fashion = s.get("fashion_study_rate") or 0.0
    lingerie = s.get("lingerie_fashion_rate") or 0.0
    coverage = s.get("study_coverage") or 0.0
    captions = s.get("caption_coverage") or 0.0

    # --- gates / context codes ----------------------------------------------
    small = n < _SIZE_SMALL
    sufficient = n >= _SIZE_SUFFICIENT
    if small:
        codes.append("dataset_size_small")
        warns.append(f"Only {n} selected training frames - specialized "
                     "presets are gated; a balanced preset reduces overfit "
                     "risk on small datasets.")
    elif sufficient:
        codes.append("sufficient_dataset_size")

    identity_ok = idstr is not None and idstr >= 0.55
    identity_weak = idstr is not None and idstr < 0.45
    if identity_ok:
        codes.append("strong_identity_signal")
    elif identity_weak:
        codes.append("identity_consistency_low")
        warns.append("Identity consistency is below the recommended "
                     "threshold (0.45) - review curation and anchor "
                     "references before a specialized study run.")
    elif idstr is None:
        codes.append("identity_signal_unavailable")
        warns.append("No identity similarity data for this dataset - run "
                     "smart curation for identity-aware recommendations.")

    if quality is not None and quality >= 0.65:
        codes.append("quality_score_good")
    elif quality is not None and quality < 0.45:
        codes.append("quality_score_low")
        warns.append("Dataset quality signals are weak (sharpness / "
                     "consistency / caption coverage) - consider another "
                     "curation pass before training.")

    if coverage < 0.5:
        codes.append("study_coverage_incomplete")
        if coverage == 0.0:
            warns.append("No Study Intelligence classifications found - "
                         "run `study classify` for study-aware "
                         "recommendations.")
    if captions < 0.8 and n:
        codes.append("caption_coverage_low")
        warns.append("Caption coverage is below 80% - caption the dataset "
                     "before training for stable token binding.")

    balance = 1.0 - abs(full_body - close)
    if full_body and close and balance >= 0.7:
        codes.append("framing_balance_good")
    elif fm and max(fm.values() or [0]) > 0.75:
        codes.append("framing_imbalanced")
        warns.append("Framing distribution is concentrated in one band - "
                     "consider broadening the selection for flexibility.")

    rname = (recipe_name or s.get("recipe_name") or "").lower()
    rtype = (s.get("recipe_type") or "").lower()
    rstudy = s.get("recipe_study_primary") or ""
    if rname or rtype:
        codes.append(f"recipe_context_{rtype or 'custom'}")

    # --- candidate scores ------------------------------------------------------
    cand: dict[str, tuple[float, str]] = {}

    def add(target: str, score: float, reason: str):
        score = max(0.0, min(0.97, score))
        if target not in cand or score > cand[target][0]:
            cand[target] = (round(score, 2), reason)

    # safe baseline always present
    add("balanced_study", 0.55 + (0.08 if not small else 0.12),
        "Stable preset for mixed datasets; preserves identity consistency "
        "while covering figure and fashion study material.")

    if small or identity_weak or (quality is not None and quality < 0.45):
        add("intimate_figure", 0.5,
            "Conservative figure preset with strong identity lock - "
            "appropriate while dataset size or consistency is limited.")
        # specialized presets stay gated
        codes.append("specialized_presets_gated")
    else:
        # figure / form path
        if figure >= 0.4 and (full_body + upper) >= 0.4:
            base = 0.6 + figure * 0.3 + (0.08 if identity_ok else 0.0) \
                   + (0.05 if sufficient else 0.0)
            add("fine_art_figure_study", base,
                "Strong figure and form-proportion study coverage with "
                "full/upper-body framing; fine-art figure preset rewards "
                "high-quality reference sets.")
            add("figure_study", base - 0.06,
                "Identity-locked figure study emphasis as a slightly "
                "more conservative alternative.")
            if (quality or 0) >= 0.65 and sufficient and close >= 0.15:
                add("intimate_anatomy_study", base - 0.03,
                    "High-detail anatomical study option: quality, size "
                    "and close-framing support fine form rendering.")
            codes.append("figure_study_rate_high")
        # fashion / garment path
        if fashion >= 0.3:
            base = 0.58 + fashion * 0.35 + (0.06 if captions >= 0.8 else 0.0)
            if lingerie >= fashion * 0.5:
                add("lingerie_form_study", base + 0.03,
                    "Significant lingerie-fashion study coverage with "
                    "garment structure focus; balances identity lock with "
                    "fabric and form interaction.")
            add("fashion_editorial", base - 0.04,
                "Fashion study coverage suits wardrobe styling and "
                "garment-detail emphasis.")
            codes.append("fashion_study_rate_high")
        # high-detail path (strictly gated)
        if ((quality or 0) >= 0.75 and identity_ok and n >= 400
                and (s.get("sharpness_mean") or 0) >= 80):
            add("explicit_body_detail", 0.62 + (quality or 0) * 0.15,
                "Dataset meets the strict quality, identity-consistency "
                "and size bar for the high-detail preset; monitor the "
                "best-epoch sweep closely for overfit.")
            codes.append("detail_preset_eligible")
        elif figure >= 0.5 and close >= 0.3:
            codes.append("detail_preset_gated")
            warns.append("A high-detail preset was considered but gated: "
                         "it requires very clean, large, identity-stable "
                         "datasets (quality >= 0.75, 400+ frames).")

    # recipe nudges
    nudge_map = {
        "figure_study_candidate": ("fine_art_figure_study", "figure_study"),
        "form_proportion_candidate": ("fine_art_figure_study",
                                      "intimate_anatomy_study"),
        "fashion_study_candidate": ("fashion_editorial",
                                    "lingerie_form_study"),
        "lingerie_fashion_candidate": ("lingerie_form_study",
                                       "fashion_editorial"),
    }
    for tgt in nudge_map.get(rstudy, ()):
        if tgt in cand:
            sc, why = cand[tgt]
            cand[tgt] = (round(min(0.97, sc + 0.05), 2),
                         why + " Recipe study context reinforces this "
                               "selection.")
            codes.append("recipe_study_context_match")
            break
    if rtype == "character" and "balanced_study" in cand:
        sc, why = cand["balanced_study"]
        cand["balanced_study"] = (round(min(0.97, sc + 0.04), 2), why)

    return cand, codes, warns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def suggest_preset(conn, dataset_version: int,
                   recipe_name: Optional[str] = None,
                   presets: Optional[dict] = None) -> dict:
    """Analyze a built dataset version and recommend a training preset.

    conn            manifest connection (manifest.connect(output_base))
    dataset_version built dataset version number
    recipe_name     optional CLI recipe override (augments manifest recipe)
    presets         injectable preset dict (defaults to train/presets.PRESETS)
    """
    presets = presets if presets is not None else PRESETS
    signals = collect_signals(conn, dataset_version)
    if recipe_name:
        signals["recipe_override"] = recipe_name

    cand, codes, warns = _score_candidates(signals, presets,
                                           recipe_name or "")

    # map conceptual targets onto presets that actually exist
    resolved: dict[str, tuple[float, str]] = {}
    for target, (score, reason) in cand.items():
        real = _closest_available(target, presets)
        if real is None:
            continue
        if real != target:
            reason += (f" (Preset '{target}' is not installed - "
                       f"'{real}' is the closest available.)")
            score = round(max(0.3, score - 0.05), 2)
        if real not in resolved or score > resolved[real][0]:
            resolved[real] = (score, reason)
    if not resolved:
        return {"recommended_preset": None, "confidence": 0.0,
                "reason": "No training presets available in "
                          "train/presets.py.",
                "alternative_presets": [], "warnings": warns,
                "signals": signals, "reason_codes": codes + ["no_presets"]}

    ranked = sorted(resolved.items(), key=lambda kv: -kv[1][0])
    best, (best_score, best_reason) = ranked[0]
    # confidence: top score plus separation from runner-up, clamped
    gap = best_score - (ranked[1][1][0] if len(ranked) > 1 else 0.3)
    confidence = round(max(0.3, min(0.95, best_score + gap * 0.25)), 2)

    # alternatives are always reported strictly below the recommendation
    alternatives = []
    for i, (name, (sc, why)) in enumerate(ranked[1:4]):
        alternatives.append({
            "preset": name,
            "confidence": round(min(sc, confidence - 0.03 * (i + 1)), 2),
            "reason": why})

    return {
        "recommended_preset": best,
        "confidence": confidence,
        "reason": best_reason,
        "alternative_presets": alternatives,
        "warnings": warns,
        "signals": signals,
        "reason_codes": sorted(set(codes)),
    }


def format_recommendation(rec: dict) -> str:
    """Human-readable CLI summary."""
    lines = [f"Recommended preset: {rec['recommended_preset']}",
             f"Confidence: {rec['confidence']:.2f}", "",
             "Reason:", rec["reason"], ""]
    if rec["alternative_presets"]:
        lines.append("Alternative presets:")
        for a in rec["alternative_presets"]:
            lines.append(f"- {a['preset']} ({a['confidence']:.2f}) - "
                         f"{a['reason']}")
        lines.append("")
    if rec["warnings"]:
        lines.append("Warnings:")
        lines += [f"- {w}" for w in rec["warnings"]]
        lines.append("")
    lines.append("Signals:")
    keys = ("selected_frames", "identity_strength", "figure_study_rate",
            "fashion_study_rate", "quality_score", "cluster_count",
            "caption_coverage", "study_coverage", "sharpness_mean",
            "recipe_name")
    for k in keys:
        if k in rec["signals"] and rec["signals"][k] not in (None, ""):
            lines.append(f"- {k}: {rec['signals'][k]}")
    return "\n".join(lines)
