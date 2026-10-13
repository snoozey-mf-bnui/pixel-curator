"""Study Intelligence Layer - professional study classification for frames.

Classifies curated frames for artistic figure studies, fashion editorial
work, lingerie/fashion studies, and form-and-proportion studies, using only
signals already in the manifest (captions, framing, identity similarity,
sharpness, brightness, clusters).  Pure stdlib, resumable, non-destructive;
results live in the `study_labels` table (schema v7) and feed the Dataset
Factory recipes, Stack Planner, Review filters, and Playground presets.

Extension hooks: `SIGNAL_HOOKS` lets optional local models (pose estimation,
aesthetic scoring, CLIP probes) contribute scores without becoming required
dependencies.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

from . import manifest
from .config import Project
from .util import now_iso, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary - caption-signal lexicons (WD14 tag vocabulary -> study signals)
# ---------------------------------------------------------------------------

# Wardrobe / garment-structure signals (fashion editorial suitability)
GARMENT_TAGS = {
    "dress", "skirt", "shirt", "jacket", "coat", "jeans", "pants", "shorts",
    "sweater", "hoodie", "suit", "blouse", "t-shirt", "denim", "boots",
    "heels", "hat", "scarf", "belt", "stockings", "thighhighs", "gloves",
}

# Intimate-apparel / lingerie-fashion study signals
INTIMATE_APPAREL_TAGS = {
    "lingerie", "bra", "underwear", "panties", "camisole", "negligee",
    "babydoll", "bodysuit", "swimsuit", "bikini", "garter_straps",
    "garter_belt", "lace", "nightgown", "robe",
}

# Pose / movement-study signals
POSE_TAGS = {
    "standing", "sitting", "lying", "kneeling", "walking", "stretching",
    "arms_up", "hands_on_hips", "leaning", "crossed_legs", "looking_back",
    "dynamic_pose", "contrapposto", "jumping", "dancing",
}

# Frames carrying these caption indicators are routed to manual review for
# study exports - study datasets stay editorial and unambiguous by default.
REVIEW_GATE_TAGS = {
    "nude", "completely_nude", "topless", "bottomless", "explicit",
    "nsfw", "uncensored", "sex",
}

STUDY_CATEGORIES = (
    "figure_study_candidate",
    "fashion_study_candidate",
    "lingerie_fashion_candidate",
    "form_proportion_candidate",
)

# Optional-signal extension hooks: name -> fn(frame_row_dict) -> dict of
# {score_name: float} merged into the signal set.  Register from [ai]/[cluster]
# extras without adding hard dependencies here.
SIGNAL_HOOKS: dict[str, Callable[[dict], dict]] = {}

CONFIDENCE_FLOOR = 0.45        # below -> needs_review
EXPORT_FLOOR = 0.55            # export_eligible threshold


# Schema lives in manifest.MIGRATIONS v7 (study_labels table) - additive,
# auto-applied on connect(); pre-v7 manifests upgrade transparently.


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@dataclass
class StudyConfig:
    output_base: Path
    rescan: bool = False           # re-classify frames that already have labels
    batch: int = 500               # commit cadence


def _caption_tokens(caption: str | None, tags_json: str | None) -> set:
    toks: set = set()
    if caption:
        toks.update(t.strip().lower().replace(" ", "_")
                    for t in caption.split(",") if t.strip())
    if tags_json:
        try:
            toks.update(str(t).lower() for t in json.loads(tags_json))
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
    return toks


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def classify_frame(row: dict) -> dict:
    """Score one frame from manifest signals.  `row` keys: sharpness,
    brightness, framing, identity_sim, face_count, caption, tags_json,
    cluster_id.  Returns the full study_labels record (sans frame_id)."""
    toks = _caption_tokens(row.get("caption"), row.get("tags_json"))
    framing = (row.get("framing") or "none").lower()
    sharp = float(row.get("sharpness") or 0.0)
    bright = row.get("brightness")
    ident = row.get("identity_sim")
    faces = int(row.get("face_count") or 0)
    reasons: list[str] = []
    tags: list[str] = []

    # --- technical gates ---
    sharp_ok = sharp >= 50
    if sharp_ok:
        reasons.append("low_blur_pass")
    light_ok = bright is not None and 35 <= float(bright) <= 215
    if light_ok:
        reasons.append("lighting_quality_pass")

    # --- identity lock ---
    if ident is not None:
        identity_lock = _clamp(float(ident) / 0.7)
        if float(ident) >= 0.35 and faces > 0:
            reasons.append("face_identity_visible")
        elif faces == 0:
            reasons.append("identity_visibility_limited")
    else:
        identity_lock = 0.3 if faces == 0 else 0.5
        reasons.append("identity_visibility_limited")

    # --- pose clarity (framing readability x sharpness) ---
    framing_pose = {"full_body": 1.0, "upper_body": 0.7,
                    "portrait": 0.4, "closeup": 0.2}.get(framing, 0.3)
    pose_hits = toks & POSE_TAGS
    pose_clarity = _clamp(0.7 * framing_pose + 0.15 * (1.0 if sharp_ok else 0.4)
                          + (0.15 if pose_hits else 0.0))
    if framing == "full_body":
        reasons.append("clear_full_body_framing")
        tags.append("full_body_frame")
    elif framing == "upper_body":
        tags.append("upper_body_frame")
    elif framing == "closeup":
        tags.append("detail_frame")
    if pose_hits:
        reasons.append("strong_pose_readability")
        tags.append("movement_study_frame")
    else:
        reasons.append("pose_detection_unavailable")

    # --- silhouette clarity ---
    silhouette = _clamp(0.55 * framing_pose + 0.25 * (1.0 if light_ok else 0.3)
                        + 0.20 * (1.0 if sharp_ok else 0.3))

    # --- garment visibility ---
    garment_hits = toks & GARMENT_TAGS
    apparel_hits = toks & INTIMATE_APPAREL_TAGS
    garment_visibility = _clamp(
        0.25 * min(len(garment_hits | apparel_hits), 3)
        + (0.25 if framing in ("upper_body", "full_body") else 0.0))
    if garment_hits or apparel_hits:
        reasons.append("garment_structure_visible")
        reasons.append("caption_signal_matched")
        tags.append("wardrobe_focus_frame")

    # --- composite study scores ---
    figure = _clamp(0.40 * pose_clarity + 0.30 * silhouette
                    + 0.30 * identity_lock)
    fashion = _clamp(0.45 * garment_visibility + 0.25 * pose_clarity
                     + 0.30 * identity_lock)
    lingerie_fashion = _clamp(
        (0.6 if apparel_hits else 0.0)
        + 0.20 * silhouette + 0.20 * identity_lock) if apparel_hits else 0.0
    form_proportion = _clamp(0.5 * (1.0 if framing == "closeup" else
                                    0.6 if framing == "full_body" else 0.2)
                             + 0.5 * (min(sharp, 150.0) / 150.0))
    if sharp_ok and light_ok and framing != "none":
        reasons.append("high_composition_quality")
    if light_ok:
        tags.append("studio_lighting_frame")
    if framing == "closeup" and sharp_ok:
        tags.append("anatomy_detail_frame")

    # --- primary category + confidence ---
    scores = {"figure_study_candidate": figure,
              "fashion_study_candidate": fashion,
              "lingerie_fashion_candidate": lingerie_fashion,
              "form_proportion_candidate": form_proportion}
    primary = max(scores, key=lambda k: scores[k])
    confidence = scores[primary]

    review = "auto"
    if toks & REVIEW_GATE_TAGS:
        review = "needs_review"
        reasons.append("requires_manual_review")
        tags.append("needs_review_study_label")
    elif confidence < CONFIDENCE_FLOOR:
        review = "needs_review"
        reasons.append("requires_manual_review")

    # optional-signal hooks (pose estimation, aesthetics, CLIP probes...)
    for hook in SIGNAL_HOOKS.values():
        try:
            for k, v in (hook(row) or {}).items():
                scores[k] = _clamp(float(v))
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass

    export_eligible = int(confidence >= EXPORT_FLOOR
                          and review == "auto"
                          and identity_lock >= 0.4 and sharp_ok)
    if export_eligible:
        reasons.append("sufficient_resolution")

    return {
        "study_primary": primary,
        "study_tags": json.dumps(sorted(set(tags))),
        "study_confidence": round(confidence, 3),
        "study_reason_codes": json.dumps(sorted(set(reasons))),
        "figure_study_score": round(figure, 3),
        "fashion_study_score": round(fashion, 3),
        "lingerie_fashion_score": round(lingerie_fashion, 3),
        "pose_clarity_score": round(pose_clarity, 3),
        "silhouette_clarity_score": round(silhouette, 3),
        "garment_visibility_score": round(garment_visibility, 3),
        "identity_lock_score": round(identity_lock, 3),
        "study_review_status": review,
        "study_export_eligible": export_eligible,
    }


def classify_generator(prj: Project,
                       cfg: StudyConfig) -> Generator[str, None, None]:
    """Classify all selected/packaged frames; resumable + manifest-tracked.
    Manual overrides are never overwritten."""
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)

    where = "f.status IN ('selected','packaged')"
    if not cfg.rescan:
        where += (" AND f.frame_id NOT IN "
                  "(SELECT frame_id FROM study_labels)")
    rows = conn.execute(
        f"SELECT f.frame_id, f.sharpness, f.brightness, f.cluster_id, "
        f"d.framing, d.identity_sim, d.face_count, "
        f"c.caption_text, c.tags_json "
        f"FROM frames f "
        f"LEFT JOIN detections d ON d.frame_id = f.frame_id "
        f"LEFT JOIN captions c ON c.frame_id = f.frame_id "
        f"WHERE {where} ORDER BY f.frame_id").fetchall()
    if not rows:
        yield "Nothing to classify (all frames labeled). Use --rescan to redo."
        return
    yield f"Study Intelligence Layer: classifying {len(rows)} frames..."

    counts: dict[str, int] = {}
    review_n = export_n = 0
    for i, r in enumerate(rows, 1):
        rec = classify_frame({
            "sharpness": r[1], "brightness": r[2], "cluster_id": r[3],
            "framing": r[4], "identity_sim": r[5], "face_count": r[6],
            "caption": r[7], "tags_json": r[8],
        })
        conn.execute(
            "INSERT INTO study_labels (frame_id, study_primary, study_tags, "
            "study_confidence, study_reason_codes, figure_study_score, "
            "fashion_study_score, lingerie_fashion_score, pose_clarity_score, "
            "silhouette_clarity_score, garment_visibility_score, "
            "identity_lock_score, study_review_status, study_export_eligible, "
            "manual_override, classified_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?) "
            "ON CONFLICT(frame_id) DO UPDATE SET "
            "study_primary=excluded.study_primary, "
            "study_tags=excluded.study_tags, "
            "study_confidence=excluded.study_confidence, "
            "study_reason_codes=excluded.study_reason_codes, "
            "figure_study_score=excluded.figure_study_score, "
            "fashion_study_score=excluded.fashion_study_score, "
            "lingerie_fashion_score=excluded.lingerie_fashion_score, "
            "pose_clarity_score=excluded.pose_clarity_score, "
            "silhouette_clarity_score=excluded.silhouette_clarity_score, "
            "garment_visibility_score=excluded.garment_visibility_score, "
            "identity_lock_score=excluded.identity_lock_score, "
            "study_review_status=excluded.study_review_status, "
            "study_export_eligible=excluded.study_export_eligible, "
            "classified_at=excluded.classified_at "
            "WHERE study_labels.manual_override = 0",
            (r[0], rec["study_primary"], rec["study_tags"],
             rec["study_confidence"], rec["study_reason_codes"],
             rec["figure_study_score"], rec["fashion_study_score"],
             rec["lingerie_fashion_score"], rec["pose_clarity_score"],
             rec["silhouette_clarity_score"], rec["garment_visibility_score"],
             rec["identity_lock_score"], rec["study_review_status"],
             rec["study_export_eligible"], now_iso()))
        counts[rec["study_primary"]] = counts.get(rec["study_primary"], 0) + 1
        review_n += rec["study_review_status"] == "needs_review"
        export_n += rec["study_export_eligible"]
        if i % cfg.batch == 0:
            conn.commit()
            yield f"  {i}/{len(rows)} classified..."
    conn.commit()
    manifest.meta_set(conn, "study_classified_at", now_iso())
    yield "Classification complete:"
    for cat in STUDY_CATEGORIES:
        yield f"  {cat:<30} {counts.get(cat, 0)}"
    yield f"  needs_review {review_n} | export_eligible {export_n}"


def set_manual_label(conn, frame_id: str, study_primary: str,
                     review_status: str = "approved") -> None:
    """Persist a manual override (survives re-classification)."""
    conn.execute(
        "INSERT INTO study_labels (frame_id, study_primary, "
        "study_review_status, manual_override, classified_at) "
        "VALUES (?,?,?,1,?) ON CONFLICT(frame_id) DO UPDATE SET "
        "study_primary=excluded.study_primary, "
        "study_review_status=excluded.study_review_status, "
        "manual_override=1, classified_at=excluded.classified_at",
        (frame_id, study_primary, review_status, now_iso()))
    conn.commit()


def study_report(conn) -> dict:
    """Aggregate study statistics (Review dashboard + CLI report)."""
    out: dict = {"categories": {}, "export_eligible": 0, "needs_review": 0}
    try:
        for cat, n in conn.execute(
                "SELECT study_primary, COUNT(*) FROM study_labels "
                "GROUP BY study_primary"):
            out["categories"][cat] = n
        out["export_eligible"] = conn.execute(
            "SELECT COUNT(*) FROM study_labels "
            "WHERE study_export_eligible=1").fetchone()[0]
        out["needs_review"] = conn.execute(
            "SELECT COUNT(*) FROM study_labels "
            "WHERE study_review_status='needs_review'").fetchone()[0]
        for col in ("identity_lock_score", "pose_clarity_score",
                    "garment_visibility_score", "silhouette_clarity_score"):
            v = conn.execute(
                f"SELECT AVG({col}) FROM study_labels").fetchone()[0]
            out[col + "_mean"] = round(v, 3) if v is not None else None
    except Exception:
        pass  # pre-v7 manifest - report stays empty (backward compatible)
    return out


# ---------------------------------------------------------------------------
# Dataset Factory recipes (same shape as builder.RECIPE_DEFAULTS)
# ---------------------------------------------------------------------------

STUDY_RECIPES: dict[str, dict] = {
    # Balanced artistic figure-study set with identity preservation.
    "figure_study_v1": {
        "type": "character", "repeats": 8, "max_total": 300,
        "max_per_video": 30, "min_identity": 0.40, "min_sharpness": 50,
        "quota": "full_body=0.40,upper_body=0.30,portrait=0.20,closeup=0.10",
        "val_fraction": 0.05,
        "study_primary": "figure_study_candidate",
        "study_min_confidence": 0.45,
    },
    # Wardrobe styling, garment structure, textile detail, editorial posing.
    "fashion_editorial_v1": {
        "type": "outfit", "repeats": 8, "max_total": 250,
        "max_per_video": 25, "min_identity": 0.35, "min_sharpness": 50,
        "quota": "full_body=0.35,upper_body=0.30,closeup=0.20,portrait=0.15",
        "val_fraction": 0.05,
        "study_primary": "fashion_study_candidate",
        "study_min_confidence": 0.45,
    },
    # Fit, silhouette, styling, fabric detail with strong identity lock.
    "lingerie_fashion_study_v1": {
        "type": "outfit", "repeats": 10, "max_total": 200,
        "max_per_video": 20, "min_identity": 0.45, "min_sharpness": 60,
        "quota": "full_body=0.35,upper_body=0.25,portrait=0.20,closeup=0.20",
        "val_fraction": 0.05,
        "study_primary": "lingerie_fashion_candidate",
        "study_min_confidence": 0.50,
        "study_export_only": True,       # excludes needs-review frames
    },
    # Identity-first set with figure/fashion support frames mixed in.
    "balanced_character_study_v1": {
        "type": "character", "repeats": 10, "max_total": 400,
        "max_per_video": 40, "min_identity": 0.40,
        "quota": "closeup=0.20,portrait=0.30,upper_body=0.25,full_body=0.25",
        "val_fraction": 0.05,
        # no study filter: the planner mixes all curated identity frames
    },
}


def register_study_recipes(prj: Project) -> list[str]:
    """Additively expose the study recipes in the project (existing recipes
    are never overwritten)."""
    added = []
    prj.recipes = prj.recipes or {}
    for name, rcp in STUDY_RECIPES.items():
        if name not in prj.recipes:
            prj.recipes[name] = dict(rcp)
            added.append(name)
    return added


# ---------------------------------------------------------------------------
# Stack Planner - study production modes
# ---------------------------------------------------------------------------

STACK_MODES: dict[str, dict] = {
    "character_identity": {
        "loras": [("character", "primary", 0.85)],
        "notes": "Identity LoRA alone - the reference for every QA compare.",
    },
    "character_fashion_study": {
        "loras": [("character", "primary", 0.85),
                  ("fashion_editorial", "wardrobe", 0.55)],
        "notes": "Identity stays dominant; reduce wardrobe weight if face "
                 "consistency drifts in the QA grid.",
    },
    "character_figure_study": {
        "loras": [("character", "primary", 0.85),
                  ("figure_study", "form", 0.50)],
        "notes": "Figure-study LoRA reinforces full-body framing, form and "
                 "proportion; keep secondary or pose language overwhelms "
                 "identity.",
    },
    "character_style": {
        "loras": [("character", "primary", 0.85),
                  ("style", "flavor", 0.35)],
        "notes": "Strong style LoRAs reduce identity fidelity - apply "
                 "cautiously and verify with the likeness matrix.",
    },
    "character_fashion_style": {
        "loras": [("character", "primary", 0.85),
                  ("fashion_editorial", "wardrobe", 0.50),
                  ("style", "flavor", 0.30)],
        "notes": "Three-way stacks compound drift: drop style first when "
                 "identity slips, wardrobe second.",
    },
    "character_figure_fashion_editorial": {
        "loras": [("character", "primary", 0.85),
                  ("figure_study", "form", 0.45),
                  ("fashion_editorial", "wardrobe", 0.45)],
        "notes": "Full editorial production stack. Run Merge QA: Identity "
                 "Preservation before promoting any merge of this stack.",
    },
}


def suggest_study_stack(mode: str, base_model: str | None = None) -> dict:
    """Stack recommendation for a study production mode: which LoRAs to
    load, weights, merge order, block weights, and conflict warnings."""
    from .base_models import blocks_string, detect_profile
    if mode not in STACK_MODES:
        raise KeyError(f"Unknown mode '{mode}'. "
                       f"Available: {', '.join(sorted(STACK_MODES))}")
    prof = detect_profile(base_model)
    m = STACK_MODES[mode]
    role_map = {"primary": "primary", "wardrobe": "wardrobe",
                "form": "refiner", "flavor": "flavor"}
    stack = []
    for lora_type, role, weight in m["loras"]:
        blocks = prof["blocks"][role_map[role]]
        stack.append({"type": lora_type, "role": role, "weight": weight,
                      "blocks": dict(blocks),
                      "blocks_cli": blocks_string(blocks)})
    warnings = [m["notes"],
                "Identity LoRA remains primary; load it first and merge it "
                "last so its weights dominate the concat result.",
                "Prefer separate LoRAs at runtime while iterating; merge "
                "only after a best-epoch sweep, and keep the un-merged "
                "checkpoints (merges are reversible only via the originals)."]
    if len(stack) >= 3:
        warnings.append("Conflict risk: two support LoRAs compete for the "
                        "same up-blocks; if textures degrade, lower the "
                        "weaker LoRA by 0.1 steps.")
    return {"mode": mode, "profile": prof["label"], "stack": stack,
            "merge_order": [s["type"] for s in reversed(stack)],
            "warnings": warnings}


# ---------------------------------------------------------------------------
# Playground presets (CyberRealistic Pony v18.0 CoreShift via base_models)
# ---------------------------------------------------------------------------

_PRESET_TEMPLATES: list[tuple[str, str, str, str]] = [
    # (preset name, study descriptor, framing/pose tail, QA note)
    ("Identity-Locked Figure Study",
     "artistic figure study",
     "full body framing, balanced standing pose, form and proportion "
     "clarity, soft studio lighting, clean background",
     "QA: face consistency at full-body distance; coherent proportions."),
    ("Fashion Editorial Study",
     "fashion editorial study",
     "full body framing, expressive editorial pose, garment structure "
     "visible, textile detail, studio lighting, clean composition",
     "QA: garment structure and fit; identity retained under styling."),
    ("Lingerie/Fashion Study",
     "lingerie fashion study",
     "full body silhouette, tasteful editorial styling, textile detail, "
     "soft studio lighting, clean composition",
     "QA: silhouette clarity and textile detail; identity lock intact."),
    ("Character + Figure Consistency Test",
     "artistic figure study",
     "upper body framing, relaxed pose, natural lighting, neutral "
     "background",
     "QA: compare against identity-only render at the same seed."),
    ("Character + Fashion Consistency Test",
     "fashion editorial study",
     "upper body framing, editorial styling, garment structure visible, "
     "studio lighting",
     "QA: wardrobe adherence without face drift; same-seed compare."),
    ("Merge QA - Identity Preservation",
     "neutral portrait",
     "upper body framing, natural expression, soft lighting, clean "
     "background",
     "QA: run before promoting any merge; likeness must match the "
     "un-merged identity LoRA."),
    ("Merge QA - Full-Body Consistency",
     "artistic figure study",
     "full body framing, balanced pose, form and proportion clarity, "
     "neutral lighting",
     "QA: anatomy coherence and proportion stability at distance."),
    ("Merge QA - Editorial Fashion",
     "fashion editorial study",
     "full body framing, editorial pose, garment structure visible, "
     "studio lighting",
     "QA: styling fidelity after merge; watch for textile detail loss."),
]


def write_study_presets(prj: Project, path: Path | None = None,
                        loras: list | None = None) -> Path:
    """Write the study preset pack for the project's base model profile."""
    from .base_models import detect_profile, preset_payload
    prof = detect_profile(prj.base_model)
    trig = f"{prj.trigger_token} {prj.class_word}".strip()
    target = Path(path) if path else (
        Path(__file__).resolve().parent / "outputs"
        / "playground_presets.json")
    presets: dict = {}
    try:
        presets = json.loads(target.read_text())
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    for name, study, tail, qa in _PRESET_TEMPLATES:
        body = preset_payload(prof, trig, base_model=prj.base_model,
                              loras=loras or [])
        body["prompt"] = (
            f"{prof['quality_prefix']}{trig}, consistent character identity, "
            f"{study}, {tail}, high detail, natural anatomy, "
            f"coherent proportions")
        body["_study"] = {"qa_notes": qa, "profile": prof["label"],
                          "written_at": now_iso()}
        presets[name] = body
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(presets, indent=2))
    return target
