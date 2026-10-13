"""MergeForge - production-grade LoRA merge engine.

Layers (single flat module, repo convention):

    library analyzer   - classify every LoRA with reason codes + health score
    compatibility      - pairwise CompatibilityResult (0-100, verdict, codes)
    weight recommender - merge-context weights (identity-first, conservative)
    recommendations    - five curated merge groups from the live library
    merge engine       - weighted_sum (delta-compose + SVD re-extract) and
                         concat (reuses Phase-7 merge.merge_state_dicts)
    recipes            - reproducible JSON recipes (schema v1, sha256 inputs)

Everything is additive: model files are only read, outputs never overwrite,
sidecar/manifest writes use new "mergeforge" fields only.
Backend: numpy (+ safetensors.numpy only when touching real files).
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Generator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import manifest
from .config import Project
from .lora_explorer import (
    FAMILY_WEIGHT_RANGES,
    LoraCard,
    scan_loras_cached,
    sidecar_path,
)
from .merge import block_group, merge_state_dicts, module_bases
from .util import now_iso, safe_slug

logger = logging.getLogger(__name__)

RECIPE_VERSION = 1
MERGE_METHODS = ("weighted_sum", "concat")
RECIPE_DIRNAME = "mergeforge_recipes"

# merge-context weight bands (tighter than playground stacking bands:
# merged weights are baked in forever, so stay conservative)
MERGE_WEIGHT_BANDS: dict[str, tuple[float, float]] = {
    "identity": (0.65, 0.75), "character": (0.65, 0.75),
    "wardrobe": (0.25, 0.35), "fashion": (0.25, 0.35),
    "style": (0.15, 0.30), "pose": (0.15, 0.30),
    "texture": (0.10, 0.25), "detail": (0.15, 0.30),
    "refinement": (0.15, 0.30), "lighting": (0.15, 0.30),
    "camera": (0.10, 0.25), "environment": (0.15, 0.30),
    "composition": (0.10, 0.25),
}

# role classification: family -> merge role
_FAMILY_ROLE: dict[str, str] = {
    "identity": "identity_anchor", "character": "identity_anchor",
    "wardrobe": "wardrobe_layer", "fashion": "wardrobe_layer",
    "style": "style_layer", "pose": "pose_layer",
    "texture": "detail_refiner", "detail": "detail_refiner",
    "refinement": "detail_refiner", "lighting": "ambience_layer",
    "camera": "ambience_layer", "environment": "ambience_layer",
    "composition": "ambience_layer",
}

VERDICTS = ("excellent", "good", "workable", "risky", "avoid")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@dataclass
class LoraHealth:
    lora_id: str
    score: int = 100                  # 0-100
    grade: str = "A"                  # A/B/C/D
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def assess_health(card: LoraCard) -> LoraHealth:
    """Score one library entry: metadata richness, sidecar, previews,
    sane dimensions.  Never raises."""
    score, reasons = 100, []
    if not card.sd_metadata and not card.network_dim:
        score -= 25
        reasons.append("no embedded safetensors training metadata")
    dim = card.network_dim
    if dim is not None and not (4 <= dim <= 256):
        score -= 15
        reasons.append(f"unusual network dim {dim}")
    if not card.metadata_source or card.metadata_source == "inferred":
        score -= 20
        reasons.append("no .concept.json sidecar (family inferred from name)")
    if not card.has_preview:
        score -= 10
        reasons.append("no preview image")
    try:
        p = Path(card.path)
        if p.suffix == ".safetensors" and p.exists() and p.stat().st_size < 1_000_000:
            score -= 20
            reasons.append("file under 1 MB - possibly truncated")
    except OSError:
        pass
    if card.profile.family not in MERGE_WEIGHT_BANDS:
        score -= 10
        reasons.append(f"family '{card.profile.family}' has no merge band")
    score = max(0, min(100, score))
    grade = "A" if score >= 85 else "B" if score >= 70 else \
            "C" if score >= 50 else "D"
    if not reasons:
        reasons.append("metadata, sidecar and preview all present")
    return LoraHealth(lora_id=card.lora_id, score=score, grade=grade,
                      reasons=reasons)


# ---------------------------------------------------------------------------
# Classification (reason-coded)
# ---------------------------------------------------------------------------

def classify_for_merge(card: LoraCard) -> dict:
    """Merge-role classification with reason codes."""
    fam = card.profile.family
    role = _FAMILY_ROLE.get(fam, "unclassified")
    codes = []
    codes.append("family_from_sidecar"
                 if card.metadata_source and card.metadata_source != "inferred"
                 else "family_from_name_hint")
    codes.append(f"family_{fam}")
    if card.profile.priority_hint == "anchor":
        role = "identity_anchor"
        codes.append("priority_hint_anchor")
    if card.profile.identity_risk == "high":
        codes.append("high_identity_risk")
    band = MERGE_WEIGHT_BANDS.get(fam)
    if band is None:
        codes.append("no_merge_band")
    return {"lora_id": card.lora_id, "role": role, "family": fam,
            "identity_risk": card.profile.identity_risk,
            "reason_codes": codes,
            "merge_band": list(band) if band else None}


# ---------------------------------------------------------------------------
# Library analyzer
# ---------------------------------------------------------------------------

def analyze_library(prj: Project,
                    cards: list[LoraCard] | None = None) -> dict:
    """Full library analysis: classification + health per LoRA, role
    counts, merge readiness."""
    cards = cards if cards is not None else scan_loras_cached(prj)
    entries, roles = [], {}
    for c in cards:
        cls = classify_for_merge(c)
        health = assess_health(c)
        roles[cls["role"]] = roles.get(cls["role"], 0) + 1
        entries.append({**cls, "path": c.path,
                        "display_name": c.display_name,
                        "weight_default": c.profile.weight_default,
                        "health": health.to_json()})
    anchors = roles.get("identity_anchor", 0)
    return {"entries": entries, "role_counts": roles,
            "library_size": len(cards),
            "merge_ready": anchors >= 1 and len(cards) >= 2,
            "notes": ([] if anchors else
                      ["no identity anchor found - merges will be "
                       "concept-only (no character lock)"])}


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

@dataclass
class CompatibilityResult:
    lora_a: str
    lora_b: str
    score: int = 100                       # 0-100
    verdict: str = "excellent"             # see VERDICTS
    reason_codes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommended_weights: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def _band_mid(fam: str) -> float:
    lo, hi = MERGE_WEIGHT_BANDS.get(
        fam, FAMILY_WEIGHT_RANGES.get(fam, (0.15, 0.30)))
    return round((lo + hi) / 2, 2)


def static_metadata(card: LoraCard) -> dict | None:
    """Load lora_metadata.py's per-LoRA metadata.json when it has been
    generated (<lora_dir>/_metadata/<name>.json).  Returns None otherwise, so
    callers fall back to family-band heuristics.  Read-only, never raises."""
    try:
        p = Path(card.path)
        mp = p.parent / "_metadata" / f"{p.stem}.json"
        if mp.exists():
            return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return None


def static_ideal_weight(md: dict | None) -> float | None:
    """Pull ideal_weight.default out of a loaded metadata.json, if present."""
    if md and isinstance(md.get("ideal_weight"), dict):
        d = md["ideal_weight"].get("default")
        if isinstance(d, (int, float)):
            return float(d)
    return None


def static_stack_note(md: dict | None) -> str:
    """Best human-readable stack note out of a loaded metadata.json."""
    if not md:
        return ""
    sn = md.get("stack_notes")
    if isinstance(sn, (list, tuple)):
        sn = "; ".join(str(x) for x in sn if x)
    return str(sn or "").strip()


def score_compatibility(a: LoraCard, b: LoraCard) -> CompatibilityResult:
    """Pairwise merge compatibility.  Deterministic, heuristic, honest:
    based on declared metadata only - no tensor inspection."""
    score, codes, warns = 100, [], []
    fa, fb = a.profile.family, b.profile.family

    # explicit conflicts (bidirectional, id and family level)
    for card, other in ((a, b), (b, a)):
        if other.lora_id in (card.profile.known_conflicts or []):
            score -= 60
            codes.append("declared_conflict_pair")
            warns.append(f"'{card.lora_id}' lists '{other.lora_id}' "
                         "as a known conflict")
        if other.profile.family in (card.profile.conflict_families or []):
            score -= 40
            codes.append("declared_conflict_family")
            warns.append(f"'{card.lora_id}' declares a conflict with "
                         f"family '{other.profile.family}'")

    # same-family duplication: two identities or two styles fight
    if fa == fb:
        if fa in ("identity", "character"):
            score -= 45
            codes.append("dual_identity")
            warns.append("two identity LoRAs in one merge usually blend "
                         "faces - keep one anchor")
        else:
            score -= 20
            codes.append("same_family_overlap")

    # identity risk pressure
    risk_pts = {"none": 0, "low": 0, "medium": 8, "high": 18}
    for card in (a, b):
        score -= risk_pts.get(card.profile.identity_risk, 8)
    if "high" in (a.profile.identity_risk, b.profile.identity_risk):
        codes.append("high_risk_member")

    # base model overlap
    bma = {m.upper() for m in (a.profile.base_models or [])}
    bmb = {m.upper() for m in (b.profile.base_models or [])}
    if bma and bmb and not (bma & bmb):
        score -= 25
        codes.append("base_model_mismatch")
        warns.append(f"declared base models differ: {sorted(bma)} vs "
                     f"{sorted(bmb)}")

    # complementary bonus signals
    roles = {_FAMILY_ROLE.get(fa, "x"), _FAMILY_ROLE.get(fb, "x")}
    if "identity_anchor" in roles and len(roles) == 2:
        codes.append("anchor_plus_concept")
    if {"wardrobe_layer", "detail_refiner"} <= roles:
        codes.append("wardrobe_detail_synergy")

    score = max(0, min(100, score))
    verdict = ("excellent" if score >= 85 else "good" if score >= 70 else
               "workable" if score >= 50 else "risky" if score >= 30
               else "avoid")
    return CompatibilityResult(
        lora_a=a.lora_id, lora_b=b.lora_id, score=score, verdict=verdict,
        reason_codes=sorted(set(codes)), warnings=warns,
        recommended_weights={a.lora_id: _band_mid(fa),
                             b.lora_id: _band_mid(fb)})


def stack_compatibility(cards: list[LoraCard]) -> dict:
    """All-pairs compatibility for a candidate merge set; overall score
    is the weakest pair (a merge is only as safe as its worst pairing)."""
    pairs = []
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            pairs.append(score_compatibility(cards[i], cards[j]))
    overall = min((p.score for p in pairs), default=100)
    verdict = ("excellent" if overall >= 85 else "good" if overall >= 70
               else "workable" if overall >= 50 else "risky"
               if overall >= 30 else "avoid")
    return {"pairs": [p.to_json() for p in pairs],
            "overall_score": overall, "overall_verdict": verdict}


# ---------------------------------------------------------------------------
# Weight recommender
# ---------------------------------------------------------------------------

def recommend_merge_weights(cards: list[LoraCard]) -> dict:
    """Identity-first merge weights.  Anchor gets its band, every other
    member gets the conservative midpoint of its merge band, damped when
    multiple non-identity members compete."""
    weights, notes = {}, []
    others = [c for c in cards
              if _FAMILY_ROLE.get(c.profile.family) != "identity_anchor"]
    damp = 1.0 if len(others) <= 2 else 0.85 if len(others) == 3 else 0.7
    if damp < 1.0:
        notes.append(f"{len(others)} concept layers - weights damped "
                     f"x{damp} to protect identity")
    used_static = 0
    for c in cards:
        fam = c.profile.family
        is_anchor = _FAMILY_ROLE.get(fam) == "identity_anchor"
        md = static_metadata(c)
        ideal = static_ideal_weight(md)
        if ideal is not None:
            used_static += 1
            base_w = ideal                       # measured from the tensors
            note = static_stack_note(md)
            if note:
                notes.append(f"{c.lora_id}: {note}")
        else:
            base_w = 0.70 if is_anchor else _band_mid(fam)   # family fallback
        # anchor keeps its measured/standard weight; concept layers are damped
        weights[c.lora_id] = round(base_w if is_anchor else base_w * damp, 2)
    if used_static:
        notes.insert(0, f"{used_static}/{len(cards)} weights from static LoRA "
                        "analysis (lora_metadata); others from family bands")
    total = round(sum(weights.values()), 2)
    if total > 1.6:
        notes.append(f"total weight {total} > 1.60 - consider dropping "
                     "the weakest concept layer")
    return {"weights": weights, "total": total, "notes": notes}


# ---------------------------------------------------------------------------
# Smart recommendations (five groups)
# ---------------------------------------------------------------------------

def _by_role(cards: list[LoraCard]) -> dict[str, list[LoraCard]]:
    out: dict[str, list[LoraCard]] = {}
    for c in cards:
        out.setdefault(_FAMILY_ROLE.get(c.profile.family, "unclassified"),
                       []).append(c)
    return out


def build_recommendations(prj: Project,
                          cards: list[LoraCard] | None = None) -> list[dict]:
    """Five curated merge groups assembled from the live library.
    Only groups whose members exist are returned."""
    cards = cards if cards is not None else scan_loras_cached(prj)
    roles = _by_role(cards)
    anchor = (roles.get("identity_anchor") or [None])[0]
    groups: list[dict] = []

    def emit(group_id, title, members, rationale):
        if len(members) < 2:
            return
        rec = recommend_merge_weights(members)
        compat = stack_compatibility(members)
        groups.append({
            "group_id": group_id, "title": title, "rationale": rationale,
            "members": [{"lora_id": m.lora_id,
                         "family": m.profile.family,
                         "weight": rec["weights"][m.lora_id]}
                        for m in members],
            "weight_total": rec["total"], "weight_notes": rec["notes"],
            "compatibility": {"score": compat["overall_score"],
                              "verdict": compat["overall_verdict"]}})

    if anchor:
        emit("character_complete", "Character Complete",
             [anchor] + (roles.get("wardrobe_layer") or [])[:1]
             + (roles.get("detail_refiner") or [])[:1],
             "Identity anchor + one wardrobe layer + one detail refiner: "
             "the standard deployable character build.")
        emit("character_style_fusion", "Character + Style Fusion",
             [anchor] + (roles.get("style_layer") or [])[:1],
             "Identity with a single style surface; style stays in its "
             "conservative band to protect facial consistency.")
        emit("scene_ready", "Scene-Ready Character",
             [anchor] + (roles.get("ambience_layer") or [])[:2],
             "Identity plus lighting/camera/environment ambience for "
             "consistent scene rendering.")
    emit("wardrobe_capsule", "Wardrobe Capsule",
         (roles.get("wardrobe_layer") or [])[:2]
         + (roles.get("detail_refiner") or [])[:1],
         "Garment-focused merge without identity - reusable across "
         "character LoRAs at stack time.")
    emit("refinement_pass", "Refinement Pass",
         (roles.get("detail_refiner") or [])[:3],
         "Detail/texture refiners folded into a single low-weight "
         "finishing LoRA.")
    return groups


# ---------------------------------------------------------------------------
# Merge engine - weighted_sum (delta compose + SVD re-extract)
# ---------------------------------------------------------------------------

def _as_2d(arr):
    """(r, in, kh, kw) -> (r, in*kh*kw); 2D passes through.
    Returns (mat2d, original_shape)."""
    shape = arr.shape
    if arr.ndim == 2:
        return arr, shape
    return arr.reshape(shape[0], -1), shape


def merge_weighted_sum(dicts: list[dict], weights: list[float],
                       blocks: list[dict],
                       target_rank: int = 0) -> tuple[dict, dict]:
    """True weighted-sum merge: per module, compose the full delta
        delta = sum_i  s_i * (up_i @ down_i),  s_i = w_i*(alpha_i/rank_i)*blk
    then re-extract a low-rank LoRA via SVD.  Mathematically exact up to
    rank truncation (concat is exact but grows rank; this keeps it fixed).
    Returns (merged_state_dict, stats)."""
    import numpy as np

    merged: dict = {}
    stats = {"modules": 0, "skipped_unmatched": 0, "zeroed": 0,
             "rank_truncated": 0}
    all_bases = sorted(set().union(*[module_bases(d.keys()) for d in dicts]))
    for base in all_bases:
        delta = None
        ranks, up_shape, down_shape = [], None, None
        for d, w, blk in zip(dicts, weights, blocks):
            dk, uk, ak = (f"{base}.lora_down.weight",
                          f"{base}.lora_up.weight", f"{base}.alpha")
            if dk not in d or uk not in d:
                continue
            down = np.asarray(d[dk], dtype=np.float32)
            up = np.asarray(d[uk], dtype=np.float32)
            rank = down.shape[0]
            alpha = (float(np.asarray(d[ak]).reshape(-1)[0])
                     if ak in d else float(rank))
            s = w * (alpha / max(1, rank)) * float(
                blk.get(block_group(base), 1.0))
            if s == 0.0:
                stats["zeroed"] += 1
                continue
            d2, dsh = _as_2d(down)
            orig_up_shape = up.shape
            u2 = (up.reshape(up.shape[0], rank) if up.ndim > 2 else up)
            ranks.append(rank)
            up_shape, down_shape = orig_up_shape, dsh
            contrib = s * (u2 @ d2)
            delta = contrib if delta is None else delta + contrib
        if delta is None:
            stats["skipped_unmatched"] += 1
            continue
        full_rank = min(sum(ranks), min(delta.shape))
        new_rank = min(target_rank or max(ranks), min(delta.shape))
        if new_rank < full_rank:
            stats["rank_truncated"] += 1
        try:
            u, sv, vt = np.linalg.svd(delta, full_matrices=False)
        except np.linalg.LinAlgError:
            stats["skipped_unmatched"] += 1
            continue
        root = np.sqrt(sv[:new_rank])
        m_up = (u[:, :new_rank] * root).astype(np.float16)
        m_down = (root[:, None] * vt[:new_rank]).astype(np.float16)
        if down_shape is not None and len(down_shape) > 2:
            m_down = m_down.reshape((new_rank,) + tuple(down_shape[1:]))
        if up_shape is not None and len(up_shape) > 2:
            m_up = m_up.reshape(tuple(up_shape[:1]) + (new_rank,)
                                + tuple(up_shape[2:]))
        merged[f"{base}.lora_down.weight"] = m_down
        merged[f"{base}.lora_up.weight"] = m_up
        merged[f"{base}.alpha"] = np.asarray(float(new_rank),
                                             dtype=np.float16)
        stats["modules"] += 1
    return merged, stats


def _unique_path(p: Path) -> Path:
    """Never overwrite: foo.safetensors -> foo_1.safetensors -> ..."""
    if not p.exists():
        return p
    for i in range(1, 1000):
        cand = p.with_name(f"{p.stem}_{i}{p.suffix}")
        if not cand.exists():
            return cand
    raise FileExistsError(f"cannot find free name near {p}")


# ---------------------------------------------------------------------------
# Recipes (schema v1, sha256-pinned, reproducible)
# ---------------------------------------------------------------------------

def _sha256(path: Path, limit: int = 0) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_recipe(name: str, inputs: list[tuple[str, float]],
                method: str = "weighted_sum",
                blocks: dict | None = None,
                base_model: str = "", notes: str = "",
                compat: dict | None = None,
                hash_files: bool = True) -> dict:
    """Reproducible merge recipe.  inputs: [(path, weight), ...]."""
    if method not in MERGE_METHODS:
        raise ValueError(f"method must be one of {MERGE_METHODS}")
    if len(inputs) < 2:
        raise ValueError("a merge recipe needs at least 2 inputs")
    recs = []
    for path, w in inputs:
        p = Path(path).expanduser()
        recs.append({
            "path": str(p), "name": p.stem, "weight": round(float(w), 4),
            "blocks": dict((blocks or {}).get(str(path),
                           {"te": 1.0, "down": 1.0, "mid": 1.0, "up": 1.0})),
            "sha256": (_sha256(p) if hash_files and p.exists() else None),
        })
    return {
        "recipe_version": RECIPE_VERSION,
        "recipe_id": f"mf_{uuid.uuid4().hex[:10]}",
        "name": str(name or "merge"),
        "created_at": now_iso(),
        "method": method,
        "inputs": recs,
        "base_model": base_model,
        "compatibility": compat or {},
        "notes": notes,
        "output": None,                       # filled after execution
    }


def validate_recipe(payload: dict) -> tuple[dict, list[str]]:
    """Tolerant normalization; never raises. Returns (recipe, problems)."""
    problems: list[str] = []
    p = dict(payload or {})
    if p.get("recipe_version") != RECIPE_VERSION:
        problems.append("unknown recipe_version - treating as v1")
    p.setdefault("recipe_id", f"mf_{uuid.uuid4().hex[:10]}")
    p.setdefault("name", "imported_merge")
    p.setdefault("method", "weighted_sum")
    if p["method"] not in MERGE_METHODS:
        problems.append(f"unsupported method '{p['method']}' - "
                        "falling back to weighted_sum")
        p["method"] = "weighted_sum"
    inputs = []
    for raw in (p.get("inputs") or []):
        if not isinstance(raw, dict) or not raw.get("path"):
            problems.append("dropped malformed input entry")
            continue
        try:
            w = float(raw.get("weight", 0))
        except (TypeError, ValueError):
            problems.append(f"bad weight on {raw.get('path')} - dropped")
            continue
        inputs.append({"path": str(raw["path"]),
                       "name": raw.get("name") or Path(raw["path"]).stem,
                       "weight": w,
                       "blocks": dict(raw.get("blocks") or
                                      {"te": 1.0, "down": 1.0,
                                       "mid": 1.0, "up": 1.0}),
                       "sha256": raw.get("sha256")})
    p["inputs"] = inputs
    if len(inputs) < 2:
        problems.append("fewer than 2 valid inputs")
    return p, problems


def recipe_dir(output_base: Path) -> Path:
    d = Path(output_base).expanduser() / RECIPE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_recipe(output_base: Path, recipe: dict) -> Path:
    target = recipe_dir(output_base) / f"{recipe['recipe_id']}.json"
    target.write_text(json.dumps(recipe, indent=2))
    try:
        conn = manifest.connect(Path(output_base))
        manifest.meta_set(conn, f"mergeforge:{recipe['recipe_id']}",
                          json.dumps({"name": recipe["name"],
                                      "created_at": recipe["created_at"],
                                      "method": recipe["method"],
                                      "output": recipe.get("output")}))
        conn.close()
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    return target


def load_recipes(output_base: Path) -> list[dict]:
    out = []
    d = Path(output_base).expanduser() / RECIPE_DIRNAME
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json"), reverse=True):
        try:
            rec, _ = validate_recipe(json.loads(f.read_text()))
            out.append(rec)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            continue
    return out


# ---------------------------------------------------------------------------
# Recipe execution
# ---------------------------------------------------------------------------

def execute_recipe(prj: Project, output_base: Path, recipe: dict,
                   output_name: str = "",
                   target_rank: int = 0) -> Generator[str, None, None]:
    """Run a recipe against real .safetensors files.  Validates inputs
    (existence, sha256 drift, key overlap), merges, writes a brand-new
    file (never overwrites), records sidecar + manifest fragments."""
    rec, problems = validate_recipe(recipe)
    for pb in problems:
        yield f"  recipe note: {pb}"
    if len(rec["inputs"]) < 2:
        yield "FATAL: need at least 2 valid inputs."
        return
    try:
        from safetensors.numpy import load_file, save_file
    except Exception as exc:
        yield f"FATAL: merge needs numpy + safetensors: {exc}"
        return

    dicts, weights, blocks, names = [], [], [], []
    for item in rec["inputs"]:
        p = Path(item["path"]).expanduser()
        if not p.exists():
            yield f"FATAL: input not found: {p}"
            return
        if item.get("sha256"):
            actual = _sha256(p)
            if actual != item["sha256"]:
                yield (f"  WARNING: sha256 drift on {p.name} - file changed "
                       "since recipe was written (continuing)")
        try:
            sd = load_file(str(p))
        except Exception as exc:
            yield (f"FATAL: {p.name} is not a readable .safetensors file "
                   f"({exc.__class__.__name__}: {exc}). The file may be "
                   "truncated, corrupted, or a different format - "
                   "re-export or re-download it, then retry.")
            return
        if not module_bases(sd.keys()):
            yield f"FATAL: {p.name} has no kohya lora_down/up modules."
            return
        dicts.append(sd)
        weights.append(float(item["weight"]))
        blocks.append(dict(item["blocks"]))
        names.append(p.stem)
        yield f"  loaded {p.name} (w={item['weight']}, blocks={item['blocks']})"

    common = set.intersection(*[module_bases(d.keys()) for d in dicts])
    union = set.union(*[module_bases(d.keys()) for d in dicts])
    overlap = 100.0 * len(common) / max(1, len(union))
    yield (f"  module overlap: {len(common)}/{len(union)} "
           f"({overlap:.0f}%) shared")
    if overlap < 30:
        yield ("  WARNING: low module overlap - inputs may target "
               "different architectures.")

    method = rec["method"]
    yield f"Merging via {method}..."
    if method == "weighted_sum":
        merged, stats = merge_weighted_sum(dicts, weights, blocks,
                                           target_rank=target_rank)
    else:
        merged, stats = merge_state_dicts(dicts, weights, blocks)
    if not merged:
        yield "FATAL: no mergeable modules produced."
        return

    out_dir = Path(prj.lora_output_dir
                   or (Path(output_base) / "LORA_OUTPUT")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _unique_path(
        out_dir / f"{safe_slug(output_name or rec['name'])}.safetensors")

    fragment = {"recipe_id": rec["recipe_id"], "name": rec["name"],
                "method": method, "created_at": now_iso(),
                "inputs": [{"name": n, "weight": w}
                           for n, w in zip(names, weights)],
                "stats": stats}
    save_file(merged, str(out_path),
              metadata={"pixel_curator_mergeforge": json.dumps(fragment)})
    rec["output"] = str(out_path)
    save_recipe(Path(output_base), rec)

    # additive sidecar for the new artifact
    try:
        sc = sidecar_path(out_path)
        data = (json.loads(sc.read_text()) if sc.exists() else {})
        data.setdefault("display_name", rec["name"])
        data["mergeforge"] = fragment
        sc.write_text(json.dumps(data, indent=2))
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass

    yield (f"\nMERGEFORGE DONE -> {out_path}\n"
           f"Method: {method} | modules: {stats['modules']} "
           f"| skipped: {stats['skipped_unmatched']} "
           f"| zeroed: {stats['zeroed']}"
           + (f" | rank-truncated: {stats['rank_truncated']}"
              if "rank_truncated" in stats else "")
           + f"\nRecipe saved: {RECIPE_DIRNAME}/{rec['recipe_id']}.json\n"
           f"Test it: pixel-curator previews --lora \"{out_path}\"")


# ---------------------------------------------------------------------------
# Guided wizard payload (UI step engine)
# ---------------------------------------------------------------------------

def wizard_plan(prj: Project, lora_ids: list[str],
                cards: list[LoraCard] | None = None) -> dict:
    """One call powering the guided merge wizard: selection -> compat,
    recommended weights, suggested method, ready-to-save recipe draft."""
    cards = cards if cards is not None else scan_loras_cached(prj)
    by_id = {c.lora_id: c for c in cards}
    chosen = [by_id[i] for i in lora_ids if i in by_id]
    missing = [i for i in lora_ids if i not in by_id]
    if len(chosen) < 2:
        return {"ready": False,
                "error": "select at least 2 LoRAs from the library",
                "missing": missing}
    compat = stack_compatibility(chosen)
    rec = recommend_merge_weights(chosen)
    inputs = [(c.path, rec["weights"][c.lora_id]) for c in chosen]
    draft = make_recipe(
        name="_".join(safe_slug(c.lora_id)[:16] for c in chosen[:3]),
        inputs=inputs, method="weighted_sum",
        base_model=getattr(prj, "base_model", "") or "",
        compat={"score": compat["overall_score"],
                "verdict": compat["overall_verdict"]},
        hash_files=False)                      # hash at execution time
    return {"ready": compat["overall_verdict"] != "avoid",
            "missing": missing,
            "compatibility": compat,
            "weights": rec,
            "classification": [classify_for_merge(c) for c in chosen],
            "health": [assess_health(c).to_json() for c in chosen],
            "recipe_draft": draft}
