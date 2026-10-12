"""Stack intelligence - explainable multi-LoRA stack resolution.

Takes a candidate stack (identity anchor + concept LoRAs), applies the
weight-intelligence heuristics, detects conflicts/overlaps/identity risk,
normalizes excessive concept strength, and returns a fully explained,
UI-ready recommendation with reason codes and an identity preservation
score.  Pure stdlib; block weights come from the base-model profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .lora_explorer import FAMILY_WEIGHT_RANGES, LoraCard

# thresholds (weight intelligence guidelines)
CONCEPT_STRENGTH_MODERATE = 1.20
CONCEPT_STRENGTH_STRONG = 1.60
IDENTITY_ANCHOR_MIN = 0.60
IDENTITY_WARN = 0.75
IDENTITY_STRONG_WARN = 0.60
FAMILY_OVERLAP_LIMIT = 2

# concept families whose goals tend to fight each other when stacked
CONFLICTING_FAMILIES: set[frozenset] = {
    frozenset(("style", "texture")),
    frozenset(("style", "refinement")),
    frozenset(("environment", "composition")),
}

_RISK_PENALTY = {"none": 0.0, "low": 0.02, "medium": 0.05, "high": 0.10}

# Concept priority: who yields first when the stack must be rebalanced.
# Identity is never auto-reduced; lower-priority concepts are trimmed first.
CONCEPT_PRIORITY: dict[str, int] = {
    "identity": 100, "character": 90,
    "wardrobe": 60, "fashion": 60, "lighting": 60, "pose": 60,
    "composition": 60, "camera": 55,
    "detail": 40, "refinement": 40, "texture": 40,
    "environment": 35, "style": 30,
    "unknown": 20,
}

# Identity risk levels (from the preservation score)
RISK_LEVELS = (("stable", 0.85), ("watch", 0.75),
               ("elevated", 0.60), ("high", 0.0))

# Warning severities: info < advisory < caution < critical
SEVERITY_ORDER = {"info": 0, "advisory": 1, "caution": 2, "critical": 3}


def risk_level_for(score: float) -> str:
    for name, floor in RISK_LEVELS:
        if score >= floor:
            return name
    return "high"


@dataclass
class StackWarning:
    code: str
    severity: str        # info | advisory | caution | critical
    message: str


@dataclass
class LoraStackItem:
    lora_id: str
    weight: float
    family: str
    identity_risk: str = "medium"
    reason: str = ""
    blocks_cli: str = ""
    requested_weight: float = 0.0      # what the user/sliders asked for
    adjusted: bool = False             # resolver changed it
    pinned: bool = False               # manual override - never auto-adjusted
    priority: int = 50


@dataclass
class StackRecommendation:
    code: str
    message: str
    proposed_weights: dict = field(default_factory=dict)   # lora_id -> w


@dataclass
class ResolvedLoraStack:
    base_model: str
    identity_anchor: Optional[LoraStackItem]
    concept_loras: list[LoraStackItem] = field(default_factory=list)
    warnings: list[StackWarning] = field(default_factory=list)
    total_concept_strength: float = 0.0
    identity_preservation_score: float = 1.0
    reason_codes: list[str] = field(default_factory=list)
    risk_level: str = "stable"
    conflicts: list[str] = field(default_factory=list)
    recommendations: list[StackRecommendation] = field(default_factory=list)
    influence_pressure: list[dict] = field(default_factory=list)
    recommended_weights: dict = field(default_factory=dict)
    summary: str = ""

    def to_json(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _hint_priority(base: int, hint: str) -> int:
    """Sidecar priority hints adjust resolver priority conservatively."""
    from .concept_metadata import priority_for_hint
    return priority_for_hint(base, hint)


def _role_for_family(family: str) -> str:
    """Map a concept family to a base-model block-weight role."""
    return {"identity": "primary", "character": "primary",
            "wardrobe": "wardrobe", "fashion": "wardrobe",
            "detail": "refiner", "refinement": "refiner",
            "texture": "refiner"}.get(family, "flavor")


def resolve_stack(cards: list[LoraCard],
                  weights: Optional[dict[str, float]] = None,
                  base_model: Optional[str] = None,
                  normalize: bool = True,
                  pinned: Optional[set] = None) -> ResolvedLoraStack:
    """Resolve a selected set of LoRAs into an explained, identity-first
    stack.

    `weights` are the requested per-LoRA weights (otherwise profile
    defaults); `pinned` LoRAs are manual overrides - never auto-adjusted,
    only warned about.  Identity preservation is the highest-priority
    behavior: when the stack must be rebalanced, concept weights are
    trimmed lowest-priority-first and the identity anchor is never
    reduced automatically.
    """
    from .base_models import blocks_string, detect_profile
    prof = detect_profile(base_model)
    weights = weights or {}
    pin = set(pinned or ())

    items: list[LoraStackItem] = []
    for c in cards:
        fam = c.profile.family
        if fam not in FAMILY_WEIGHT_RANGES:
            fam = "unknown"
        w_req = float(weights.get(c.lora_id, c.profile.weight_default))
        w = max(0.0, min(w_req, 1.2))
        lo, hi = FAMILY_WEIGHT_RANGES.get(fam, (0.10, 0.30))
        items.append(LoraStackItem(
            lora_id=c.lora_id, weight=round(w, 2), family=fam,
            identity_risk=c.profile.identity_risk,
            reason=(f"{fam} concept at {w:.2f} "
                    f"(recommended {lo:.2f}-{hi:.2f})"),
            blocks_cli=blocks_string(prof["blocks"][_role_for_family(fam)]),
            requested_weight=round(w_req, 2),
            adjusted=abs(w - w_req) > 1e-9,
            pinned=c.lora_id in pin,
            priority=_hint_priority(
                CONCEPT_PRIORITY.get(fam, CONCEPT_PRIORITY["unknown"]),
                getattr(c.profile, "priority_hint", "normal"))))

    id_items = [i for i in items if i.family in ("identity", "character")]
    anchor = max(id_items, key=lambda i: i.weight) if id_items else None
    concepts = [i for i in items if i is not anchor]
    warnings: list[StackWarning] = []
    codes: list[str] = []
    conflicts: list[str] = []
    recs: list[StackRecommendation] = []

    # ----- identity anchor -----
    if anchor:
        anchor.reason = "Primary identity preservation anchor"
        codes.append("identity_anchor_present")
        if anchor.weight < IDENTITY_ANCHOR_MIN:
            warnings.append(StackWarning(
                "identity_anchor_weak", "critical",
                f"Identity anchor at {anchor.weight:.2f} - raise to "
                f">= {IDENTITY_ANCHOR_MIN:.2f} to hold facial consistency."))
            codes.append("identity_anchor_below_minimum")
            recs.append(StackRecommendation(
                "raise_identity_anchor",
                "Increase the identity anchor within its recommended "
                "range (0.65-0.85).",
                {anchor.lora_id: 0.75}))
    else:
        warnings.append(StackWarning(
            "identity_anchor_missing", "critical",
            "No identity/character LoRA selected - concept modulation "
            "without an identity anchor will drift the character."))
        codes.append("identity_anchor_missing")

    # ----- per-item range advisories -----
    for i in concepts:
        lo, hi = FAMILY_WEIGHT_RANGES.get(i.family, (0.10, 0.30))
        if i.weight > hi + 1e-9:
            sev = "caution" if i.family == "unknown" else "advisory"
            warnings.append(StackWarning(
                "weight_above_recommended", sev,
                f"'{i.lora_id}' at {i.weight:.2f} exceeds the {i.family} "
                f"recommended range ({lo:.2f}-{hi:.2f})."))
            codes.append("weight_above_recommended")
            if i.family == "unknown":
                codes.append("unknown_family_aggressive")

    # ----- total concept strength: priority-aware rebalance -----
    total = round(sum(i.weight for i in concepts), 2)
    if normalize and total > CONCEPT_STRENGTH_STRONG:
        excess = total - CONCEPT_STRENGTH_STRONG
        protected = {c.lora_id for c in cards
                     if getattr(c.profile, "priority_hint", "") == "anchor"}
        order = sorted([i for i in concepts
                        if not i.pinned and i.lora_id not in protected],
                       key=lambda i: (i.priority, -i.weight))
        for i in order:
            lo, _hi = FAMILY_WEIGHT_RANGES.get(i.family, (0.10, 0.30))
            floor = max(lo * 0.5, 0.05)
            cut = min(max(i.weight - floor, 0.0), excess)
            if cut > 0:
                i.weight = round(i.weight - cut, 2)
                i.adjusted = True
                i.reason += (f"; trimmed -{cut:.2f} (stack balance, "
                             f"priority {i.priority})")
                excess = round(excess - cut, 4)
            if excess <= 1e-9:
                break
        codes.append("concept_strength_normalized")
        total = round(sum(i.weight for i in concepts), 2)
        # absorb rounding drift so the ceiling is honored exactly
        drift = round(total - CONCEPT_STRENGTH_STRONG, 2)
        if 0 < drift <= 0.05:
            tail = next((i for i in reversed(order) if i.adjusted), None)
            if tail:
                tail.weight = round(tail.weight - drift, 2)
                total = round(sum(i.weight for i in concepts), 2)
        if excess > 1e-9:
            warnings.append(StackWarning(
                "pinned_prevent_rebalance", "caution",
                "Pinned weights prevent a full rebalance - total concept "
                "strength remains above the studio-safe ceiling."))
            codes.append("pinned_prevent_rebalance")
    if total > CONCEPT_STRENGTH_STRONG:
        warnings.append(StackWarning(
            "concept_strength_excessive", "critical",
            f"Total concept strength {total:.2f} > "
            f"{CONCEPT_STRENGTH_STRONG} - identity will not hold."))
    elif total > CONCEPT_STRENGTH_MODERATE:
        warnings.append(StackWarning(
            "concept_strength_high", "caution",
            f"Total concept strength {total:.2f} > "
            f"{CONCEPT_STRENGTH_MODERATE} - watch facial consistency; "
            f"consider lowering the weakest concept by 0.1."))
        codes.append("concept_strength_high")
        low = min(concepts, key=lambda i: (i.priority, i.weight),
                  default=None)
        if low:
            recs.append(StackRecommendation(
                "lower_total_strength",
                f"Lower total concept strength: reduce "
                f"'{low.lora_id}' by 0.1.",
                {low.lora_id: round(max(low.weight - 0.1, 0.05), 2)}))

    # ----- family overlap -----
    fam_counts: dict[str, int] = {}
    for i in concepts:
        fam_counts[i.family] = fam_counts.get(i.family, 0) + 1
    for fam, n in fam_counts.items():
        if n > FAMILY_OVERLAP_LIMIT:
            warnings.append(StackWarning(
                "family_overlap", "caution",
                f"{n} LoRAs from the '{fam}' family - overlapping concepts "
                f"dilute each other; keep the strongest two."))
            codes.append(f"family_overlap_{fam}")

    # ----- conflicting families + explicit known conflicts -----
    fams = set(fam_counts)
    for pair in CONFLICTING_FAMILIES:
        if pair <= fams:
            a, b = sorted(pair)
            conflicts.append(f"{a}+{b}")
            warnings.append(StackWarning(
                "concept_conflict", "caution",
                f"'{a}' and '{b}' families pull composition in different "
                f"directions - reduce one or alternate per batch."))
            codes.append(f"concept_conflict_{a}_{b}")
            loser = min((i for i in concepts if i.family in (a, b)),
                        key=lambda i: (i.priority, i.weight), default=None)
            if loser:
                recs.append(StackRecommendation(
                    "deprioritize_conflict",
                    f"Deprioritize the conflicting concept: halve "
                    f"'{loser.lora_id}' ({loser.family}).",
                    {loser.lora_id: round(loser.weight / 2, 2)}))
    by_id = {c.lora_id: c for c in cards}
    active_families = {i.family for i in items}
    for i in concepts:
        hint = getattr(by_id[i.lora_id].profile, "priority_hint", "normal")
        if hint == "supporting" and anchor and i.weight > anchor.weight:
            warnings.append(StackWarning(
                "supporting_dominates", "caution",
                f"'{i.lora_id}' carries a supporting priority hint but "
                f"outweighs the identity anchor - reduce it below "
                f"{anchor.weight:.2f}."))
            codes.append("supporting_dominates_stack")
        if hint == "experimental":
            warnings.append(StackWarning(
                "experimental_concept", "advisory",
                f"'{i.lora_id}' is marked experimental - validate with a "
                f"consistency test before production use."))
        for famc in getattr(by_id[i.lora_id].profile,
                            "conflict_families", []):
            if famc and famc in active_families:
                conflicts.append(f"{i.lora_id}+family:{famc}")
                warnings.append(StackWarning(
                    "known_conflict", "caution",
                    f"'{i.lora_id}' declares a known conflict with the "
                    f"active '{famc}' family."))
                codes.append("known_family_conflict")
        for rival in by_id[i.lora_id].profile.known_conflicts:
            if rival in by_id and rival != i.lora_id:
                conflicts.append(f"{i.lora_id}+{rival}")
                warnings.append(StackWarning(
                    "known_conflict", "critical",
                    f"'{i.lora_id}' lists '{rival}' as a known conflict."))
                codes.append("known_conflict_pair")

    # ----- high identity-risk stacking -----
    high_risk = [i for i in concepts if i.identity_risk == "high"]
    if len(high_risk) >= 2:
        warnings.append(StackWarning(
            "identity_risk_stack", "critical",
            f"{len(high_risk)} high-identity-risk LoRAs stacked "
            f"({', '.join(i.lora_id for i in high_risk)}) - run a "
            f"Merge QA: Identity Preservation pass before trusting output."))
        codes.append("multiple_high_risk_loras")
    if len(concepts) >= 4:
        recs.append(StackRecommendation(
            "use_batch_variation",
            "Many visual modifiers are active at once - consider a "
            "controlled variation batch instead of stacking everything.",
            {}))
        codes.append("many_modifiers_active")

    # ----- identity preservation score + risk level -----
    score = 1.0
    score -= max(0.0, total - 0.8) * 0.25
    for i in concepts:
        score -= _RISK_PENALTY.get(i.identity_risk, 0.05)
    score -= 0.05 * sum(1 for c2 in codes
                        if c2.startswith("concept_conflict"))
    if anchor and anchor.weight >= IDENTITY_ANCHOR_MIN:
        score += 0.05
    else:
        score -= 0.20
    score = round(max(0.0, min(1.0, score)), 2)
    level = risk_level_for(score)
    if score < IDENTITY_STRONG_WARN:
        warnings.append(StackWarning(
            "identity_preservation_critical", "critical",
            f"Identity preservation score {score:.2f} < 0.60 - rebuild the "
            f"stack around the identity anchor."))
    elif score < IDENTITY_WARN:
        warnings.append(StackWarning(
            "identity_preservation_low", "caution",
            f"Identity preservation score {score:.2f} < 0.75 - lower "
            f"concept weights or drop the highest-risk LoRA."))
    if score < IDENTITY_WARN and concepts:
        worst = max(concepts, key=lambda i:
                    (_RISK_PENALTY.get(i.identity_risk, 0.05), i.weight))
        codes.append("safer_alternative_available")
        warnings.append(StackWarning(
            "safer_alternative", "info",
            f"Safer alternative: drop or halve '{worst.lora_id}' "
            f"({worst.family}, {worst.identity_risk} risk)."))
        recs.append(StackRecommendation(
            "reduce_highest_risk",
            f"Reduce the highest-pressure concept: halve "
            f"'{worst.lora_id}'.",
            {worst.lora_id: round(worst.weight / 2, 2)}))

    # ----- influence pressure ranking -----
    pressure = sorted((
        {"lora_id": i.lora_id, "family": i.family, "weight": i.weight,
         "pressure": round(i.weight
                           * (1 + 10 * _RISK_PENALTY.get(i.identity_risk,
                                                         0.05))
                           * (110 - i.priority) / 100, 3)}
        for i in concepts), key=lambda d: -d["pressure"])

    # ----- one-click suggested balance -----
    recommended = {i.lora_id: i.weight for i in items}
    for r in recs:
        recommended.update(r.proposed_weights)
    if anchor:
        recommended[anchor.lora_id] = max(
            recommended.get(anchor.lora_id, anchor.weight),
            anchor.weight)                       # anchor never reduced

    adjusted_n = sum(1 for i in items if i.adjusted)
    parts = [f"Identity {level} (preservation {score:.2f})."]
    parts.append(f"Anchor {anchor.lora_id} @ {anchor.weight:.2f}."
                 if anchor else "No identity anchor.")
    parts.append(f"{len(concepts)} concept LoRA(s), total strength "
                 f"{total:.2f}.")
    if adjusted_n:
        parts.append(f"{adjusted_n} weight(s) auto-adjusted for stack "
                     f"balance.")
    if pressure:
        parts.append(f"Highest influence pressure: "
                     f"{pressure[0]['lora_id']}.")

    warnings.sort(key=lambda w: -SEVERITY_ORDER.get(w.severity, 0))
    return ResolvedLoraStack(
        base_model=prof["label"], identity_anchor=anchor,
        concept_loras=concepts, warnings=warnings,
        total_concept_strength=total,
        identity_preservation_score=score,
        reason_codes=sorted(set(codes)),
        risk_level=level, conflicts=sorted(set(conflicts)),
        recommendations=recs, influence_pressure=pressure,
        recommended_weights={k: round(v, 2)
                             for k, v in recommended.items()},
        summary=" ".join(parts))
