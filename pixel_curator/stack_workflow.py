"""Character + Concept Stack workflow - guided one-click production flow.

Glues the existing pieces (Explorer cards, stack intelligence, presets,
Playground handoff) into one guided path:

    identity anchor -> concept layers -> resolved stack -> Playground

Also defines the v2 stack preset record (id, timestamps, requested vs
resolved weights, slider state, preservation score, warnings, handoff
snapshot) with validation and graceful fallback for older presets.
"""
from __future__ import annotations

import uuid
from typing import Optional

from .config import Project
from .lora_explorer import LoraCard
from .stack_intelligence import ResolvedLoraStack
from .util import now_iso

PRESET_VERSION = 2


# ---------------------------------------------------------------------------
# Workflow overview - what the library offers, step by step
# ---------------------------------------------------------------------------

def workflow_overview(cards: list[LoraCard]) -> dict:
    """Library summarized for the guided builder: identity candidates and
    concept layers grouped by family (UI step 1 + 2)."""
    identity = [c.lora_id for c in cards
                if c.profile.family in ("identity", "character")]
    families: dict[str, list[str]] = {}
    for c in cards:
        if c.profile.family in ("identity", "character"):
            continue
        families.setdefault(c.profile.family, []).append(c.lora_id)
    return {"identity_candidates": sorted(identity),
            "concept_families": {k: sorted(v)
                                 for k, v in sorted(families.items())},
            "library_size": len(cards),
            "ready": bool(identity)}


# ---------------------------------------------------------------------------
# Preset v2 records
# ---------------------------------------------------------------------------

def make_stack_preset(name: str, stack: ResolvedLoraStack,
                      requested_weights: Optional[dict] = None,
                      slider_state: Optional[dict] = None,
                      notes: str = "") -> dict:
    """Resolved stack -> v2 preset payload (human-readable, manifest-ready)."""
    anchor = stack.identity_anchor
    return {
        "preset_version": PRESET_VERSION,
        "preset_id": f"stack_{uuid.uuid4().hex[:10]}",
        "name": name,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "identity": ({"lora_id": anchor.lora_id, "weight": anchor.weight}
                     if anchor else None),
        "concepts": [{"lora_id": i.lora_id, "weight": i.weight,
                      "family": i.family,
                      "requested_weight": i.requested_weight,
                      "adjusted": i.adjusted}
                     for i in stack.concept_loras],
        "requested_weights": dict(requested_weights or {}),
        "slider_state": dict(slider_state or {}),
        "preservation_score": stack.identity_preservation_score,
        "risk_level": stack.risk_level,
        "warnings": [w.message for w in stack.warnings],
        "reason_codes": list(stack.reason_codes),
        "base_model": stack.base_model,
        "handoff": {"loras": ([[anchor.lora_id, anchor.weight]]
                              if anchor else [])
                    + [[i.lora_id, i.weight]
                       for i in stack.concept_loras]},
        "notes": notes,
    }


def validate_stack_preset(payload: dict) -> tuple[dict, list[str]]:
    """Normalize any preset payload (v2, v1 {'sel': ...}, or partial) into
    a v2-shaped record.  Never raises; returns (record, warnings)."""
    warnings: list[str] = []
    p = dict(payload or {})
    if p.get("preset_version") == PRESET_VERSION and "handoff" in p:
        # forward shape - light touch
        p.setdefault("concepts", [])
        p.setdefault("slider_state", {})
        return p, warnings
    # legacy: {'sel': {lora_id: weight}, 'slider_state': {...}, ...}
    sel = p.get("sel") or p.get("requested_weights") or {}
    if sel:
        warnings.append("legacy preset normalized to v2")
    loras = [[k, round(float(v), 2)] for k, v in sel.items()]
    return ({
        "preset_version": PRESET_VERSION,
        "preset_id": f"stack_{uuid.uuid4().hex[:10]}",
        "name": str(p.get("name") or "imported_preset"),
        "created_at": str(p.get("created_at") or now_iso()),
        "updated_at": now_iso(),
        "identity": None,
        "concepts": [],
        "requested_weights": dict(sel),
        "slider_state": dict(p.get("slider_state") or {}),
        "preservation_score": p.get("preservation_score"),
        "risk_level": p.get("risk_level") or "unknown",
        "warnings": list(p.get("warnings") or []),
        "reason_codes": [],
        "base_model": str(p.get("base_model") or ""),
        "handoff": {"loras": loras},
        "notes": str(p.get("notes") or ""),
    }, warnings)


# ---------------------------------------------------------------------------
# One-click handoff
# ---------------------------------------------------------------------------

def send_stack_to_playground(prj: Project, name: str,
                             stack: ResolvedLoraStack):
    """Resolved stack -> complete Playground preset (base model, sampler,
    prompt structure, negative) via the existing profile-driven writer."""
    from .pipeline_dag import write_playground_preset
    loras = ([(stack.identity_anchor.lora_id, stack.identity_anchor.weight)]
             if stack.identity_anchor else [])
    loras += [(i.lora_id, i.weight) for i in stack.concept_loras]
    if not loras:
        raise ValueError("empty stack - select an identity anchor and at "
                         "least one concept layer first")
    return write_playground_preset(prj, name, loras)

def diff_stack_presets(a: dict, b: dict) -> dict:
    """Field-by-field comparison of two stack presets (any generation)."""
    pa, _ = validate_stack_preset(a)
    pb, _ = validate_stack_preset(b)
    wa = {l[0]: l[1] for l in pa["handoff"]["loras"]}
    wb = {l[0]: l[1] for l in pb["handoff"]["loras"]}
    rows = [{"lora_id": k, "a": wa.get(k), "b": wb.get(k),
             "delta": (round(wb[k] - wa[k], 2)
                       if k in wa and k in wb else None)}
            for k in sorted(set(wa) | set(wb))]
    return {"a_name": pa["name"], "b_name": pb["name"], "rows": rows,
            "only_in_a": sorted(set(wa) - set(wb)),
            "only_in_b": sorted(set(wb) - set(wa)),
            "score_a": pa.get("preservation_score"),
            "score_b": pb.get("preservation_score")}


# TODO(extension): per-LoRA response-curve overrides from sidecar
# control_axes can plug into concept_control.map_slider_to_weight here once
# axis ids are bound to sliders; preset diff view can compare two
# validate_stack_preset() outputs field by field.
