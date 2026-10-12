"""Unified studio asset registry.

One read-only view of every studio asset: identity anchor LoRAs,
artistic concept LoRAs, base model + guidance models (ControlNet /
IP-Adapter / FaceID), training runs, evaluation reports, model cards,
figure-study batches and production packages. Reuses the existing LoRA
Explorer scan and Identity Integration detector; missing metadata
produces warnings and recommendations, never exceptions. Nothing is
moved or modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

IDENTITY_FAMILIES = ("identity", "character")


def _lora_assets(prj) -> tuple[list[dict], list[dict], list[str]]:
    """(identity anchors, concept loras, warnings) from the library."""
    warnings: list[str] = []
    try:
        from ..lora_explorer import FAMILY_WEIGHT_RANGES, scan_loras
        cards = scan_loras(prj)
    except Exception as exc:
        return [], [], [f"LoRA library scan unavailable: {exc}"]
    if not cards:
        warnings.append("No LoRAs found in the configured library - set "
                        "lora_output_dir / forge_root to your model "
                        "folders.")

    identity: list[dict] = []
    concept: list[dict] = []
    for c in cards:
        fam = getattr(c.profile, "family", "style")
        lo, hi = FAMILY_WEIGHT_RANGES.get(fam, (0.15, 0.35))
        has_meta = bool(c.metadata_source) or bool(c.concept_meta)
        entry = {
            "name": c.lora_id,
            "path": c.path,
            "family": fam,
            "role": ("identity_anchor" if fam in IDENTITY_FAMILIES
                     else "concept"),
            "recommended_weight_range": [lo, hi],
            "identity_risk": getattr(c.profile, "identity_risk",
                                     "medium"),
            "metadata_status": "complete" if has_meta else "inferred",
            "has_preview": c.has_preview,
            "warnings": ([] if has_meta else
                         ["No sidecar metadata - role and weights were "
                          "inferred; add a metadata sidecar for precise "
                          "stacking."]),
        }
        (identity if entry["role"] == "identity_anchor"
         else concept).append(entry)

    if not identity:
        warnings.append("No identity anchor LoRA detected - train or "
                        "register one for character consistency.")
    return identity, concept, warnings


def _guidance_models(prj) -> list[dict]:
    """ControlNet / IP-Adapter / FaceID readiness as registry rows."""
    try:
        from ..identity_integration import detect_identity_tools
        tools = detect_identity_tools(prj)
    except Exception:
        return []
    rows = []
    for key, entry in tools.items():
        rows.append({
            "name": key,
            "kind": ("ip_adapter_faceid" if "ip_adapter" in key
                     else "controlnet" if key.startswith("controlnet")
                     else "runtime" if "runtime" in key
                     else "guidance"),
            "status": entry.get("status", "missing"),
            "path": entry.get("path"),
        })
    return rows


def _run_assets(conn) -> list[dict]:
    from .. import manifest as mf
    from ..train.kohya import list_runs
    out = []
    for r in list_runs(conn):
        out.append({
            "name": r["name"], "status": r["status"],
            "dataset_version": r["dataset_version"],
            "preset": r["preset"],
            "has_sweep": bool(mf.meta_get(conn, f"sweep:{r['name']}")),
            "has_evaluation":
                bool(mf.meta_get(conn, f"evaluation:{r['name']}")),
            "has_model_card":
                bool(mf.meta_get(conn, f"model_card:{r['name']}")),
        })
    return out


def _batch_assets(output_base: Path) -> list[dict]:
    try:
        from ..artistic.review import list_batches
        return list_batches(output_base)
    except Exception:
        return []


def _package_assets(output_base: Path) -> list[dict]:
    out = []
    for root_name in ("figure_study_packages", "studio_handoffs"):
        root = Path(output_base) / root_name
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir(), reverse=True):
            if d.is_dir():
                out.append({"name": d.name, "kind": root_name,
                            "path": str(d),
                            "has_readme": (d / "README.md").exists()})
    return out


def build_asset_registry(prj, conn=None, output_base: Optional[Path] = None,
                         role: str = "", missing_metadata: bool = False
                         ) -> dict:
    """Assemble the unified asset registry.

    role filters LoRA rows ('identity_anchor' / 'concept'). When
    missing_metadata is True, only LoRAs with inferred (incomplete)
    metadata are returned. conn (read-only) and output_base are optional;
    absent inputs simply yield empty sections.
    """
    base = Path(output_base or prj.output_base).expanduser()
    identity, concept, warnings = _lora_assets(prj)

    loras = identity + concept
    if role in ("identity_anchor", "identity"):
        loras = [l for l in loras if l["role"] == "identity_anchor"]
    elif role in ("concept", "artistic"):
        loras = [l for l in loras if l["role"] == "concept"]
    if missing_metadata:
        loras = [l for l in loras if l["metadata_status"] != "complete"]

    runs = _run_assets(conn) if conn is not None else []
    batches = _batch_assets(base)
    packages = _package_assets(base)
    guidance = _guidance_models(prj)

    recommendations: list[str] = []
    incomplete = [l for l in identity + concept
                  if l["metadata_status"] != "complete"]
    if incomplete:
        recommendations.append(
            f"{len(incomplete)} LoRA(s) lack sidecar metadata - add "
            "metadata for precise role detection and stacking.")
    if not any(g["status"] == "found" for g in guidance):
        recommendations.append(
            "No guidance backends detected - ControlNet / IP-Adapter / "
            "FaceID planning will run in dry-run mode only.")

    return {
        "identity_anchors": identity,
        "concept_loras": concept,
        "loras": loras,
        "guidance_models": guidance,
        "base_model": getattr(prj, "base_model", "") or None,
        "runs": runs,
        "batches": batches,
        "packages": packages,
        "summary": {
            "identity_anchor_count": len(identity),
            "concept_lora_count": len(concept),
            "guidance_backends_available":
                sum(1 for g in guidance if g["status"] == "found"),
            "run_count": len(runs),
            "batch_count": len(batches),
            "package_count": len(packages),
        },
        "warnings": warnings,
        "recommendations": recommendations,
    }


def render_registry(reg: dict) -> str:
    s = reg["summary"]
    lines = ["STUDIO ASSET REGISTRY",
             f"  identity anchors: {s['identity_anchor_count']} | "
             f"concept LoRAs: {s['concept_lora_count']} | guidance "
             f"backends: {s['guidance_backends_available']}",
             f"  runs: {s['run_count']} | batches: {s['batch_count']} | "
             f"packages: {s['package_count']}"]
    if reg["identity_anchors"]:
        lines.append("  identity anchors:")
        for l in reg["identity_anchors"]:
            lo, hi = l["recommended_weight_range"]
            lines.append(f"    {l['name']:<30} weight {lo:.2f}-{hi:.2f} "
                         f"[{l['metadata_status']}]")
    if reg["concept_loras"]:
        lines.append("  concept LoRAs:")
        for l in reg["concept_loras"][:12]:
            lines.append(f"    {l['name']:<30} {l['family']:<12} "
                         f"risk {l['identity_risk']}")
    for w in reg["warnings"]:
        lines.append(f"  WARNING: {w}")
    for r in reg["recommendations"]:
        lines.append(f"  RECOMMEND: {r}")
    return "\n".join(lines)
