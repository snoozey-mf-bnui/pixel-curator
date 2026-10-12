"""Phase 4: Smart Training Parameter Advisor - dataset-aware, explainable.

Where preset_recommender picks *which* preset, this module recommends the
*numbers*: repeats, epochs, total steps, learning rates, batch size and a
wall-clock estimate, sized to the actual curated dataset (frame count,
identity consistency, quality score, diversity). Every parameter carries a
one-line rationale, and the output ends with override hints - suggestions,
never mandates.

Tolerant by design: works on datasets that haven't run every quality pass
(missing signals simply skip their adjustment and add a note).
"""

from __future__ import annotations

import math
from typing import Optional

from .presets import PRESETS

# total-steps comfort bands per preset family: (low, high).
# Under-shooting the low end under-trains; far above the high end mostly
# buys overfit risk and wall-clock.
_STEP_BANDS: dict[str, tuple[int, int]] = {
    "character": (2000, 2800),
    "balanced": (2000, 2800),
    "style": (2400, 3400),
    "outfit": (1800, 2600),
    "fashion": (1800, 2600),
    "lingerie": (1800, 2600),
    "pose": (1400, 2000),
    "detail": (1200, 1800),
    "figure": (2000, 2800),
    "intimate": (2000, 2800),
    "anatomy": (1600, 2400),
    "explicit": (1400, 2000),
}
_DEFAULT_BAND = (1800, 2600)

# rough wall-clock rates (seconds per optimizer step, batch-adjusted below)
_SEC_PER_STEP = {"local_mps_m2max": 6.0, "colab_a100": 1.3}

_MIN_EPOCHS = 4     # below this the best-epoch sweep has nothing to compare


def _band_for(preset_name: str) -> tuple[int, int]:
    name = (preset_name or "").lower()
    for key, band in _STEP_BANDS.items():
        if key in name:
            return band
    return _DEFAULT_BAND


def _fmt_duration(seconds: float) -> str:
    if seconds < 3600:
        return f"~{max(1, round(seconds / 60))} min"
    return f"~{seconds / 3600:.1f} h"


def recommend_training_params(conn, dataset_version: int,
                              preset_name: Optional[str] = None,
                              recipe_name: Optional[str] = None,
                              presets: Optional[dict] = None) -> dict:
    """Dataset-aware parameter recommendation for a built dataset version.

    When ``preset_name`` is omitted the preset recommender picks one first.
    Raises ValueError only when the dataset version doesn't exist (same
    contract as suggest_preset).
    """
    presets = presets if presets is not None else PRESETS
    from .preset_recommender import collect_signals, suggest_preset

    warnings: list[str] = []
    preset_confidence = None
    if preset_name and preset_name in presets:
        signals = collect_signals(conn, dataset_version)
    else:
        if preset_name:
            warnings.append(f"Preset '{preset_name}' not found - "
                            "falling back to the recommender.")
        rec = suggest_preset(conn, dataset_version,
                             recipe_name=recipe_name, presets=presets)
        preset_name = rec["recommended_preset"]
        preset_confidence = rec["confidence"]
        signals = rec["signals"]
        warnings.extend(rec.get("warnings", []))
        if preset_name is None:
            return {"preset": None, "params": {}, "warnings": warnings,
                    "error": "no training presets available"}

    base = dict(presets.get(preset_name) or {})
    n = int(signals.get("selected_frames") or 0)
    identity = signals.get("identity_strength")
    quality = signals.get("quality_score")
    clusters = int(signals.get("cluster_count") or 0)

    params: dict[str, dict] = {}

    def put(key, value, why):
        params[key] = {"value": value, "why": why}

    if n <= 0:
        return {"preset": preset_name, "preset_confidence": preset_confidence,
                "params": {}, "signals_used": signals,
                "warnings": warnings + [
                    "Dataset has no train frames - build it before asking "
                    "for parameter recommendations."],
                "error": "empty dataset"}

    # --- static preset structure (dim/alpha stay preset-defined) -----------
    batch = int(base.get("batch_size") or 1)
    epochs = int(base.get("epochs") or 10)
    unet_lr = float(base.get("unet_lr") or 1e-4)
    te_lr = float(base.get("te_lr") or 0.0)
    put("network_dim", base.get("network_dim", 32),
        f"preset '{preset_name}' structure (capacity matches LoRA type)")
    put("network_alpha", base.get("network_alpha", 16),
        "preset-defined (alpha = dim/2 convention)")
    put("batch_size", batch,
        "preset-defined for the local memory budget")

    # --- repeats/epochs sized to the steps comfort band --------------------
    lo, hi = _band_for(preset_name)
    mid = (lo + hi) // 2
    repeats = max(1, min(40, round(mid * batch / max(1, n * epochs))))
    total = math.ceil(n * repeats / batch) * epochs
    if total > hi and repeats == 1:
        # large dataset: trim epochs into the band instead of inflating time
        epochs = max(_MIN_EPOCHS, math.floor(hi * batch / n))
        total = math.ceil(n * repeats / batch) * epochs
        warnings.append(
            f"Large dataset ({n} frames): epochs trimmed to {epochs} to "
            f"stay near the ~{hi}-step band - more epochs would mostly add "
            "wall-clock and overfit risk.")
    if total < lo:
        # tiny even at max repeats: say so instead of silently under-training
        warnings.append(
            f"Even at {repeats} repeats this dataset reaches only ~{total} "
            f"steps (band {lo}-{hi}); expect a softer LoRA or add data.")
    put("repeats", repeats,
        f"{n} frames x {repeats} repeats x {epochs} epochs / batch {batch} "
        f"~= {total} steps (target band {lo}-{hi} for '{preset_name}')")
    put("epochs", epochs,
        f"keeps every-epoch checkpoints meaningful for the best-epoch "
        f"sweep (minimum {_MIN_EPOCHS})")

    # --- dataset-aware LR adjustments ---------------------------------------
    lr_notes = []
    if n < 150:
        unet_lr = round(unet_lr * 0.8, 8)
        lr_notes.append(f"small dataset ({n} frames): UNet LR lowered 20% "
                        "to reduce overfit")
    if identity is not None and identity < 0.45 and te_lr > 0:
        te_lr = min(te_lr, 1e-5)
        lr_notes.append(f"identity consistency {identity:.2f} is weak: "
                        "TE LR capped at 1e-5 to protect the identity token")
    elif identity is None:
        warnings.append("No identity similarity data - run smart curation "
                        "for identity-aware LR adjustments.")
    put("unet_lr", unet_lr,
        "; ".join(lr_notes) or f"preset '{preset_name}' default")
    put("te_lr", te_lr,
        ("frozen text encoder (preset)" if te_lr == 0 else
         ("; ".join(x for x in lr_notes if "TE LR" in x)
          or f"preset '{preset_name}' default")))

    # --- quality / diversity guidance ---------------------------------------
    if quality is not None and quality < 0.45:
        warnings.append(f"Quality score {quality:.2f} is low - another "
                        "curation pass will likely help more than any "
                        "parameter tuning.")
    if clusters and clusters < 5 and n >= 300:
        warnings.append(f"Only {clusters} visual clusters for {n} frames - "
                        "low diversity; consider broadening selection or "
                        "lowering repeats.")

    # --- lint cross-link (Phase 4 Part A) ------------------------------------
    try:
        from ..lint import (detect_caption_contradictions,
                            detect_caption_duplicate_tags,
                            detect_dataset_count_drift)
        n_fixable = sum(len(d(conn)) for d in (
            detect_caption_contradictions, detect_caption_duplicate_tags,
            detect_dataset_count_drift))
        if n_fixable:
            warnings.append(f"{n_fixable} auto-fixable lint issue(s) found - "
                            "run `insights lint --fix` before training.")
    except Exception:
        pass

    # --- totals + wall-clock estimates ---------------------------------------
    steps_per_epoch = math.ceil(n * repeats / batch)
    total_steps = steps_per_epoch * epochs
    estimates = {
        backend: _fmt_duration(total_steps * sec)
        for backend, sec in _SEC_PER_STEP.items()
    }

    return {
        "preset": preset_name,
        "preset_confidence": preset_confidence,
        "params": params,
        "totals": {"train_frames": n, "steps_per_epoch": steps_per_epoch,
                   "total_steps": total_steps,
                   "step_band": [lo, hi]},
        "estimates": estimates,
        "estimate_note": ("rough SDXL LoRA rates: "
                          f"{_SEC_PER_STEP['local_mps_m2max']:.0f}s/step "
                          "local MPS (M2 Max), "
                          f"{_SEC_PER_STEP['colab_a100']:.1f}s/step "
                          "Colab A100; actual speed varies with resolution "
                          "and buckets"),
        "signals_used": {k: signals.get(k) for k in (
            "selected_frames", "identity_strength", "quality_score",
            "cluster_count", "caption_coverage", "sharpness_mean")},
        "warnings": warnings,
        "override_hint": ("These are suggestions - override any value at "
                          "launch:  train-legacy --dataset "
                          f"{dataset_version} --preset {preset_name} "
                          "--set epochs=12 --set unet_lr=8e-5"),
    }


def render_params(rec: dict) -> str:
    if rec.get("error"):
        lines = [f"Parameter advisor: {rec['error']}"]
        lines += [f"  ! {w}" for w in rec.get("warnings", [])]
        return "\n".join(lines)
    t = rec["totals"]
    lines = ["", "Suggested training parameters "
             f"(preset '{rec['preset']}'"
             + (f", confidence {rec['preset_confidence']:.2f})"
                if rec.get("preset_confidence") is not None else ")"),
             "-" * 60]
    for key, p in rec["params"].items():
        lines.append(f"  {key:<14} {p['value']!s:<10} {p['why']}")
    lines.append("-" * 60)
    lines.append(f"  steps: {t['steps_per_epoch']}/epoch -> "
                 f"{t['total_steps']} total "
                 f"(band {t['step_band'][0]}-{t['step_band'][1]})")
    est = " | ".join(f"{k}: {v}" for k, v in rec["estimates"].items())
    lines.append(f"  estimated wall-clock: {est}")
    lines.append(f"  ({rec['estimate_note']})")
    for w in rec.get("warnings", []):
        lines.append(f"  ! {w}")
    lines.append(f"\n{rec['override_hint']}")
    return "\n".join(lines)


__all__ = ["recommend_training_params", "render_params"]
