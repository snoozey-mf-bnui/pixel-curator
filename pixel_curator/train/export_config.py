"""Phase 10: Unified export + automatic training config generation.

Composes the intelligence that already exists - the Phase 4 parameter
advisor (dataset-aware params with per-key reasons), the Phase 8
experiment registry (past outcomes) and the Kohya config serializer -
into a ready-to-train bundle written beside every dataset export:

  training_config.toml   kohya sd-scripts config, reasons as comments
  train.sh               the exact accelerate launch command
  training_config.json   machine-readable bundle: params + why,
                         signals used, experiment hints, warnings

Everything here is a *suggestion layer*: generated files are helpful
starting points, never authoritative, and generation failures must
never break an export (integration points call through
:func:`generate_for_export`, which converts failures to warnings).
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path

from ..util import now_iso
from .kohya_config_builder import render_toml
from .param_advisor import _MIN_EPOCHS, _band_for, recommend_training_params
from .presets import PRESETS

log = logging.getLogger(__name__)

TOML_NAME = "training_config.toml"
SH_NAME = "train.sh"
JSON_NAME = "training_config.json"

# trainer entry points per model type (sd-scripts)
_TRAIN_SCRIPT = {"lora": "train_network.py",
                 "dreambooth": "sdxl_train.py",
                 "checkpoint": "sdxl_train.py"}


# ---------------------------------------------------------------------------
# parameter sources
# ---------------------------------------------------------------------------

def _fallback_params(frame_count: int, preset: str,
                     presets: dict | None = None) -> dict:
    """Count-based recommendation for unversioned folder exports.

    Same steps-comfort-band math as the Phase 4 advisor, driven only by
    the frame count - used when a selection/export has no built dataset
    version to analyze.
    """
    presets = presets or PRESETS
    base = dict(presets.get(preset) or presets.get("character") or {})
    n = max(1, int(frame_count))
    batch = int(base.get("batch_size") or 1)
    epochs = int(base.get("epochs") or 10)
    lo, hi = _band_for(preset)
    mid = (lo + hi) // 2
    repeats = max(1, min(40, round(mid * batch / max(1, n * epochs))))
    total = math.ceil(n * repeats / batch) * epochs
    warnings: list[str] = []
    if total > hi and repeats == 1:
        epochs = max(_MIN_EPOCHS, math.floor(hi * batch / n))
        total = math.ceil(n * repeats / batch) * epochs
        warnings.append(
            f"Large dataset ({n} frames): epochs trimmed to {epochs} to "
            f"stay near the ~{hi}-step band.")
    if total < lo:
        warnings.append(
            f"Even at {repeats} repeats this dataset reaches only ~{total} "
            f"steps (band {lo}-{hi}); expect a softer result or add data.")
    params = {
        "network_dim": {"value": base.get("network_dim", 32),
                        "why": f"preset '{preset}' structure"},
        "network_alpha": {"value": base.get("network_alpha", 16),
                          "why": "preset-defined (alpha = dim/2)"},
        "batch_size": {"value": batch, "why": "preset-defined"},
        "repeats": {"value": repeats,
                    "why": f"{n} frames x {repeats} repeats x {epochs} "
                           f"epochs / batch {batch} ~= {total} steps "
                           f"(band {lo}-{hi})"},
        "epochs": {"value": epochs,
                   "why": "steps-band sizing on the exported frame count"},
        "unet_lr": {"value": float(base.get("unet_lr") or 1e-4),
                    "why": f"preset '{preset}' default"},
        "te_lr": {"value": float(base.get("te_lr") or 0.0),
                  "why": f"preset '{preset}' default"},
    }
    if n < 150:
        params["unet_lr"] = {
            "value": round(params["unet_lr"]["value"] * 0.8, 8),
            "why": f"small dataset ({n} frames): UNet LR lowered 20% "
                   "to reduce overfit"}
    return {
        "preset": preset,
        "params": params,
        "totals": {"train_frames": n,
                   "steps_per_epoch": math.ceil(n * repeats / batch),
                   "total_steps": total, "step_band": [lo, hi]},
        "signals_used": {"selected_frames": n,
                         "source": "export frame count only"},
        "warnings": warnings + [
            "No built dataset version to analyze - recommendation is "
            "based on the exported frame count only."],
    }


def _advise(conn, dataset_version: int | None, frame_count: int | None,
            preset: str | None) -> dict:
    """Advisor when a version exists; count-based fallback otherwise."""
    if dataset_version is not None and conn is not None:
        try:
            rec = recommend_training_params(conn, int(dataset_version),
                                            preset_name=preset)
            if not rec.get("error"):
                return rec
            # advisor declined (e.g. empty membership) - fall through
            fc = frame_count or 0
            if fc:
                fb = _fallback_params(fc, preset or rec.get("preset")
                                      or "character")
                fb["warnings"] = rec.get("warnings", []) + fb["warnings"]
                return fb
            return rec
        except ValueError as exc:
            if frame_count:
                fb = _fallback_params(frame_count, preset or "character")
                fb["warnings"].insert(0, str(exc))
                return fb
            raise
    if not frame_count:
        raise ValueError("need a dataset_version or a frame_count to "
                         "recommend training parameters")
    return _fallback_params(int(frame_count), preset or "character")


# ---------------------------------------------------------------------------
# experiment awareness (Phase 8)
# ---------------------------------------------------------------------------

def experiment_hints(conn, *, dataset_version: int | None = None,
                     frame_count: int | None = None,
                     limit: int = 4) -> list[dict]:
    """Past training outcomes relevant to this export.

    Exact dataset-version matches first, then experiments on datasets of
    similar size (0.7x - 1.3x). keeper / rating>=4 runs become 'mirror
    these params' hints; discarded / rating<=2 runs become cautions.
    """
    if conn is None:
        return []
    hints: list[dict] = []
    seen: set[int] = set()

    def _rows(sql, params):
        try:
            return conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            return []

    base_sql = ("SELECT e.exp_id, e.name, e.outcome, e.rating, "
                "e.params_json, d.image_count, e.dataset_version "
                "FROM experiments e "
                "LEFT JOIN datasets d ON d.version = e.dataset_version ")
    rows = []
    if dataset_version is not None:
        rows += [("same dataset version", r) for r in _rows(
            base_sql + "WHERE e.dataset_version = ? "
                       "ORDER BY e.exp_id DESC", (int(dataset_version),))]
    if frame_count:
        lo, hi = int(frame_count * 0.7), int(math.ceil(frame_count * 1.3))
        rows += [("similar dataset size", r) for r in _rows(
            base_sql + "WHERE d.image_count BETWEEN ? AND ? "
                       "AND (e.dataset_version IS NOT ? ) "
                       "ORDER BY e.rating DESC, e.exp_id DESC",
            (lo, hi, dataset_version))]

    for relation, r in rows:
        exp_id, name, outcome, rating, params_json, img_count, dsv = r
        if exp_id in seen:
            continue
        seen.add(exp_id)
        try:
            params = json.loads(params_json or "{}")
        except Exception:
            params = {}
        positive = (outcome == "keeper") or (rating or 0) >= 4
        negative = (outcome == "discarded") or (
            rating is not None and rating <= 2)
        if not positive and not negative:
            continue
        key_params = {k: v for k, v in params.items()
                      if k in ("preset", "epochs", "unet_lr", "te_lr",
                               "learning_rate", "repeats", "network_dim",
                               "final_loss", "max_train_epochs")}
        stars = f", rated {rating}/5" if rating else ""
        if positive:
            text = (f"experiment #{exp_id} '{name}' ({relation}"
                    f"{stars}) worked well"
                    + (f" with {json.dumps(key_params)}" if key_params
                       else "") + " - consider mirroring")
        else:
            text = (f"experiment #{exp_id} '{name}' ({relation}"
                    f"{stars}) was {outcome or 'rated low'}"
                    + (f" with {json.dumps(key_params)}" if key_params
                       else "") + " - avoid repeating those settings")
        hints.append({"exp_id": exp_id, "name": name, "relation": relation,
                      "outcome": outcome, "rating": rating,
                      "sentiment": "mirror" if positive else "avoid",
                      "params": key_params, "text": text})
        if len(hints) >= limit:
            break
    return hints


# ---------------------------------------------------------------------------
# config composition
# ---------------------------------------------------------------------------

def _plain_params(rec: dict) -> dict:
    return {k: v["value"] for k, v in (rec.get("params") or {}).items()}


def compose_kohya_config(rec: dict, *, export_dir: Path,
                         model_type: str = "lora",
                         base_model: str | None = None,
                         output_name: str | None = None,
                         overrides: dict | None = None) -> tuple[dict, dict]:
    """Merge advisor values into a full kohya config.

    Returns ``(config, reasons)`` where reasons maps config keys to the
    advisor's 'why' strings (plus 'user override' for overrides).
    """
    p = _plain_params(rec)
    reasons: dict[str, str] = {}

    def why(advisor_key):
        d = (rec.get("params") or {}).get(advisor_key)
        return d["why"] if d else ""

    cfg: dict = {
        "pretrained_model_name_or_path": base_model or "BASE_MODEL.safetensors",
        "train_data_dir": str(export_dir),
        "output_dir": str(Path(export_dir) / "output"),
        "output_name": output_name or f"{Path(export_dir).name}_{model_type}",
        "resolution": "1024,1024",
        "train_batch_size": int(p.get("batch_size", 1)),
        "gradient_accumulation_steps": 2,
        "max_train_epochs": int(p.get("epochs", 10)),
        "save_every_n_epochs": 1,
        "lr_scheduler": "cosine",
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "cache_latents": True,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 1,
    }
    reasons["train_batch_size"] = why("batch_size")
    reasons["max_train_epochs"] = why("epochs")
    reasons["save_every_n_epochs"] = ("every-epoch checkpoints enable the "
                                      "best-epoch sweep")
    if model_type == "lora":
        cfg.update({
            "network_module": "networks.lora",
            "network_dim": int(p.get("network_dim", 32)),
            "network_alpha": int(p.get("network_alpha", 16)),
            "learning_rate": float(p.get("unet_lr", 1e-4)),
            "unet_lr": float(p.get("unet_lr", 1e-4)),
            "text_encoder_lr": float(p.get("te_lr", 0.0)),
            "min_snr_gamma": 5,
        })
        reasons["network_dim"] = why("network_dim")
        reasons["network_alpha"] = why("network_alpha")
        reasons["unet_lr"] = why("unet_lr")
        reasons["text_encoder_lr"] = why("te_lr")
    else:
        # full finetune (DreamBooth / checkpoint): LoRA-scale LRs would
        # fry the model - use the conservative full-finetune scale.
        cfg.update({
            "learning_rate": 1e-6,
            "gradient_accumulation_steps": 8,
            "max_train_epochs": min(int(p.get("epochs", 6)), 6),
            "mixed_precision": "bf16",
            "save_precision": "fp16",
            "min_snr_gamma": 5,
        })
        reasons["learning_rate"] = ("full-finetune scale (1e-6); LoRA-scale "
                                    "LRs overfit a checkpoint in minutes")
        reasons["gradient_accumulation_steps"] = ("effective batch 8 smooths "
                                                  "batch-1 SDXL gradients")
        reasons["max_train_epochs"] = ("full finetunes peak early; best "
                                       "epoch is usually 2-4")

    for k, v in (overrides or {}).items():
        cfg[k] = v
        reasons[k] = "user override"
    return cfg, reasons


def render_config_toml(cfg: dict, reasons: dict, rec: dict,
                       hints: list[dict], model_type: str) -> str:
    """Kohya TOML with the transparency header baked in as comments."""
    lines = [
        "# training_config.toml - generated by pixel-curator export "
        f"({now_iso()})",
        "# A suggested starting point, not an authoritative config: "
        "override freely.",
        f"# model type: {model_type} | preset basis: "
        f"{rec.get('preset', '-')}",
    ]
    totals = rec.get("totals") or {}
    if totals:
        lines.append(
            f"# sizing: {totals.get('train_frames')} frames -> "
            f"~{totals.get('total_steps')} steps "
            f"(band {'-'.join(str(x) for x in totals.get('step_band', []))})")
    rp = (rec.get("params") or {}).get("repeats")
    if rp:
        lines.append(f"# repeats: {rp['value']} - {rp['why']}")
        lines.append("#          (kohya reads repeats from the image folder "
                     "prefix, e.g. rename to "
                     f"'{rp['value']}_subject'; not a TOML key)")
    why_lines = [(k, reasons.get(k, "")) for k in cfg
                 if reasons.get(k)]
    if why_lines:
        lines.append("#")
        lines.append("# why these values:")
        for k, w in why_lines:
            lines.append(f"#   {k}: {w}")
    for w in rec.get("warnings") or []:
        lines.append(f"# WARNING: {w}")
    for h in hints:
        lines.append(f"# EXPERIENCE: {h['text']}")
    lines.append("")
    return "\n".join(lines) + render_toml(cfg)


def render_train_sh(config_path: Path, model_type: str) -> str:
    script = _TRAIN_SCRIPT.get(model_type, "train_network.py")
    return ("#!/usr/bin/env bash\nset -euo pipefail\n"
            "# generated by pixel-curator export - suggested launch command\n"
            f"accelerate launch {script} --config_file {config_path}\n")


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------

def generate_training_bundle(conn, export_dir: Path | str, *,
                             dataset_version: int | None = None,
                             frame_count: int | None = None,
                             model_type: str = "lora",
                             preset: str | None = None,
                             base_model: str | None = None,
                             output_name: str | None = None,
                             overrides: dict | None = None,
                             write: bool = True) -> dict:
    """Generate the ready-to-train bundle for an exported dataset.

    Returns a dict with the composed config, per-key reasons, experiment
    hints, warnings and (when ``write``) the paths of the three files
    written beside the dataset.
    """
    export_dir = Path(export_dir).expanduser()
    rec = _advise(conn, dataset_version, frame_count, preset)
    fc = (rec.get("totals") or {}).get("train_frames") or frame_count
    hints = experiment_hints(conn, dataset_version=dataset_version,
                             frame_count=fc)
    cfg, reasons = compose_kohya_config(
        rec, export_dir=export_dir, model_type=model_type,
        base_model=base_model, output_name=output_name, overrides=overrides)
    toml_text = render_config_toml(cfg, reasons, rec, hints, model_type)
    config_path = export_dir / TOML_NAME
    sh_text = render_train_sh(config_path, model_type)

    bundle = {
        "generated_at": now_iso(),
        "model_type": model_type,
        "dataset_version": dataset_version,
        "preset": rec.get("preset"),
        "config": cfg,
        "reasons": reasons,
        "advisor": {
            "params": rec.get("params"),
            "totals": rec.get("totals"),
            "estimates": rec.get("estimates"),
            "signals_used": rec.get("signals_used"),
        },
        "experiment_hints": hints,
        "warnings": list(rec.get("warnings") or []),
        "override_hint": ("Edit training_config.toml directly, or "
                          "regenerate with: pixel-curator training-config "
                          + (f"--dataset {dataset_version} "
                             if dataset_version is not None else "")
                          + "--set key=value"),
    }
    result = {"bundle": bundle, "toml_text": toml_text,
              "command": sh_text.strip().splitlines()[-1],
              "warnings": bundle["warnings"]}
    if write:
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / TOML_NAME).write_text(toml_text, encoding="utf-8")
        sh_path = export_dir / SH_NAME
        sh_path.write_text(sh_text, encoding="utf-8")
        try:
            sh_path.chmod(0o755)
        except OSError:                              # pragma: no cover
            pass
        (export_dir / JSON_NAME).write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False),
            encoding="utf-8")
        result["paths"] = {"toml": str(export_dir / TOML_NAME),
                           "sh": str(sh_path),
                           "json": str(export_dir / JSON_NAME)}
    return result


def generate_for_export(conn, export_dir, **kw) -> tuple[dict | None,
                                                         list[str]]:
    """Failure-proof wrapper for export integration points.

    Returns ``(result_or_None, messages)`` - never raises, so dataset
    exports cannot fail because of config generation.
    """
    try:
        result = generate_training_bundle(conn, export_dir, **kw)
        msgs = [f"Training config written: {result['paths']['toml']}",
                f"  launch: {result['command']}"]
        for h in result["bundle"]["experiment_hints"]:
            msgs.append(f"  experience: {h['text']}")
        return result, msgs
    except Exception as exc:
        log.warning("training config generation skipped: %s", exc)
        return None, [f"Training config generation skipped: {exc}"]


__all__ = ["generate_training_bundle", "generate_for_export",
           "experiment_hints", "compose_kohya_config",
           "render_config_toml", "TOML_NAME", "SH_NAME", "JSON_NAME"]
