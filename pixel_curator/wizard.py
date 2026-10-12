"""Creator Wizard: raw curated library -> typed, trained, tested LoRA.

Replicates the Civitai creator flow, fully offline:
  1. analyze   - dataset statistics from the manifest (no re-scanning)
  2. detect    - score all 6 LoRA types with explainable heuristics
  3. create    - inject a type-tuned recipe, then build -> train -> matrix
                 -> model card as one queued chain

Type heuristics (all from data already in the manifest):
  character - high face presence + tight identity consistency + framing spread
  style     - low face dependence, high scene/cluster diversity
  outfit    - clusters concentrated (few dominant outfit clusters), body framings
  pose      - full-body dominance
  detail    - closeup dominance + high sharpness
  explicit  - explicit indicator tags present in captions (adult self-content);
              trains with character parameters + unfiltered caption policy
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

from . import manifest
from .base_models import PROFILES, blocks_string, detect_profile
from .config import Project
from .util import now_iso, safe_slug, setup_logging

EXPLICIT_TAGS = {"nude", "nsfw", "explicit", "completely_nude", "topless", "bottomless"}

# type -> (recipe template, training preset)
TYPE_TEMPLATES: dict[str, dict] = {
    "character": {
        "preset": "character", "repeats": 10, "max_total": 400, "max_per_video": 40,
        "quota": "closeup=0.30,portrait=0.30,upper_body=0.25,full_body=0.15",
        "val_fraction": 0.05,
    },
    "style": {
        "preset": "style", "repeats": 4, "max_total": 600, "max_per_video": 20,
        "quota": "", "val_fraction": 0.05,
    },
    "outfit": {
        "preset": "outfit", "repeats": 8, "max_total": 150,
        "framing": "upper_body,full_body", "val_fraction": 0.05,
    },
    "pose": {
        "preset": "pose", "repeats": 6, "max_total": 200,
        "framing": "full_body", "quota": "", "val_fraction": 0.05,
    },
    "detail": {
        "preset": "detail", "repeats": 12, "max_total": 120,
        "framing": "closeup", "min_sharpness": 80, "smart_crop": True,
        "val_fraction": 0.0,
    },
    "explicit": {
        "preset": "character", "repeats": 10, "max_total": 400, "max_per_video": 40,
        "quota": "closeup=0.25,portrait=0.25,upper_body=0.25,full_body=0.25",
        "val_fraction": 0.05,
    },
}


@dataclass
class WizardConfig:
    output_base: Path
    lora_type: str = "auto"
    trigger: str = ""              # defaults to project trigger
    class_word: str = ""
    name: str = ""                 # output name; auto if blank
    train: bool = True
    matrix: bool = True
    matrix_backend: str = "forge"
    forge_url: str = "http://127.0.0.1:7860"


# -----------------------------
# 1) Analyze
# -----------------------------

def analyze(conn) -> dict:
    a: dict = {}
    a["selected"] = conn.execute(
        "SELECT COUNT(*) FROM frames WHERE status IN ('selected','packaged')"
    ).fetchone()[0]
    a["captioned"] = conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0]

    # face / identity
    rows = conn.execute(
        "SELECT d.face_count, d.identity_sim, d.framing FROM detections d "
        "JOIN frames f ON f.frame_id = d.frame_id "
        "WHERE f.status IN ('selected','packaged')"
    ).fetchall()
    a["scanned"] = len(rows)
    faces = [r for r in rows if (r[0] or 0) > 0]
    a["face_rate"] = len(faces) / len(rows) if rows else 0.0
    sims = [r[1] for r in rows if r[1] is not None]
    a["id_mean"] = sum(sims) / len(sims) if sims else None
    a["id_std"] = (math.sqrt(sum((s - a["id_mean"]) ** 2 for s in sims) / len(sims))
                   if sims else None)

    # framing mix
    mix: dict[str, int] = {}
    for r in rows:
        mix[r[2] or "none"] = mix.get(r[2] or "none", 0) + 1
    total = max(1, sum(mix.values()))
    a["framing_mix"] = {k: round(v / total, 3) for k, v in sorted(mix.items())}

    # cluster concentration (normalized entropy; low = concentrated)
    crows = conn.execute(
        "SELECT cluster_id, COUNT(*) FROM frames "
        "WHERE status IN ('selected','packaged') AND cluster_id IS NOT NULL "
        "GROUP BY cluster_id"
    ).fetchall()
    a["clusters"] = len(crows)
    if len(crows) > 1:
        n = sum(c for _, c in crows)
        ent = -sum((c / n) * math.log(c / n) for _, c in crows)
        a["cluster_entropy"] = round(ent / math.log(len(crows)), 3)
    else:
        a["cluster_entropy"] = None

    # caption tag stats
    expl = 0
    tagged = 0
    for (tj,) in conn.execute("SELECT tags_json FROM captions WHERE tags_json IS NOT NULL"):
        tagged += 1
        try:
            names = {t[0].replace(" ", "_").lower() for t in json.loads(tj)}
            if names & EXPLICIT_TAGS:
                expl += 1
        except Exception:
            pass
    a["explicit_rate"] = round(expl / tagged, 3) if tagged else 0.0

    row = conn.execute(
        "SELECT AVG(sharpness) FROM frames WHERE status IN ('selected','packaged')"
    ).fetchone()
    a["sharpness_mean"] = round(row[0], 1) if row and row[0] else None
    return a


# -----------------------------
# 2) Detect
# -----------------------------

def detect_type(a: dict) -> list[dict]:
    """Returns [{type, score (0-1), reason}, ...] best-first."""
    fm = a.get("framing_mix", {})
    face = a.get("face_rate", 0.0)
    id_std = a.get("id_std")
    # NOTE: id_std == 0.0 means PERFECT consistency, not missing data -
    # `or` must not be used here (py3.12's exact sum() yields true 0.0).
    consistency = max(0.0, 1.0 - id_std * 4) if id_std is not None else 0.3
    entropy = a.get("cluster_entropy")
    spread = entropy if entropy is not None else 0.5
    closeup = fm.get("closeup", 0) + fm.get("portrait", 0)
    fullb = fm.get("full_body", 0)
    body = fm.get("upper_body", 0) + fullb
    expl = a.get("explicit_rate", 0.0)
    sharp = a.get("sharpness_mean") or 0

    scores = {
        "character": 0.45 * face + 0.35 * consistency
        + 0.20 * (1 - abs(closeup - 0.55)),
        "style": 0.45 * (1 - face) + 0.35 * spread + 0.20 * (1 - consistency),
        "outfit": 0.45 * body + 0.35 * (1 - spread) + 0.20 * face,
        "pose": 0.60 * fullb + 0.25 * face + 0.15 * spread,
        "detail": 0.50 * fm.get("closeup", 0) + 0.30 * min(1.0, sharp / 150)
        + 0.20 * face,
        "explicit": min(1.0, expl * 1.6) * (0.6 + 0.4 * face),
    }
    reasons = {
        "character": f"face rate {face:.0%}, identity consistency {consistency:.2f}",
        "style": f"scene diversity {spread:.2f}, face rate {face:.0%}",
        "outfit": f"body framings {body:.0%}, cluster concentration {1-spread:.2f}",
        "pose": f"full-body share {fullb:.0%}",
        "detail": f"closeup share {fm.get('closeup', 0):.0%}, sharpness {sharp:.0f}",
        "explicit": f"explicit-tag rate {expl:.0%}",
    }
    # Identity-priority guard: with strong face presence and stable
    # identity consistency, character detection must not be displaced by
    # style - scene diversity alone is not evidence of a style dataset
    # when the subject is clearly present in most frames.  Genuinely
    # low-face datasets are unaffected (guard requires face >= 0.70).
    if (face >= 0.70 and consistency >= 0.55
            and scores["style"] >= scores["character"]):
        scores["character"] = min(1.0, scores["style"] + 0.05)
        reasons["character"] += (" (identity-priority: strong face rate "
                                 "and identity consistency)")

    out = [{"type": t, "score": round(min(1.0, max(0.0, s)), 3),
            "reason": reasons[t]} for t, s in scores.items()]
    return sorted(out, key=lambda d: -d["score"])


def suggest_stack(ranking: list[dict], analysis: dict,
                  base_model: Optional[str] = None) -> dict:
    """From type scores -> a recommended multi-LoRA production stack,
    block-weighted for the project's base model profile.

    Returns {"stack": [{type, role, weight, blocks, blocks_cli}],
             "rationale": [...], "profile": <label>, "merge_name": ...}.
    The primary LoRA is the top-scoring type; complementary LoRAs are added
    when their signals are strong enough to be worth a separate training run.
    """
    prof = detect_profile(base_model) if base_model else PROFILES["pony_v6"]
    bw, ww = prof["blocks"], prof["weights"]

    def entry(lora_type: str, role: str) -> dict:
        blocks = dict(bw[role])
        return {"type": lora_type, "role": role, "weight": ww[role],
                "blocks": blocks, "blocks_cli": blocks_string(blocks)}

    scores = {r["type"]: r["score"] for r in ranking}
    fm = analysis.get("framing_mix", {})
    stack: list[dict] = []
    why: list[str] = []

    primary = ranking[0]["type"]
    stack.append(entry(primary, "primary"))
    why.append(f"primary = {primary} ({ranking[0]['reason']})")
    why.append(f"block weights tuned for {prof['label']}")

    if primary in ("character", "explicit"):
        if scores.get("style", 0) >= 0.45:
            stack.append(entry("style", "flavor"))
            why.append("style signal strong enough for a surface-only flavor "
                       "LoRA (low down/mid keeps the base's rendering intact)")
        if scores.get("outfit", 0) >= 0.50 and analysis.get("clusters", 0) >= 3:
            stack.append(entry("outfit", "wardrobe"))
            why.append("distinct outfit clusters detected - separate wardrobe LoRA")
        if fm.get("closeup", 0) >= 0.30 and (analysis.get("sharpness_mean") or 0) >= 60:
            stack.append(entry("detail", "refiner"))
            why.append("enough sharp closeups for a detail refiner")
    elif primary == "style" and scores.get("character", 0) >= 0.5:
        stack.append(entry("character", "identity"))
        why.append("identity signal present - pair the style with a character LoRA")

    return {"stack": stack, "rationale": why, "profile": prof["label"],
            "merge_name": f"{primary}_pack_v1"}


def make_recipe(lora_type: str, trigger: str, class_word: str) -> dict:
    tpl = dict(TYPE_TEMPLATES[lora_type])
    tpl.pop("preset", None)
    tpl["type"] = lora_type
    tpl["token"] = trigger
    tpl["class_word"] = class_word
    return tpl


# -----------------------------
# 3) Create (one-click chain)
# -----------------------------

def wizard_generator(prj: Project, cfg: WizardConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)

    a = analyze(conn)
    if a["selected"] == 0:
        yield ("FATAL: no curated frames. Run the pipeline through curation first "
               "(pixel-curator pipeline run full --gate caption).")
        return
    ranking = detect_type(a)
    yield ("DATASET ANALYSIS\n"
           f"  selected {a['selected']} | scanned {a['scanned']} | captioned {a['captioned']}\n"
           f"  face rate {a['face_rate']:.0%} | id mean/std "
           f"{a['id_mean']}/{a['id_std'] if a['id_std'] is None else round(a['id_std'],3)}\n"
           f"  framing {a['framing_mix']} | clusters {a['clusters']} "
           f"(entropy {a['cluster_entropy']})\n"
           f"  explicit-tag rate {a['explicit_rate']:.0%}\n")

    lora_type = cfg.lora_type
    if lora_type == "auto":
        lora_type = ranking[0]["type"]
        yield ("TYPE DETECTION\n" + "\n".join(
            f"  {r['type']:<10} {r['score']:.2f}  ({r['reason']})" for r in ranking
        ) + f"\n-> auto-selected: {lora_type}\n")
    if lora_type not in TYPE_TEMPLATES:
        yield f"FATAL: unknown type '{lora_type}'"
        return

    trigger = cfg.trigger or prj.trigger_token
    class_word = cfg.class_word or prj.class_word
    preset = TYPE_TEMPLATES[lora_type]["preset"]
    recipe_name = f"wizard_{lora_type}"
    prj.recipes[recipe_name] = make_recipe(lora_type, trigger, class_word)
    name = cfg.name or f"{safe_slug(trigger)}_{lora_type}_{now_iso()[:10].replace('-','')}"

    yield (f"PLAN: build[{recipe_name}] -> "
           f"{'train[' + preset + '] -> ' if cfg.train else ''}"
           f"{'matrix -> card' if cfg.matrix and cfg.train else 'done'}\n")

    # build
    from .builder import BuildConfig, build_generator
    for u in build_generator(prj, BuildConfig(
        output_base=cfg.output_base, recipe=recipe_name,
        note=f"wizard {lora_type}",
    )):
        yield f"[build] {u}"
    row = conn.execute("SELECT MAX(version) FROM datasets").fetchone()
    version = int(row[0]) if row and row[0] else 0
    if not version:
        yield "FATAL: build produced no dataset."
        return

    if not cfg.train:
        yield f"\nWIZARD DONE (build-only). Dataset v{version:03d} ready for training."
        return

    # train
    from .train.kohya import TrainConfig, train_generator
    for u in train_generator(prj, TrainConfig(
        output_base=cfg.output_base, dataset_version=version,
        preset=preset, name=name,
    )):
        yield f"[train] {u}"
    status = conn.execute(
        "SELECT status FROM runs WHERE name=? ORDER BY run_id DESC LIMIT 1", (name,)
    ).fetchone()
    if not status or status[0] != "completed":
        yield "\nWIZARD PAUSED: training did not complete; see Train tab / runs."
        return

    # matrix + card
    if cfg.matrix:
        lora_dir = Path(prj.lora_output_dir
                        or (Path(prj.output_base) / "LORA_OUTPUT")).expanduser()
        candidates = sorted(lora_dir.glob(f"{name}*.safetensors"))
        lora_file = str(candidates[-1]) if candidates else name
        from .eval.matrix import MatrixConfig, matrix_generator
        for u in matrix_generator(prj, MatrixConfig(
            output_base=cfg.output_base, lora=lora_file, label=name,
            backend=cfg.matrix_backend, forge_url=cfg.forge_url,
            checkpoint=prj.base_model,
            categories=["likeness", "flexibility"],
        )):
            yield f"[matrix] {u}"

    from .registry import model_card
    card = model_card(prj, conn, name)
    yield (f"\nWIZARD COMPLETE: {name} ({lora_type})\n"
           f"Dataset: v{version:03d} | Preset: {preset}\n"
           f"Model card: {card}\n"
           f"Use in Forge: <lora:{name}:0.85> with trigger '{trigger} {class_word}'")
