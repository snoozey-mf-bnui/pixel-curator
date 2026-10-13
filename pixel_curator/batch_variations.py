"""Controlled variation generation - batch grids over attribute changes.

Expands a variation grid spec (slider sweeps, seed sets, stack compares)
into concrete, manifest-tracked variation jobs with fully resolved LoRA
stacks and generation payloads tuned for the project's base-model profile.

Generation itself goes through the existing adapters (ForgeClient payloads
are produced directly; the diffusers path receives the same resolved
parameters).  If no backend is live, the batch manifest + payloads are
still written - a clean adapter seam.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field

from .concept_control import SLIDER_INDEX, ConceptSliderState, resolve_controlled_stack
from .config import Project
from .lora_explorer import LoraCard
from .util import now_iso

# ---------------------------------------------------------------------------
# Smart variation modes - conservative, professional defaults
# ---------------------------------------------------------------------------
# value_ceiling clips axis slider values; cap bounds the job count.
VARIATION_MODES: dict[str, dict] = {
    "low_risk": {
        "label": "Low-Risk Studio Sweep", "cap": 24, "value_ceiling": 0.60,
        "notes": "Small changes, strongest identity preservation, strict "
                 "normalization. Recommended for production character "
                 "consistency.",
    },
    "balanced": {
        "label": "Balanced Exploration", "cap": 64, "value_ceiling": 0.85,
        "notes": "Moderate variation; identity still prioritized; conflict "
                 "warnings active. Recommended for testing concept "
                 "combinations.",
    },
    "creative": {
        "label": "Creative Exploration", "cap": 128, "value_ceiling": 1.0,
        "notes": "Wider variation range; still identity-aware; requires "
                 "explicit selection. Never weakens the identity anchor "
                 "below its safe minimum. Not for default production use.",
    },
}
DEFAULT_MODE = "low_risk"


@dataclass
class VariationAxis:
    """One named variation axis: explicit values OR min/max/step."""
    slider: str
    values: list[float] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None

    def resolve_values(self, mode: str = DEFAULT_MODE) -> list[float]:
        spec = SLIDER_INDEX.get(self.slider)
        if spec is None:
            return []
        ceil = VARIATION_MODES.get(mode, VARIATION_MODES[DEFAULT_MODE])[
            "value_ceiling"]
        # identity axis: protected - never swept below its safe band and
        # never clipped upward (raising identity is always allowed)
        identity_axis = self.slider == "identity_anchor_strength"
        vals = list(self.values)
        if not vals and self.minimum is not None and self.maximum is not None:
            step = self.step or spec.step or 0.1
            v, out = float(self.minimum), []
            while v <= float(self.maximum) + 1e-9 and len(out) < 50:
                out.append(round(v, 4))
                v += step
            vals = out
        clipped = []
        for v in vals:
            v = max(spec.minimum, min(spec.maximum, float(v)))
            if identity_axis:
                v = max(v, 0.5)        # anchor stays within its safe band
            else:
                v = min(v, ceil)
            clipped.append(round(v, 4))
        # deterministic + de-duplicated, order preserved
        seen, out = set(), []
        for v in clipped:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out


def estimate_job_count(axes: list[VariationAxis], seeds: list[int],
                       mode: str = DEFAULT_MODE) -> dict:
    """Planned grid size BEFORE creation - powers the UI preview."""
    n = 1
    for ax in axes:
        vals = ax.resolve_values(mode)
        if vals:
            n *= len(vals)
    n *= max(1, len(seeds or [42]))
    cap = VARIATION_MODES.get(mode, VARIATION_MODES[DEFAULT_MODE])["cap"]
    return {"estimated": n, "cap": cap, "within_cap": n <= cap,
            "mode": mode,
            "guidance": ("" if n <= cap else
                         f"Planned grid ({n}) exceeds the {mode} cap "
                         f"({cap}) - reduce axis values, raise the step, "
                         f"or choose fewer seeds.")}


def job_risk_level(score: float, has_anchor: bool) -> str:
    """Per-job risk classification for the variation manifest."""
    if not has_anchor or score < 0.50:
        return "blocked_or_needs_review"
    if score < 0.70:
        return "high_risk"
    if score < 0.85:
        return "caution"
    return "info"


@dataclass
class VariationGrid:
    """Spec for one controlled-variation batch.

    axes: list of {"slider": id, "values": [..]} and/or
          {"seeds": [..]} and/or {"stacks": [{name, weights{}}, ...]}
    """
    prompt_tail: str = ""               # composition goal / style descriptors
    negative_extra: str = ""
    seeds: list[int] = field(default_factory=lambda: [42])
    slider_axes: list[dict] = field(default_factory=list)
    base_state: dict = field(default_factory=dict)     # slider_id -> value
    max_jobs: int = 64                  # hard cap - no runaway grids
    mode: str = DEFAULT_MODE            # low_risk | balanced | creative
    overrides: dict = field(default_factory=dict)      # pinned lora weights


@dataclass
class VariationJob:
    batch_id: str
    variation_id: str
    prompt: str
    negative: str
    seed: int
    loras: list[list]                   # [[lora_id, weight], ...]
    slider_state: dict
    warnings: list[str]
    payload: dict
    output_path: str = ""
    preservation_score: float = 1.0
    risk_level: str = "info"
    status: str = "planned"            # planned | generated | failed


def parse_axes(slider_axes: list[dict]) -> list[VariationAxis]:
    known = VariationAxis.__dataclass_fields__
    return [VariationAxis(**{k: v for k, v in ax.items() if k in known})
            for ax in (slider_axes or []) if ax.get("slider")]


def _grid_states(grid: VariationGrid) -> Generator[dict, None, None]:
    """Deterministic cartesian product over up to three validated axes."""
    axes = []
    for ax in parse_axes(grid.slider_axes)[:3]:
        vals = ax.resolve_values(grid.mode)
        if vals:
            axes.append((ax.slider, vals))
    if not axes:
        yield dict(grid.base_state)
        return

    def rec(i: int, acc: dict):
        if i == len(axes):
            yield dict(acc)
            return
        sid, vals = axes[i]
        for v in vals:
            acc[sid] = v
            yield from rec(i + 1, acc)
    yield from rec(0, dict(grid.base_state))


def expand_grid(prj: Project, cards: list[LoraCard],
                grid: VariationGrid) -> list[VariationJob]:
    """Grid spec -> resolved variation jobs (identity anchor held fixed by
    the stack intelligence on every job)."""
    from .base_models import build_prompt, detect_profile
    from .eval.forge_api import ForgeClient
    prof = detect_profile(prj.base_model)
    trig = f"{prj.trigger_token} {prj.class_word}".strip()
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"
    jobs: list[VariationJob] = []

    mode_cap = VARIATION_MODES.get(grid.mode,
                                   VARIATION_MODES[DEFAULT_MODE])["cap"]
    cap = min(grid.max_jobs, mode_cap) if grid.max_jobs else mode_cap
    for state_dict in _grid_states(grid):
        for seed in (grid.seeds or [42]):
            if len(jobs) >= cap:
                return jobs
            state = ConceptSliderState(values=state_dict)
            stack = resolve_controlled_stack(
                cards, state, prj.base_model,
                overrides=grid.overrides or None)
            loras = ([[stack.identity_anchor.lora_id,
                       stack.identity_anchor.weight]]
                     if stack.identity_anchor else [])
            loras += [[i.lora_id, i.weight] for i in stack.concept_loras]
            prompt = build_prompt(prof, trig).rstrip(", ")
            prompt += ", consistent character identity"
            if grid.prompt_tail:
                prompt += f", {grid.prompt_tail.strip(', ')}"
            negative = prof["negative"]
            negative += (", identity drift, inconsistent facial structure, "
                         "unstable anatomy, distorted proportions")
            if grid.negative_extra:
                negative += f", {grid.negative_extra.strip(', ')}"
            payload = ForgeClient.build_txt2img_payload(
                prompt=prompt, negative=negative,
                steps=prof["steps"], cfg=prof["cfg"],
                width=prof["width"], height=prof["height"], seed=seed,
                sampler=prof["sampler"], clip_skip=prof["clip_skip"],
                loras=[(n, w) for n, w in loras])
            jobs.append(VariationJob(
                batch_id=batch_id,
                variation_id=f"v{len(jobs):03d}",
                prompt=prompt, negative=negative, seed=seed,
                loras=loras, slider_state=dict(state_dict),
                warnings=[w.message for w in stack.warnings
                          if w.severity != "info"],
                payload=payload,
                preservation_score=stack.identity_preservation_score,
                risk_level=job_risk_level(
                    stack.identity_preservation_score,
                    stack.identity_anchor is not None)))
    return jobs


@dataclass
class PromptSequence:
    """Spec for the Prompt Sequencer: one prompt template with a {slot}
    token (e.g. {garment}) cycled across a list of values, optionally
    crossed with a {pose} list and seeds, over a FIXED LoRA stack. Built
    for catalog-style 'same model, many wardrobe variations' batches."""
    template: str                                  # must contain a {slot}
    slot_values: list[str]                         # the wardrobe/variation list
    slot_token: str = "garment"                    # token name inside {..}
    pose_values: list[str] = field(default_factory=list)   # optional {pose}
    seeds: list[int] = field(default_factory=lambda: [42])
    loras: list[list] = field(default_factory=list)        # [[name, weight]]
    negative_extra: str = ""
    max_jobs: int = 64


def expand_prompt_sequence(prj: Project,
                           seq: PromptSequence) -> list[VariationJob]:
    """Expand a PromptSequence into resolved VariationJobs - one per
    (slot value x pose value x seed), capped. The LoRA stack is held fixed
    across the whole sequence so only the wardrobe/pose text changes."""
    from .base_models import detect_profile
    from .eval.forge_api import ForgeClient
    prof = detect_profile(prj.base_model)
    batch_id = f"seq_{uuid.uuid4().hex[:8]}"
    slot = seq.slot_token
    poses = seq.pose_values or [""]
    loras = [(str(n), float(w)) for n, w in (seq.loras or [])]
    negative = prof["negative"] + (", identity drift, inconsistent facial "
                                   "structure, unstable anatomy, distorted "
                                   "proportions, garment artifacts")
    if seq.negative_extra:
        negative += f", {seq.negative_extra.strip(', ')}"

    jobs: list[VariationJob] = []
    cap = min(seq.max_jobs or 64, 256)
    for slot_val in seq.slot_values:
        for pose_val in poses:
            for seed in (seq.seeds or [42]):
                if len(jobs) >= cap:
                    return jobs
                prompt = seq.template
                # token substitution (tolerant of missing {pose})
                prompt = prompt.replace("{" + slot + "}", str(slot_val))
                prompt = prompt.replace("{pose}", str(pose_val))
                prompt = prompt.strip().strip(",").strip()
                payload = ForgeClient.build_txt2img_payload(
                    prompt=prompt, negative=negative,
                    steps=prof["steps"], cfg=prof["cfg"],
                    width=prof["width"], height=prof["height"], seed=seed,
                    sampler=prof["sampler"], clip_skip=prof["clip_skip"],
                    loras=loras)
                jobs.append(VariationJob(
                    batch_id=batch_id,
                    variation_id=f"v{len(jobs):03d}",
                    prompt=payload["prompt"], negative=negative, seed=seed,
                    loras=[[n, w] for n, w in loras],
                    slider_state={"slot": slot_val, "pose": pose_val},
                    warnings=[], payload=payload))
    return jobs


def save_batch(conn, prj: Project, grid: VariationGrid,
               jobs: list[VariationJob]) -> str:
    """Persist the batch manifest (v8: variation_batches / variation_jobs)."""
    if not jobs:
        return ""
    bid = jobs[0].batch_id
    anchor = ""
    if jobs[0].loras:
        anchor = f"{jobs[0].loras[0][0]}:{jobs[0].loras[0][1]}"
    cap = VARIATION_MODES.get(grid.mode,
                              VARIATION_MODES[DEFAULT_MODE])["cap"]
    conn.execute(
        "INSERT OR REPLACE INTO variation_batches "
        "(batch_id, spec, base_model, job_count, created_at, mode, "
        "source_state, identity_anchor, hard_cap) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (bid, json.dumps({"prompt_tail": grid.prompt_tail,
                          "slider_axes": grid.slider_axes,
                          "seeds": grid.seeds,
                          "base_state": grid.base_state}),
         prj.base_model or "", len(jobs), now_iso(), grid.mode,
         json.dumps(grid.base_state), anchor, cap))
    for j in jobs:
        conn.execute(
            "INSERT OR REPLACE INTO variation_jobs (batch_id, variation_id, "
            "prompt, negative, seed, loras, slider_state, warnings, "
            "output_path, created_at, preservation_score, risk_level, "
            "status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (j.batch_id, j.variation_id, j.prompt, j.negative, j.seed,
             json.dumps(j.loras), json.dumps(j.slider_state),
             json.dumps(j.warnings), j.output_path, now_iso(),
             j.preservation_score, j.risk_level, j.status))
    conn.commit()
    return bid


def load_batch(conn, batch_id: str) -> dict:
    """Batch manifest + jobs for review/resume (empty dict if unknown)."""
    row = conn.execute(
        "SELECT spec, base_model, job_count, created_at, mode, "
        "identity_anchor, hard_cap FROM variation_batches "
        "WHERE batch_id=?", (batch_id,)).fetchone()
    if not row:
        return {}
    jobs = [{"variation_id": r[0], "seed": r[1],
             "loras": json.loads(r[2] or "[]"),
             "slider_state": json.loads(r[3] or "{}"),
             "warnings": json.loads(r[4] or "[]"),
             "output_path": r[5], "preservation_score": r[6],
             "risk_level": r[7], "status": r[8] or "planned",
             "measured_face_sim": r[9]}
            for r in conn.execute(
                "SELECT variation_id, seed, loras, slider_state, warnings, "
                "output_path, preservation_score, risk_level, status, "
                "measured_face_sim "
                "FROM variation_jobs WHERE batch_id=? ORDER BY variation_id",
                (batch_id,))]
    return {"batch_id": batch_id, "spec": json.loads(row[0] or "{}"),
            "base_model": row[1], "job_count": row[2], "created_at": row[3],
            "mode": row[4] or DEFAULT_MODE, "identity_anchor": row[5],
            "hard_cap": row[6], "jobs": jobs}


def run_batch_generator(prj: Project, conn, jobs: list[VariationJob],
                        forge_url: str = "http://127.0.0.1:7860",
                        out_dir: str | None = None
                        ) -> Generator[str, None, None]:
    """Execute a batch through the Forge adapter when it is live; each
    completed variation updates its manifest row (resumable: rows with an
    output_path are skipped)."""
    from pathlib import Path

    from .eval.forge_api import ForgeClient
    client = ForgeClient(forge_url)
    if not client.alive():
        yield (f"Forge API not reachable at {forge_url} - batch manifest "
               f"saved; start Forge and re-run to generate.")
        return
    base = Path(out_dir or (prj.output_path / "variation_batches"))
    for j in jobs:
        done = conn.execute(
            "SELECT output_path FROM variation_jobs WHERE batch_id=? AND "
            "variation_id=?", (j.batch_id, j.variation_id)).fetchone()
        if done and done[0]:
            yield f"  {j.variation_id} already generated - skip"
            continue
        try:
            png = client._post("/sdapi/v1/txt2img", j.payload)
            import base64
            img = base64.b64decode(png["images"][0])
            p = base / j.batch_id
            p.mkdir(parents=True, exist_ok=True)
            fp = p / f"{j.variation_id}_seed{j.seed}.png"
            fp.write_bytes(img)
            sim = None
            try:
                from .identity_integration import face_similarity
                ref = None
                if prj.anchor_dir:
                    refs = sorted(Path(prj.anchor_dir).expanduser().glob(
                        "*.[jp][pn]g"))
                    ref = refs[0] if refs else None
                if ref:
                    sim = face_similarity(fp, ref, prj)
            except Exception:
                sim = None
            conn.execute(
                "UPDATE variation_jobs SET output_path=?, status='generated',"
                " measured_face_sim=? WHERE batch_id=? AND variation_id=?",
                (str(fp), sim, j.batch_id, j.variation_id))
            conn.commit()
            yield f"  {j.variation_id} -> {fp.name}"
        except Exception as exc:                          # noqa: BLE001
            conn.execute(
                "UPDATE variation_jobs SET status='failed' WHERE "
                "batch_id=? AND variation_id=?",
                (j.batch_id, j.variation_id))
            conn.commit()
            yield f"  {j.variation_id} FAILED: {exc}"
    yield "Batch complete."
