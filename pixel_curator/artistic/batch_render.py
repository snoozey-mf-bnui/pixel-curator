"""Batch render executor - turns a planned figure-study batch into images.

Closes the loop between planning (prompts, LoRA stacks, multi-pass
manifests) and pixels: reads a batch directory's batch_manifest.json,
renders every generation item through the Forge txt2img API and every
refinement item through img2img (init image = its rendered parent), and
writes the results into the batch's samples/ and refined/ folders. A
render_manifest.json records per-item status so review, best-variant
selection and packaging operate on real images.

Handles both manifest shapes: Phase 4 flat batches (top-level `items`)
and Phase 5 multi-pass production batches (`passes` -> `items`). LoRAs
are invoked the webui way (<lora:name:weight>); the checkpoint and clip
skip are pinned so output matches a known-good Forge session. Optional
inswapper face lock applies a source face onto each render.

Streaming generator (yields progress strings) to match the studio's
other long-running flows. The Forge client is injectable for testing,
so the executor runs with no real server, GPU or model files.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Generator
from pathlib import Path

logger = logging.getLogger(__name__)


def _lora_pairs(stack: list[dict]) -> list[tuple[str, float]]:
    return [(s["name"], s["weight"]) for s in (stack or [])
            if s.get("weight", 0) > 0]


def _iter_render_units(plan: dict) -> list[dict]:
    """Normalize both manifest shapes into a flat list of render units:
    {item_id, kind: 'txt2img'|'img2img', seed, prompt, negative_prompt,
     lora_stack, denoise, refines, pass_id}."""
    units: list[dict] = []
    if plan.get("passes"):                       # Phase 5 multi-pass
        from .figure_study import PRODUCTION_PASSES
        for p in plan["passes"]:
            kind_map = PRODUCTION_PASSES.get(p["pass_type"], {}).get("kind")
            if kind_map not in ("generation", "refinement"):
                continue
            for it in p["items"]:
                units.append({
                    "item_id": it.get("item_id"),
                    "kind": ("img2img" if kind_map == "refinement"
                             else "txt2img"),
                    "seed": it.get("seed", 42),
                    "prompt": it.get("prompt", ""),
                    "negative_prompt": it.get("negative_prompt", ""),
                    "lora_stack": it.get("lora_stack", []),
                    "denoise": it.get("denoise", 0.45),
                    "refines": it.get("refines"),
                    "pass_id": p.get("pass_id"),
                })
    else:                                        # Phase 4 flat batch
        for it in plan.get("items", []):
            iid = it.get("item_id") or f"item_{it.get('index', 0):04d}"
            units.append({
                "item_id": iid, "kind": "txt2img",
                "seed": it.get("seed", 42), "prompt": it.get("prompt", ""),
                "negative_prompt": it.get("negative_prompt", ""),
                "lora_stack": it.get("lora_stack", []),
                "denoise": 0.0, "refines": None, "pass_id": "single"})
    return units


def render_batch(
        batch_dir: Path, prj,
        forge_url: str = "http://127.0.0.1:7860",
        client=None,
        checkpoint: str = "",
        clip_skip: int = 2,
        width: int = 1024, height: int = 1024,
        steps: int = 28, cfg: float = 6.0,
        limit: int = 0,
        face_lock: bool = False,
        reference_image: str = "",
        overwrite: bool = False,
        face_swap_fn: Callable | None = None
        ) -> Generator[str, None, dict]:
    """Render a planned batch through Forge. Yields progress lines and
    returns a summary dict.

    client: inject a Forge-API-compatible client for testing; otherwise a
    real ForgeClient(forge_url) is created. checkpoint defaults to the
    project base model stem so renders match your working session.
    """
    batch_dir = Path(batch_dir)
    plan_path = batch_dir / "batch_manifest.json"
    if not plan_path.exists():
        yield f"FATAL: no batch_manifest.json in {batch_dir}"
        return {"ok": False, "error": "manifest missing"}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    if client is None:
        from ..eval.forge_api import ForgeClient
        client = ForgeClient(forge_url)
    if not client.alive():
        yield ("FATAL: Forge API not reachable - start reForge with "
               "--api, then re-run.")
        return {"ok": False, "error": "forge unreachable"}

    if not checkpoint:
        bm = getattr(prj, "base_model", "") or ""
        checkpoint = Path(bm).stem if bm else ""
    if face_lock and face_swap_fn is None:
        try:
            from ..identity_integration import apply_face_identity_postprocess as face_swap_fn
        except Exception:
            face_swap_fn = None

    samples = batch_dir / "samples"
    refined = batch_dir / "refined"
    samples.mkdir(exist_ok=True)
    refined.mkdir(exist_ok=True)

    units = _iter_render_units(plan)
    if limit > 0:
        units = units[:limit]
    rendered: dict[str, str] = {}
    records: list[dict] = []
    ok = err = skipped = 0

    yield (f"RENDER BATCH '{plan.get('batch_name', batch_dir.name)}' - "
           f"{len(units)} item(s) | checkpoint "
           f"{checkpoint or '(current Forge model)'}")

    for u in units:
        iid = u["item_id"]
        is_refine = u["kind"] == "img2img"
        target_dir = refined if is_refine else samples
        fpath = target_dir / f"{iid}.png"
        if fpath.exists() and not overwrite:
            yield f"  {iid}: exists, skip (use --overwrite to regenerate)"
            rendered[iid] = str(fpath)
            skipped += 1
            continue
        try:
            loras = _lora_pairs(u["lora_stack"])
            if is_refine:
                parent = rendered.get(u["refines"]) or str(
                    samples / f"{u['refines']}.png")
                if not Path(parent).exists():
                    yield (f"  {iid}: refine source '{u['refines']}' not "
                           "rendered - skipping")
                    err += 1
                    records.append({"item_id": iid, "status": "no_source",
                                    "refines": u["refines"]})
                    continue
                yield f"  {iid}: refining {u['refines']} (denoise {u['denoise']})..."
                png = client.img2img(
                    Path(parent).read_bytes(), prompt=u["prompt"],
                    negative=u["negative_prompt"], strength=u["denoise"],
                    steps=steps, cfg=cfg, seed=u["seed"], loras=loras)
            else:
                yield f"  {iid}: generating (seed {u['seed']})..."
                png = client.txt2img(
                    prompt=u["prompt"], negative=u["negative_prompt"],
                    steps=steps, cfg=cfg, width=width, height=height,
                    seed=u["seed"], loras=loras, clip_skip=clip_skip,
                    checkpoint=checkpoint)
            fpath.write_bytes(png)
            locked = False
            if face_lock and reference_image and face_swap_fn:
                try:
                    locked = bool(face_swap_fn(fpath, reference_image, prj))
                except Exception:
                    locked = False
            rendered[iid] = str(fpath)
            records.append({"item_id": iid, "status": "rendered",
                            "path": str(fpath), "kind": u["kind"],
                            "seed": u["seed"], "face_locked": locked})
            # self-improvement loop: record every render with its full
            # settings (+ identity score if a reference is available) so
            # the learning profile can grow. Best-effort, never fatal.
            try:
                from .. import manifest as mf
                from ..studio.learning import auto_score_identity, record_generation
                ids = (auto_score_identity(fpath, reference_image, prj)
                       if reference_image else None)
                lconn = mf.connect(batch_dir.parent.parent)
                record_generation(
                    lconn, str(fpath), prompt=u["prompt"],
                    negative_prompt=u["negative_prompt"],
                    lora_stack=u["lora_stack"], seed=u["seed"],
                    recipe=plan.get("recipe", ""),
                    batch_id=batch_dir.name, identity_score=ids)
                lconn.close()
            except Exception:
                logger.debug("suppressed exception", exc_info=True)
                pass
            ok += 1
        except Exception as exc:                            # noqa: BLE001
            yield f"  {iid}: render failed - {exc}"
            records.append({"item_id": iid, "status": "error",
                            "error": str(exc)})
            err += 1

    # render manifest
    (batch_dir / "render_manifest.json").write_text(
        json.dumps({"batch": plan.get("batch_name", batch_dir.name),
                    "rendered": ok, "skipped": skipped, "errors": err,
                    "items": records}, indent=2), encoding="utf-8")

    # annotate review manifest with rendered sample paths (best-effort)
    rp = batch_dir / "review_manifest.json"
    if rp.exists():
        try:
            review = json.loads(rp.read_text(encoding="utf-8"))
            for entry in review.get("items", []):
                if entry["item_id"] in rendered:
                    entry["sample_path"] = rendered[entry["item_id"]]
            rp.write_text(json.dumps(review, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass

    yield (f"DONE - {ok} rendered, {skipped} skipped, {err} error(s) -> "
           f"{samples}")
    yield (f"Review: pixel-curator figure-review --batch {batch_dir.name} "
           "--summary")
    return {"ok": err == 0, "rendered": ok, "skipped": skipped,
            "errors": err, "samples_dir": str(samples)}
