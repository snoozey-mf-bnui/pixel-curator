"""Self-improvement loop - learn from every generation.

A LoRA's weights are frozen at inference, so generating images cannot
change the .safetensors. What *can* learn is everything around it: which
prompts, negative chains, LoRA stacks, weights, recipes and seeds
produce images you rate highly. This module closes that loop in four
stages:

  1. capture   - record every generation's full settings + an automatic
                 identity score, and let you rate it (record_generation /
                 rate_generation / auto_score_identity);
  2. learn     - aggregate the history into a transparent "what works for
                 me" preference profile (aggregate_preferences);
  3. bias      - recommend the proven settings, with light exploration,
                 to steer future generation (recommend_from_preferences);
  4. harvest   - select your best, high-identity images as candidate
                 training data for a sharper next LoRA
                 (select_training_candidates / stage_harvest).

Everything is deterministic and degradation-safe: identity scoring is
optional (skipped cleanly without InsightFace), and an empty history
yields an empty-but-valid profile rather than an error.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# reward = blend of your rating and the measured identity score
REWARD_RATING_WEIGHT = 0.7
REWARD_IDENTITY_WEIGHT = 0.3
# rating scale: -1 (down), 0 (unrated), 1..5 (stars); normalized to 0..1
_RATING_NORM = {-1: 0.0, 0: 0.5, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}

WEIGHT_BUCKETS = [(0.0, 0.4, "low"), (0.4, 0.7, "mid"),
                  (0.7, 1.2, "high")]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _weight_bucket(w: float) -> str:
    for lo, hi, label in WEIGHT_BUCKETS:
        if lo <= w < hi:
            return label
    return "high"


def reward(rating: int, identity_score: float | None) -> float:
    """Transparent scalar reward in 0..1 from a rating + identity score."""
    r = _RATING_NORM.get(int(rating), 0.5)
    if identity_score is None:
        return round(r, 4)
    return round(REWARD_RATING_WEIGHT * r
                 + REWARD_IDENTITY_WEIGHT * float(identity_score), 4)


# ---------------------------------------------------------------------------
# Stage 1: capture
# ---------------------------------------------------------------------------

def record_generation(conn, image_path: str, prompt: str = "",
                      negative_prompt: str = "", lora_stack=None,
                      seed: int = 0, recipe: str = "", batch_id: str = "",
                      identity_score: float | None = None,
                      aesthetic_score: float | None = None,
                      settings: dict | None = None) -> int:
    """Record one generation. Idempotent on image_path: re-recording the
    same path updates it rather than duplicating."""
    stack_json = json.dumps(lora_stack or [])
    settings_json = json.dumps(settings or {})
    row = conn.execute(
        "SELECT feedback_id FROM generation_feedback WHERE image_path=?",
        (image_path,)).fetchone()
    if row:
        # re-recording the same image only overwrites fields that were
        # actually supplied - a partial re-record never clobbers data
        conn.execute(
            "UPDATE generation_feedback SET "
            "prompt=COALESCE(NULLIF(?,''), prompt), "
            "negative_prompt=COALESCE(NULLIF(?,''), negative_prompt), "
            "lora_stack=CASE WHEN ?='[]' THEN lora_stack ELSE ? END, "
            "seed=CASE WHEN ?=0 THEN seed ELSE ? END, "
            "recipe=COALESCE(NULLIF(?,''), recipe), "
            "batch_id=COALESCE(NULLIF(?,''), batch_id), "
            "identity_score=COALESCE(?, identity_score), "
            "aesthetic_score=COALESCE(?, aesthetic_score), "
            "settings_json=CASE WHEN ?='{}' THEN settings_json ELSE ? END "
            "WHERE feedback_id=?",
            (prompt, negative_prompt, stack_json, stack_json, seed, seed,
             recipe, batch_id, identity_score, aesthetic_score,
             settings_json, settings_json, row[0]))
        conn.commit()
        return int(row[0])
    cur = conn.execute(
        "INSERT INTO generation_feedback(image_path, prompt, "
        "negative_prompt, lora_stack, seed, recipe, batch_id, "
        "identity_score, aesthetic_score, rating, tags, settings_json, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,0,'',?,?)",
        (image_path, prompt, negative_prompt, stack_json, seed, recipe,
         batch_id, identity_score, aesthetic_score, settings_json, _now()))
    conn.commit()
    return int(cur.lastrowid)


def rate_generation(conn, image_path: str = "", feedback_id: int = 0,
                    rating: int = 0, tags=None) -> dict:
    """Attach your rating (-1, or 1..5) and optional tags to a recorded
    generation, keyed by image path or feedback id."""
    if feedback_id:
        where, val = "feedback_id=?", feedback_id
    else:
        where, val = "image_path=?", image_path
    row = conn.execute(
        f"SELECT feedback_id, tags FROM generation_feedback WHERE {where}",
        (val,)).fetchone()
    if not row:
        raise KeyError("no recorded generation for "
                       f"{'id ' + str(feedback_id) if feedback_id else image_path}")
    merged = sorted(set(
        [t for t in (row[1] or "").split(",") if t] + list(tags or [])))
    conn.execute(
        "UPDATE generation_feedback SET rating=?, tags=?, rated_at=? "
        "WHERE feedback_id=?",
        (int(rating), ",".join(merged), _now(), row[0]))
    conn.commit()
    return {"feedback_id": int(row[0]), "rating": int(rating),
            "tags": merged}


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _read_png_parameters(path: Path) -> str:
    """Extract the A1111/Forge `parameters` text chunk from a PNG (the
    generation metadata embedded by the webui). Stdlib only; returns ''
    for non-PNG or chunkless files."""
    try:
        with open(path, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return ""
            while True:
                head = f.read(8)
                if len(head) < 8:
                    return ""
                length = int.from_bytes(head[:4], "big")
                ctype = head[4:8]
                data = f.read(length)
                f.read(4)                       # CRC
                if ctype in (b"tEXt", b"iTXt"):
                    # keyword\x00...text
                    if b"parameters" in data[:20]:
                        text = data.split(b"\x00", 1)[-1]
                        # iTXt has extra compression/lang bytes; strip them
                        return text.split(b"\x00")[-1].decode(
                            "utf-8", "ignore")
                if ctype == b"IDAT" or ctype == b"IEND":
                    return ""                   # metadata precedes pixels
    except Exception:
        return ""
    return ""


def parse_generation_params(params: str) -> dict:
    """Parse an A1111/Forge parameters string into prompt, negative,
    seed and LoRA stack (<lora:name:weight> tags)."""
    import re
    out: dict = {"prompt": "", "negative_prompt": "", "seed": 0,
                 "lora_stack": []}
    if not params:
        return out
    neg_split = re.split(r"\nNegative prompt:\s*", params, maxsplit=1)
    out["prompt"] = neg_split[0].strip()
    rest = neg_split[1] if len(neg_split) > 1 else ""
    # the settings line (Steps:, Seed:, ...) starts the param block
    m = re.search(r"\n(Steps:.*)", rest, re.S)
    if m:
        out["negative_prompt"] = rest[:m.start()].strip()
        settings = m.group(1)
    else:
        out["negative_prompt"] = rest.strip()
        settings = params
    sm = re.search(r"Seed:\s*(\d+)", settings)
    if sm:
        out["seed"] = int(sm.group(1))
    for name, weight in re.findall(r"<lora:([^:>]+):([\d.]+)>", params):
        out["lora_stack"].append({"name": name.strip(),
                                  "weight": float(weight)})
    return out


def ingest_directory(conn, directory, reference_image: str = "",
                     prj=None, recipe: str = "", recursive: bool = True,
                     limit: int = 0) -> dict:
    """Record every image in a folder of stored generations into the
    feedback loop - reading embedded prompt/seed/LoRA metadata when
    present and (optionally) auto-scoring identity. Idempotent: images
    already recorded are updated, not duplicated. Great for ingesting a
    Forge txt2img-images folder so 'wherever your generations are' feeds
    the learning profile."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        return {"error": f"not a directory: {directory}", "scanned": 0,
                "recorded": 0, "skipped": 0}
    walker = directory.rglob("*") if recursive else directory.iterdir()
    scanned = recorded = skipped = scored = 0
    for p in sorted(walker):
        if not (p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            continue
        scanned += 1
        already = conn.execute(
            "SELECT feedback_id FROM generation_feedback WHERE image_path=?",
            (str(p),)).fetchone()
        params = parse_generation_params(_read_png_parameters(p))
        ids = None
        if reference_image:
            ids = auto_score_identity(str(p), reference_image, prj)
            if ids is not None:
                scored += 1
        record_generation(
            conn, str(p), prompt=params["prompt"],
            negative_prompt=params["negative_prompt"],
            lora_stack=params["lora_stack"], seed=params["seed"],
            recipe=recipe, identity_score=ids,
            settings={"source": "ingest", "dir": str(directory)})
        if already:
            skipped += 1
        else:
            recorded += 1
        if limit and recorded >= limit:
            break
    return {"directory": str(directory), "scanned": scanned,
            "recorded": recorded, "updated": skipped, "scored": scored}


def auto_score_identity(image_path: str, reference_image: str,
                        prj=None) -> float | None:
    """Measured face-consistency score (0..1) via the Identity
    Integration Layer; None when scoring is unavailable."""
    try:
        from ..identity_integration import face_similarity
        return face_similarity(image_path, reference_image, prj)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Stage 2: learn a transparent preference profile
# ---------------------------------------------------------------------------

def _accumulate(table: dict, key: str, r: float):
    e = table.setdefault(key, {"reward_sum": 0.0, "n": 0})
    e["reward_sum"] += r
    e["n"] += 1


def _ranked(table: dict, min_n: int = 1) -> list[dict]:
    out = []
    for key, e in table.items():
        if e["n"] < min_n:
            continue
        out.append({"value": key, "mean_reward": round(e["reward_sum"]
                                                        / e["n"], 4),
                    "n": e["n"]})
    return sorted(out, key=lambda x: (-x["mean_reward"], -x["n"]))


def aggregate_preferences(conn, min_samples: int = 1,
                          store: bool = True) -> dict:
    """Aggregate feedback history into a per-dimension reward profile:
    LoRAs, recipes, weight buckets per LoRA role, negative chains and
    seeds. Transparent (mean reward + sample count) - no black box."""
    rows = conn.execute(
        "SELECT lora_stack, recipe, seed, identity_score, rating, "
        "settings_json FROM generation_feedback").fetchall()
    loras: dict = {}
    recipes: dict = {}
    seeds: dict = {}
    weight_buckets: dict = {}
    neg_chains: dict = {}
    total_reward = 0.0
    rated = 0

    for stack_json, recipe, seed, identity_score, rating, settings_json \
            in rows:
        r = reward(rating, identity_score)
        total_reward += r
        if rating:
            rated += 1
        if recipe:
            _accumulate(recipes, recipe, r)
        if seed:
            _accumulate(seeds, str(seed), r)
        try:
            stack = json.loads(stack_json or "[]")
        except Exception:
            stack = []
        for item in stack:
            name = (item.get("name") if isinstance(item, dict)
                    else str(item))
            w = (item.get("weight", 0) if isinstance(item, dict) else 0)
            if name:
                _accumulate(loras, name, r)
                _accumulate(weight_buckets, f"{name}@{_weight_bucket(w)}",
                            r)
        try:
            settings = json.loads(settings_json or "{}")
        except Exception:
            settings = {}
        for chain in settings.get("negative_chains", []) or []:
            _accumulate(neg_chains, chain, r)

    profile = {
        "samples": len(rows),
        "rated_samples": rated,
        "mean_reward": round(total_reward / len(rows), 4) if rows else 0.0,
        "top_loras": _ranked(loras, min_samples),
        "top_recipes": _ranked(recipes, min_samples),
        "top_seeds": _ranked(seeds, min_samples)[:10],
        "weight_buckets": _ranked(weight_buckets, min_samples),
        "negative_chains": _ranked(neg_chains, min_samples),
        "updated_at": _now(),
    }
    if store:
        try:
            from .. import manifest as mf
            mf.meta_set(conn, "learning:preferences", profile)
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass
    return profile


def load_preferences(conn) -> dict:
    try:
        from .. import manifest as mf
        return mf.meta_get(conn, "learning:preferences") or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Stage 3: bias generation toward winners
# ---------------------------------------------------------------------------

def recommend_from_preferences(conn, candidate_loras: list | None = None,
                               max_concepts: int = 3,
                               explore: float = 0.15,
                               seed: int = 0) -> dict:
    """Recommend proven concept LoRAs + a preferred weight bucket + the
    best recipe/seed from learned history, with epsilon exploration so it
    keeps discovering. candidate_loras (compatible names) constrains the
    pool; without it, all seen LoRAs are eligible."""
    import random
    rng = random.Random(seed or 1)
    prof = aggregate_preferences(conn, store=False)
    if prof["samples"] == 0:
        return {"status": "cold_start",
                "reason": "No feedback yet - generate and rate a few "
                          "images, then recommendations will improve.",
                "loras": [], "recipe": None, "seed": None,
                "explore": explore}

    pool = prof["top_loras"]
    if candidate_loras is not None:
        allowed = set(candidate_loras)
        pool = [p for p in pool if p["value"] in allowed]

    chosen: list[dict] = []
    for entry in pool:
        if len(chosen) >= max_concepts:
            break
        # exploit winners; occasionally explore a lower-ranked option
        if entry["mean_reward"] >= prof["mean_reward"] \
                or rng.random() < explore:
            bucket = next((w for w in prof["weight_buckets"]
                           if w["value"].startswith(entry["value"] + "@")),
                          None)
            chosen.append({
                "name": entry["value"],
                "mean_reward": entry["mean_reward"], "n": entry["n"],
                "preferred_weight_bucket":
                    bucket["value"].split("@")[-1] if bucket else "mid"})

    best_recipe = (prof["top_recipes"][0]["value"]
                   if prof["top_recipes"] else None)
    best_seed = (int(prof["top_seeds"][0]["value"])
                 if prof["top_seeds"] else None)
    return {"status": "learned", "loras": chosen, "recipe": best_recipe,
            "seed": best_seed, "mean_reward": prof["mean_reward"],
            "explore": explore, "negative_chains":
                [c["value"] for c in prof["negative_chains"][:3]]}


# ---------------------------------------------------------------------------
# Stage 4: harvest the best generations as training data
# ---------------------------------------------------------------------------

def select_training_candidates(conn, min_rating: int = 4,
                               min_identity: float = 0.6,
                               limit: int = 200) -> list[dict]:
    """Top-rated, high-identity generations that exist on disk - the raw
    material for a sharper next LoRA."""
    rows = conn.execute(
        "SELECT image_path, prompt, seed, identity_score, rating, tags "
        "FROM generation_feedback WHERE rating >= ? "
        "ORDER BY rating DESC, identity_score DESC LIMIT ?",
        (min_rating, limit * 4)).fetchall()
    out = []
    for path, prompt, seed, ident, rating, tags in rows:
        if ident is not None and ident < min_identity:
            continue
        if not Path(path).exists():
            continue
        out.append({"image_path": path, "prompt": prompt, "seed": seed,
                    "identity_score": ident, "rating": rating,
                    "tags": tags})
        if len(out) >= limit:
            break
    return out


def stage_harvest(conn, output_base: Path, min_rating: int = 4,
                  min_identity: float = 0.6, apply: bool = False) -> dict:
    """Stage the best generations into a harvest directory + manifest -
    candidate training data for the next LoRA. Dry by default (apply=False
    plans only); non-destructive (copies, never moves)."""
    import shutil
    cands = select_training_candidates(conn, min_rating, min_identity)
    root = Path(output_base) / "harvest"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = root / f"harvest_{stamp}"
    plan = {
        "harvest_name": target.name,
        "criteria": {"min_rating": min_rating,
                     "min_identity": min_identity},
        "candidate_count": len(cands),
        "candidates": cands,
        "applied": False,
        "next_step": "Curate these into a new dataset version, then "
                     "retrain for a sharper identity LoRA: "
                     "pixel-curator build / train.",
    }
    if apply and cands:
        target.mkdir(parents=True, exist_ok=True)
        copied = 0
        for c in cands:
            try:
                shutil.copy2(c["image_path"],
                             target / Path(c["image_path"]).name)
                copied += 1
            except Exception:
                logger.debug("suppressed exception", exc_info=True)
                pass
        plan["applied"] = True
        plan["staged_dir"] = str(target)
        plan["copied"] = copied
        (target / "harvest_manifest.json").write_text(
            json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def render_profile(profile: dict) -> str:
    if not profile or profile.get("samples", 0) == 0:
        return ("LEARNING PROFILE: no feedback yet - generate and rate "
                "images to start the loop.")
    lines = [
        f"LEARNING PROFILE | {profile['samples']} generations "
        f"({profile['rated_samples']} rated) | mean reward "
        f"{profile['mean_reward']:.2f}",
        "  top LoRAs (mean reward x n):"]
    for e in profile["top_loras"][:8]:
        lines.append(f"    {e['value']:<42} {e['mean_reward']:.2f} "
                     f"(n={e['n']})")
    if profile["top_recipes"]:
        lines.append("  top recipes:")
        for e in profile["top_recipes"][:5]:
            lines.append(f"    {e['value']:<30} {e['mean_reward']:.2f} "
                         f"(n={e['n']})")
    return "\n".join(lines)
