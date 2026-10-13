"""Phase 5: Dataset Intelligence Layer - holistic pre-flight analysis.

Answers the questions a curation run can't answer about itself:

* how much *usable* data does this archive actually contain?
* what curation strategy fits it (how aggressive can filtering be)?
* what does that mean for training (usable frames vs the step bands the
  parameter advisor targets)?

Works on both raw and curated data: once frames are scored, everything
comes straight from the manifest; on a raw archive a small random sample is
scored (via ``curate.basic.score_frame_ex``) and rates are projected with
an honest "estimated from sample" label. Every recommendation is a
transparent rule with a stated reason - guidance, never automation.
"""

from __future__ import annotations

import random
from pathlib import Path

from .config import CurateConfig, PrefilterConfig

# projection uncertainty applied to sample-based estimates
_SAMPLE_MARGIN = 0.15
# dedup window mirrors curate.basic (last-12 kept frames, ordered by path)
_DUP_WINDOW = 12


# ---------------------------------------------------------------------------
# collectors
# ---------------------------------------------------------------------------

def _inventory(conn) -> dict:
    status_counts = dict(conn.execute(
        "SELECT status, COUNT(*) FROM frames GROUP BY status"))
    sources = dict(conn.execute(
        "SELECT kind, COUNT(*) FROM sources GROUP BY kind"))
    total = sum(status_counts.values())
    return {"total_frames": total, "by_status": status_counts,
            "sources": sources}


def _stage(inv: dict, conn) -> str:
    """Rough pipeline stage: raw -> basic_curated -> smart_curated ->
    captioned. Drives which collectors trust the DB vs a sample."""
    by = inv["by_status"]
    total = max(1, inv["total_frames"])
    if by.get("new", 0) / total > 0.5:
        return "raw"
    has_det = conn.execute(
        "SELECT COUNT(*) FROM detections LIMIT 1").fetchone()[0] > 0
    has_cap = conn.execute(
        "SELECT COUNT(*) FROM captions WHERE caption_text != '' "
        "LIMIT 1").fetchone()[0] > 0
    if has_cap:
        return "captioned"
    if has_det:
        return "smart_curated"
    if by.get("selected", 0) or by.get("scored", 0):
        return "basic_curated"
    return "raw"


def _scored_rates(conn, ccfg: CurateConfig) -> dict | None:
    """Projected reject rates from frames that already carry scores."""
    rows = conn.execute(
        "SELECT sharpness, brightness FROM frames "
        "WHERE sharpness IS NOT NULL").fetchall()
    if len(rows) < 20:
        return None
    n = len(rows)
    blur = sum(1 for s, _ in rows if s is not None
               and s < ccfg.min_sharpness)
    expo = sum(1 for s, b in rows if b is not None
               and (b < ccfg.min_brightness or b > ccfg.max_brightness)
               and (s is None or s >= ccfg.min_sharpness))
    sharp_vals = sorted(s for s, _ in rows if s is not None)
    return {"basis": f"{n} scored frames", "sampled": False, "n": n,
            "blur_rate": round(blur / n, 3),
            "exposure_rate": round(expo / n, 3),
            "sharpness_median": round(
                sharp_vals[len(sharp_vals) // 2], 1) if sharp_vals else None}


def _sample_rates(conn, sample: int, ccfg: CurateConfig,
                  pcfg: PrefilterConfig, seed: int = 42) -> dict | None:
    """Score a random sample of 'new' frames to project reject rates on a
    raw archive. Also runs the zero-decode prefilter battery and estimates
    within-source duplicate pressure on contiguous runs."""
    from .curate.basic import hamming, score_frame_ex
    from .prefilter import prefilter_path

    rows = conn.execute(
        "SELECT frame_id, source_id, path FROM frames WHERE status='new' "
        "ORDER BY source_id, path").fetchall()
    if not rows:
        return None
    if len(rows) > sample:
        rng = random.Random(seed)
        # contiguous runs per source keep the dup estimate meaningful
        start = rng.randrange(max(1, len(rows) - sample))
        rows = rows[start:start + sample]

    n = blur = expo = pf = dup = unreadable = 0
    prev_source = None
    recent: list[int] = []
    for _fid, source_id, path in rows:
        n += 1
        if prefilter_path(Path(path), pcfg):
            pf += 1
            continue
        res = score_frame_ex(path, text_check=pcfg.text_check)
        if res is None:
            unreadable += 1
            continue
        dh, sharp, bright, pf_reason = res
        if pf_reason:
            pf += 1
            continue
        if sharp < ccfg.min_sharpness:
            blur += 1
        elif bright < ccfg.min_brightness or bright > ccfg.max_brightness:
            expo += 1
        else:
            h = int(dh, 16)
            if source_id != prev_source:
                recent = []
                prev_source = source_id
            if any(hamming(h, k) <= ccfg.hamming_threshold
                   for k in recent[-_DUP_WINDOW:]):
                dup += 1
            else:
                recent.append(h)
    if n == 0:
        return None
    return {"basis": f"random sample of {n} 'new' frames", "sampled": True,
            "n": n,
            "prefilter_rate": round(pf / n, 3),
            "blur_rate": round(blur / n, 3),
            "exposure_rate": round(expo / n, 3),
            "duplicate_rate": round(dup / n, 3),
            "unreadable_rate": round(unreadable / n, 3)}


def _dup_pressure(conn, inv: dict) -> dict | None:
    """Actual duplicate pressure once curation ran."""
    by = inv["by_status"]
    dupes = by.get("duplicate", 0) + by.get("rejected_neardup", 0)
    considered = dupes + by.get("selected", 0) + by.get("packaged", 0)
    if considered == 0:
        return None
    return {"duplicates": dupes,
            "rate": round(dupes / considered, 3)}


def _identity(conn) -> dict | None:
    row = conn.execute(
        "SELECT COUNT(*), AVG(identity_sim) FROM detections "
        "WHERE identity_sim IS NOT NULL").fetchone()
    if not row or not row[0]:
        return None
    return {"frames_with_identity": row[0],
            "mean_identity_sim": round(row[1], 3)}


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------

def _usable_estimate(inv: dict, rates: dict | None,
                     dup: dict | None) -> dict:
    by = inv["by_status"]
    selected = by.get("selected", 0) + by.get("packaged", 0)
    pending = by.get("new", 0) + by.get("scored", 0)
    if pending == 0 or rates is None:
        return {"usable": selected, "low": selected, "high": selected,
                "basis": "current curated statuses"}
    survive = 1.0
    for key in ("prefilter_rate", "blur_rate", "exposure_rate",
                "duplicate_rate", "unreadable_rate"):
        survive *= 1.0 - float(rates.get(key, 0.0) or 0.0)
    if not rates.get("duplicate_rate") and dup:
        survive *= 1.0 - dup["rate"]
    projected = int(pending * max(0.0, survive))
    margin = _SAMPLE_MARGIN if rates.get("sampled") else 0.05
    return {"usable": selected + projected,
            "low": selected + int(projected * (1 - margin)),
            "high": selected + int(projected * (1 + margin)),
            "projected_from_pending": projected,
            "basis": f"{rates['basis']} projected over {pending} "
                     "uncurated frames"}


def _recommendations(inv: dict, stage: str, rates: dict | None,
                     dup: dict | None, ident: dict | None,
                     usable: dict, pf_summary: dict) -> list[dict]:
    recs: list[dict] = []

    def rec(action, why):
        recs.append({"action": action, "why": why})

    u = usable["usable"]
    dup_rate = (rates or {}).get("duplicate_rate") or (dup or {}).get("rate")
    blur_rate = (rates or {}).get("blur_rate")
    pf_rate = (rates or {}).get("prefilter_rate")

    if pf_rate and pf_rate > 0.10:
        rec("run `prefilter` before curation",
            f"~{pf_rate:.0%} of sampled files are ingestion junk "
            "(tiny/low-res/heavy-compression) - rejecting them first keeps "
            "curation timing and stats honest")
    if pf_summary:
        total_pf = sum(pf_summary.values())
        rec("review prefilter rejections",
            f"{total_pf} frame(s) currently rejected by the prefilter "
            f"({', '.join(f'{k}:{v}' for k, v in sorted(pf_summary.items()))})"
            " - `prefilter --restore` brings any of them back")

    if dup_rate is not None and dup_rate > 0.35:
        rec("keep dedup strict (hamming <= 4)",
            f"duplicate pressure is high (~{dup_rate:.0%}) - typical of "
            "video-heavy archives; dedup does the heavy lifting here")
    if blur_rate is not None and blur_rate > 0.25:
        rec("keep --min-sharpness at the default 35",
            f"~{blur_rate:.0%} of frames already fall below it - raising "
            "the bar further would cost too much data")

    if u and u < 300:
        rec("filter conservatively (--min-sharpness 25, hamming 3)",
            f"usable pool is small (~{u}); every frame counts more than "
            "marginal quality gains")
    elif u > 3000:
        rec("filter aggressively (--min-sharpness 45+, hamming 5-6)",
            f"usable pool is large (~{u}); trading volume for quality is "
            "nearly free at this scale")

    if stage in ("raw", "basic_curated") and ident is None:
        rec("run `curate --smart` before export",
            "no identity data yet - identity filtering and framing signals "
            "unlock the preset recommender and health checks")
    if ident and ident["mean_identity_sim"] < 0.45:
        rec("review the identity anchor set",
            f"mean identity similarity {ident['mean_identity_sim']:.2f} is "
            "weak - anchors may not represent the subject well")
    if stage == "captioned":
        rec("you're export-ready: `build` then `suggest-preset`",
            "captions present; the parameter advisor prints after build")
    return recs


def _training_outlook(usable: dict) -> dict:
    """Bridge the usable estimate to the Phase 4 step bands."""
    try:
        from .train.param_advisor import _band_for
    except Exception:                                # pragma: no cover
        def _band_for(_):
            return (1800, 2600)
    u = max(0, usable["usable"])
    out = {}
    for family in ("character", "style", "detail"):
        lo, hi = _band_for(family)
        mid = (lo + hi) // 2
        if u == 0:
            out[family] = {"band": [lo, hi], "verdict": "no usable frames"}
            continue
        repeats = max(1, min(40, round(mid / max(1, u * 10))))
        steps = u * repeats * 10
        if steps < lo * 0.7:
            verdict = (f"thin even at max repeats (~{steps} steps vs "
                       f"band {lo}-{hi})")
        elif u * 10 > hi:
            verdict = (f"comfortable at repeats 1 (epochs will be trimmed "
                       f"into the {lo}-{hi} band)")
        else:
            verdict = f"comfortable at repeats {repeats} (~{steps} steps)"
        out[family] = {"band": [lo, hi], "repeats": repeats,
                       "verdict": verdict}
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def analyze_dataset(conn, output_base=None, sample: int = 0,
                    curate_cfg: CurateConfig | None = None,
                    prefilter_cfg: PrefilterConfig | None = None,
                    seed: int = 42) -> dict:
    """Holistic dataset analysis. ``sample > 0`` enables scoring a random
    sample of 'new' frames when the archive is still raw; ``sample == 0``
    stays DB-only (cheap enough to run after every curate)."""
    base = Path(output_base) if output_base else Path(".")
    ccfg = curate_cfg or CurateConfig(output_base=base)
    pcfg = prefilter_cfg or PrefilterConfig(output_base=base)

    warnings: list[str] = []
    inv = _inventory(conn)
    if inv["total_frames"] == 0:
        return {"inventory": inv, "stage": "empty",
                "usable_estimate": {"usable": 0, "low": 0, "high": 0,
                                    "basis": "no frames"},
                "recommendations": [{"action": "run extract/photos first",
                                     "why": "the manifest has no frames"}],
                "warnings": []}
    stage = _stage(inv, conn)

    rates = _scored_rates(conn, ccfg)
    if rates is None and stage == "raw" and sample > 0:
        try:
            rates = _sample_rates(conn, sample, ccfg, pcfg, seed=seed)
        except Exception as exc:                     # pragma: no cover
            warnings.append(f"sampling failed: {exc}")
    elif rates is None and stage == "raw":
        warnings.append("archive is raw and no sample was requested - "
                        "usable estimate is a floor; rerun with "
                        "--sample 300 for a projection")

    dup = _dup_pressure(conn, inv)
    ident = _identity(conn)
    try:
        from .prefilter import prefilter_summary
        pf_summary = prefilter_summary(conn)
    except Exception:                                # pragma: no cover
        pf_summary = {}
    usable = _usable_estimate(inv, rates, dup)
    recs = _recommendations(inv, stage, rates, dup, ident, usable,
                            pf_summary)
    return {
        "inventory": inv,
        "stage": stage,
        "rates": rates,
        "duplication": dup,
        "identity": ident,
        "prefilter": pf_summary,
        "usable_estimate": usable,
        "training_outlook": _training_outlook(usable),
        "recommendations": recs,
        "warnings": warnings,
    }


def render_intelligence(report: dict) -> str:
    inv = report["inventory"]
    lines = ["", "DATASET INTELLIGENCE", "=" * 60,
             f"Stage: {report['stage']} | frames: {inv['total_frames']}"]
    if inv.get("sources"):
        lines.append("Sources: " + ", ".join(
            f"{k}: {v}" for k, v in sorted(inv["sources"].items())))
    top = sorted(inv["by_status"].items(), key=lambda kv: -kv[1])[:8]
    lines.append("Status:  " + ", ".join(f"{k}: {v}" for k, v in top))

    rates = report.get("rates")
    if rates:
        parts = [f"{k.replace('_rate', '')} {v:.0%}"
                 for k, v in rates.items()
                 if k.endswith("_rate") and isinstance(v, float)]
        lines.append(f"Reject rates ({rates['basis']}): "
                     + (", ".join(parts) or "none"))
        if rates.get("sharpness_median") is not None:
            lines.append(f"Sharpness median: {rates['sharpness_median']}")
    if report.get("duplication"):
        d = report["duplication"]
        lines.append(f"Duplicate pressure: {d['rate']:.0%} "
                     f"({d['duplicates']} frames)")
    if report.get("identity"):
        i = report["identity"]
        lines.append(f"Identity: {i['frames_with_identity']} frames, "
                     f"mean sim {i['mean_identity_sim']}")
    if report.get("prefilter"):
        lines.append("Prefilter rejections: " + ", ".join(
            f"{k}: {v}" for k, v in sorted(report["prefilter"].items())))

    u = report["usable_estimate"]
    lines += ["-" * 60,
              f"Usable-after-curation estimate: ~{u['usable']} "
              f"({u['low']}-{u['high']})  [{u['basis']}]"]
    outlook = report.get("training_outlook", {})
    for family, o in outlook.items():
        lines.append(f"  {family:<10} band {o['band'][0]}-{o['band'][1]}: "
                     f"{o['verdict']}")
    if report.get("recommendations"):
        lines += ["-" * 60, "Strategy:"]
        for r in report["recommendations"]:
            lines.append(f"  * {r['action']}")
            lines.append(f"    why: {r['why']}")
    for w in report.get("warnings", []):
        lines.append(f"  ! {w}")
    return "\n".join(lines)


def intelligence_digest(report: dict, max_recs: int = 1) -> str:
    """Three-line digest appended to curate summaries."""
    u = report["usable_estimate"]
    lines = [f"Dataset intelligence: ~{u['usable']} usable frames "
             f"({u['low']}-{u['high']}), stage '{report['stage']}'"]
    for r in report.get("recommendations", [])[:max_recs]:
        lines.append(f"  -> {r['action']}")
    lines.append("  Full report: pixel-curator insights intelligence")
    return "\n".join(lines)


__all__ = ["analyze_dataset", "render_intelligence", "intelligence_digest"]
