"""Visual review + reporting layer (zero-dependency static HTML).

Two halves of a review round-trip:

* ``build_review_report``  - writes ``index.html`` (thumbnail grid, unified
  scores, status/flag chips, pure-CSS score histogram, per-cluster
  breakdown) plus ``review_sheet.csv``. The HTML references original files
  via ``file://`` URLs - nothing is copied, so it works on 100k-image
  libraries. Open it in any browser to *see* what got dropped and why.
* ``import_review_decisions`` - reads back ``review_sheet.csv`` after the
  user fills the ``decision`` column (keep/reject) and applies it to the
  manifest (``status`` kept / quarantined). Dry-run by default.

Stdlib-only; never raises for ordinary input problems.
"""
from __future__ import annotations

import csv
import html
import logging
from collections import Counter
from pathlib import Path

from .general import (
    _unified_scores,
    iter_frame_records,
    select_records,
    select_records_balanced,
)

log = logging.getLogger("pixel_curator.export.review")

_CSS = """
body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:24px;
background:#fafafa;color:#222}
h1{font-size:22px} h2{font-size:16px;margin-top:32px}
.summary{color:#555;margin-bottom:16px}
.hist{display:flex;align-items:flex-end;gap:3px;height:120px;
margin:12px 0 4px}
.hist .bar{background:#4a7dd4;flex:1;min-width:8px;border-radius:2px 2px 0 0;
position:relative}
.hist .bar span{position:absolute;bottom:-18px;left:0;right:0;
font-size:10px;color:#777;text-align:center}
.histwrap{margin-bottom:28px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
gap:12px}
.card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;
font-size:12px}
.card img{width:100%;height:140px;object-fit:cover;border-radius:4px;
background:#eee}
.score{font-weight:600}
.chip{display:inline-block;padding:1px 6px;border-radius:8px;font-size:10px;
margin:2px 2px 0 0;background:#eee}
.chip.kept{background:#d4edda}.chip.dropped,.chip.quarantined,
.chip.rejected{background:#f8d7da}.chip.selected{background:#cce5ff}
.cap{color:#666;margin-top:4px;max-height:34px;overflow:hidden}
.cluster{margin-top:24px}
"""


def _histogram(records: list[dict], buckets: int = 10) -> str:
    counts = [0] * buckets
    for r in records:
        b = min(buckets - 1, max(0, int(float(r.get("score") or 0.0)
                                     * buckets)))
        counts[b] += 1
    peak = max(counts) or 1
    bars = "".join(
        f'<div class="bar" style="height:{100 * c / peak:.0f}%">'
        f"<span>{i / buckets:.1f}</span></div>"
        for i, c in enumerate(counts))
    return f'<div class="histwrap"><div class="hist">{bars}</div>' \
           f'<div class="summary">score distribution (0..1, ' \
           f'{len(records)} frames)</div></div>'


def _card(rec: dict) -> str:
    path = str(rec.get("path") or "")
    src = "file://" + html.escape(path)
    status = str(rec.get("status") or "")
    score = rec.get("score")
    cap = html.escape(str(rec.get("caption_text") or ""))[:140]
    chips = f'<span class="chip {html.escape(status)}">' \
            f'{html.escape(status)}</span>'
    if rec.get("cluster_id") is not None:
        chips += f'<span class="chip">cluster {rec["cluster_id"]}</span>'
    if rec.get("identity_sim") is not None:
        chips += (f'<span class="chip">identity '
                  f'{float(rec["identity_sim"]):.2f}</span>')
    return (f'<div class="card"><img loading="lazy" src="{src}" alt="">'
            f'<div class="score">score {float(score or 0):.3f}</div>'
            f'{chips}<div class="cap">{cap}</div></div>')


def build_review_report(conn, out_dir: Path | str,
                        top_pct: float | None = None,
                        min_score: float | None = None,
                        balanced: int = 0,
                        max_cards: int = 2000) -> dict:
    """Write index.html + review_sheet.csv for the selected frame set
    (default: every non-dropped frame). Returns a report dict."""
    records = list(iter_frame_records(conn))
    _unified_scores(records)
    if top_pct is None and min_score is None and not balanced:
        shown = sorted(records, key=lambda r: r.get("score", 0.0),
                       reverse=True)
    elif balanced:
        shown = select_records_balanced(records, balanced, top_pct,
                                        min_score)
    else:
        shown = select_records(records, top_pct, min_score)

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    status_counts = Counter(str(r.get("status") or "") for r in records)
    cards = "\n".join(_card(r) for r in shown[:max_cards])
    summary = (f"{len(records)} frames total | showing {min(len(shown), max_cards)}"
               f" | status: "
               + ", ".join(f"{k} {v}" for k, v in status_counts.most_common()))
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>pixel-curator review</title>
<style>{_CSS}</style></head><body>
<h1>pixel-curator review</h1>
<div class="summary">{html.escape(summary)}</div>
{_histogram(records)}
<h2>Frames (best first)</h2>
<div class="grid">{cards}</div>
<hr><div class="summary">To act on this review: fill the
<b>decision</b> column in <b>review_sheet.csv</b> (keep / reject), then run
<code>pixel-curator review import --sheet review_sheet.csv --apply</code>.
</div>
</body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")

    sheet = out / "review_sheet.csv"
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame_id", "path", "score",
                                           "status", "caption_text",
                                           "decision"])
        w.writeheader()
        for r in shown:
            w.writerow({"frame_id": r.get("frame_id"),
                        "path": r.get("path"),
                        "score": r.get("score"),
                        "status": r.get("status"),
                        "caption_text": str(r.get("caption_text") or "")[:200],
                        "decision": ""})
    return {"ok": True, "out_dir": str(out), "html": str(out / "index.html"),
            "sheet": str(sheet), "frames_total": len(records),
            "frames_shown": min(len(shown), max_cards)}


def import_review_decisions(conn, sheet_path: Path | str,
                            apply: bool = False) -> dict:
    """Apply keep/reject decisions from a filled review_sheet.csv.

    keep   -> status 'kept'
    reject -> status 'quarantined'  (recoverable; harder than 'dropped')
    Blank/unknown decisions are left untouched. Dry-run unless apply=True.
    """
    sheet = Path(sheet_path).expanduser()
    if not sheet.is_file():
        return {"ok": False, "error": f"sheet not found: {sheet}"}
    keeps, rejects, skipped, unknown_ids = [], [], 0, 0
    with open(sheet, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fid = (row.get("frame_id") or "").strip()
            dec = (row.get("decision") or "").strip().lower()
            if not fid:
                unknown_ids += 1
                continue
            if dec == "keep":
                keeps.append(fid)
            elif dec == "reject":
                rejects.append(fid)
            else:
                skipped += 1
    out = {"ok": True, "sheet": str(sheet), "keep": len(keeps),
           "reject": len(rejects), "skipped": skipped,
           "rows_without_frame_id": unknown_ids, "apply": bool(apply)}
    if apply and (keeps or rejects):
        from ..util import now_iso
        ts = now_iso()
        conn.executemany(
            "UPDATE frames SET status='kept', scored_at=? WHERE frame_id=?",
            [(ts, f) for f in keeps])
        conn.executemany(
            "UPDATE frames SET status='quarantined', scored_at=? "
            "WHERE frame_id=?", [(ts, f) for f in rejects])
        conn.commit()
        out["applied"] = True
    return out
