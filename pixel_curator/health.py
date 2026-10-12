"""Dataset health report - a read-only snapshot of the manifest.

Surfaces dataset quality *before* you commit GPU to training: how frames split
by status, the framing balance of survivors, the identity-similarity and
sharpness distributions, duplicate-cluster load, and source completion. Pure
read - it never writes to the manifest or touches source media. Output is a
single self-contained HTML file (inline CSS bars, no external deps, works
offline) plus a JSON sidecar.

CLI:  pixel-curator health [--open]
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

from . import manifest
from .config import DB_NAME, Project


def _safe(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list:
    try:
        return conn.execute(sql, params).fetchall()
    except Exception:
        return []


def _histogram(values: list[float], lo: float, hi: float, bins: int = 10) -> list[tuple[str, int]]:
    """Bin values into [lo, hi] deciles; out-of-range clamps to the edges."""
    span = (hi - lo) or 1.0
    counts = [0] * bins
    for v in values:
        idx = int((float(v) - lo) / span * bins)
        counts[max(0, min(bins - 1, idx))] += 1
    out = []
    for i, c in enumerate(counts):
        a = lo + span * i / bins
        b = lo + span * (i + 1) / bins
        out.append((f"{a:.2f}-{b:.2f}", c))
    return out


def build_health(output_base: Path) -> dict:
    """Read-only manifest summary. Never raises; missing tables -> empty pieces."""
    output_base = Path(output_base).expanduser()
    db = output_base / DB_NAME
    summary: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_base": str(output_base),
        "manifest": str(db),
        "warnings": [],
    }
    if not db.exists():
        summary["warnings"].append(f"no manifest at {db}")
        return summary

    conn = manifest.connect(output_base, readonly=True) if _accepts_ro() \
        else manifest.connect(output_base)

    # sources by kind + status
    src: dict = {}
    for kind, status, n in _safe(conn,
            "SELECT kind, status, COUNT(*) FROM sources GROUP BY kind, status"):
        src.setdefault(kind, {})[status] = n
    summary["sources"] = src

    # frame status breakdown
    fstatus = {s: n for s, n in
               _safe(conn, "SELECT status, COUNT(*) FROM frames GROUP BY status")}
    summary["frame_status"] = fstatus
    summary["frames_total"] = sum(fstatus.values())
    summary["selected"] = fstatus.get("selected", 0)
    summary["unscanned_new"] = fstatus.get("new", 0)
    summary["rejected"] = {k: v for k, v in fstatus.items() if k.startswith("rejected")}

    # framing balance of survivors
    summary["framing"] = {(fr or "none"): n for fr, n in _safe(conn,
        "SELECT d.framing, COUNT(*) FROM detections d JOIN frames f "
        "ON f.frame_id=d.frame_id WHERE f.status='selected' GROUP BY d.framing")}

    # identity-similarity distribution (over scored faces)
    ids = [r[0] for r in _safe(conn,
        "SELECT identity_sim FROM detections WHERE identity_sim IS NOT NULL") if r[0] is not None]
    summary["scored_faces"] = len(ids)
    summary["identity_hist"] = _histogram(ids, 0.0, 1.0) if ids else []
    summary["identity_median"] = round(sorted(ids)[len(ids)//2], 3) if ids else None

    # sharpness distribution (raw Laplacian variance)
    sh = [r[0] for r in _safe(conn,
        "SELECT sharpness FROM frames WHERE sharpness IS NOT NULL") if r[0] is not None]
    if sh:
        sh_sorted = sorted(sh)
        summary["sharpness_min"] = round(sh_sorted[0], 1)
        summary["sharpness_median"] = round(sh_sorted[len(sh)//2], 1)
        summary["sharpness_max"] = round(sh_sorted[-1], 1)
        summary["sharpness_hist"] = _histogram(sh, 0.0, min(sh_sorted[-1], 400.0))

    # duplicate clusters
    dgroups = _safe(conn,
        "SELECT dup_group_id, COUNT(*) c FROM frames WHERE dup_group_id IS NOT NULL "
        "GROUP BY dup_group_id HAVING c > 1")
    summary["duplicates"] = {
        "groups": len(dgroups),
        "frames_in_groups": sum(c for _, c in dgroups),
    }

    try:
        conn.close()
    except Exception:
        pass
    return summary


def _accepts_ro() -> bool:
    try:
        import inspect
        return "readonly" in inspect.signature(manifest.connect).parameters
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# HTML rendering (inline CSS bars, offline-safe)
# --------------------------------------------------------------------------- #

def _bars(rows: list[tuple[str, int]], color: str = "#5b8def") -> str:
    if not rows:
        return "<p class='muted'>no data</p>"
    mx = max((c for _, c in rows), default=1) or 1
    out = ["<div class='bars'>"]
    for label, c in rows:
        w = int(100 * c / mx)
        out.append(
            f"<div class='row'><span class='lbl'>{label}</span>"
            f"<span class='bar'><span class='fill' style='width:{w}%;background:{color}'></span></span>"
            f"<span class='val'>{c:,}</span></div>")
    out.append("</div>")
    return "".join(out)


def render_html(s: dict) -> str:
    src_rows = []
    for kind, st in (s.get("sources") or {}).items():
        for status, n in st.items():
            src_rows.append((f"{kind} · {status}", n))
    framing_rows = sorted((s.get("framing") or {}).items(), key=lambda x: -x[1])
    reject_rows = sorted((s.get("rejected") or {}).items(), key=lambda x: -x[1])
    dup = s.get("duplicates") or {}
    warn = "".join(f"<li>{w}</li>" for w in s.get("warnings", []))

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Dataset Health</title><style>
body{{font:14px -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e9ef}}
.wrap{{max-width:960px;margin:0 auto;padding:28px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:15px;margin:26px 0 10px;color:#9fb0d0}}
.muted{{color:#7a849c}} .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}}
.card{{background:#171a21;border:1px solid #232838;border-radius:10px;padding:14px}}
.card .n{{font-size:24px;font-weight:600}} .card .k{{color:#8a93a8;font-size:12px;margin-top:2px}}
.bars .row{{display:flex;align-items:center;gap:10px;margin:4px 0}}
.lbl{{width:150px;color:#aeb7cc;font-size:12px;text-align:right;flex:none}}
.bar{{flex:1;background:#1d212b;border-radius:5px;overflow:hidden;height:16px}}
.fill{{display:block;height:100%}} .val{{width:64px;text-align:right;font-variant-numeric:tabular-nums;color:#cfd6e6}}
ul{{color:#d98b8b}}
</style></head><body><div class="wrap">
<h1>Dataset Health</h1>
<div class="muted">{s.get('generated_at','')} · {s.get('output_base','')}</div>
{('<ul>'+warn+'</ul>') if warn else ''}
<div class="grid">
  <div class="card"><div class="n">{s.get('frames_total',0):,}</div><div class="k">frames total</div></div>
  <div class="card"><div class="n">{s.get('selected',0):,}</div><div class="k">selected</div></div>
  <div class="card"><div class="n">{s.get('unscanned_new',0):,}</div><div class="k">unscanned (new)</div></div>
  <div class="card"><div class="n">{s.get('scored_faces',0):,}</div><div class="k">faces scored</div></div>
</div>
<h2>Sources</h2>{_bars(src_rows, '#6dd3a8')}
<h2>Selected framing balance</h2>{_bars(framing_rows, '#5b8def')}
<h2>Identity similarity (scored faces · median {s.get('identity_median','?')})</h2>{_bars(s.get('identity_hist',[]), '#c08bff')}
<h2>Sharpness (median {s.get('sharpness_median','?')})</h2>{_bars(s.get('sharpness_hist',[]), '#e0b15b')}
<h2>Rejections</h2>{_bars(reject_rows, '#e07a7a')}
<h2>Duplicates</h2><p class="muted">{dup.get('groups',0):,} near-duplicate groups covering {dup.get('frames_in_groups',0):,} frames (curate keeps the sharpest of each).</p>
</div></body></html>"""


def health_generator(prj: Project, output_base: Path,
                     open_after: bool = False) -> Generator[str, None, None]:
    """Build the report, write HTML + JSON to reports/, yield progress."""
    output_base = Path(output_base).expanduser()
    yield "Reading manifest (read-only)..."
    summary = build_health(output_base)
    for w in summary.get("warnings", []):
        yield f"WARN: {w}"

    reports = output_base / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = reports / f"health_{stamp}.html"
    json_path = reports / f"health_{stamp}.json"
    html_path.write_text(render_html(summary), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    yield (f"frames {summary.get('frames_total',0):,} | selected "
           f"{summary.get('selected',0):,} | unscanned {summary.get('unscanned_new',0):,} | "
           f"faces scored {summary.get('scored_faces',0):,}")
    yield f"HTML : {html_path}"
    yield f"JSON : {json_path}"

    if open_after:
        try:
            import subprocess
            subprocess.run(["open", str(html_path)], check=False)
        except Exception as exc:
            yield f"(could not auto-open: {exc})"
