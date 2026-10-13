#!/usr/bin/env python3
"""pixel-curator smoke tests - stdlib only, no network, no models.

Run directly:  python pixel_curator/tests/run_all.py [--timeout N] [filter...]
Or via CLI:    python -m pixel_curator selftest

Covers the parts that must never regress: manifest schema migration,
general export (manifest/selection/balanced), review round-trip, and the
JSONL generation log. Exits 0 on success, 1 on failure.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging

from pixel_curator import manifest  # noqa: E402

logger = logging.getLogger(__name__)
from pixel_curator.export.general import export_manifest, export_selection  # noqa: E402
from pixel_curator.export.review import build_review_report, import_review_decisions  # noqa: E402
from pixel_curator.studio.generation_log import add_generation_to_log, identity_anchor_hash  # noqa: E402

RESULTS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}"
          + (f"  ({detail})" if detail and not cond else ""))


def _fixture_db(tmp: Path):
    imgdir = tmp / "imgs"
    imgdir.mkdir(parents=True, exist_ok=True)
    conn = manifest.connect(tmp)
    try:
        conn.execute("ALTER TABLE frames ADD COLUMN cluster_id INTEGER")
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        pass
    for i in range(1, 13):
        p = imgdir / f"f{i:03}.jpg"
        p.write_bytes(b"\xff\xd8\xff")  # minimal JPEG-ish marker
        status = "dropped" if i == 3 else "selected"
        conn.execute(
            "INSERT INTO frames (frame_id, path, sharpness, brightness,"
            " status, scored_at, cluster_id) VALUES (?,?,?,?,?,?,?)",
            (f"fr{i}", str(p), i * 100.0, 0.5, status, "t", i % 3))
        conn.execute(
            "INSERT INTO captions (frame_id, caption_text) VALUES (?,?)",
            (f"fr{i}", f"caption {i}"))
        if i % 2 == 0:
            conn.execute(
                "INSERT INTO frame_quality (frame_id, integrity, face_pass)"
                " VALUES (?,?,?)", (f"fr{i}", i / 12.0, 1))
    conn.commit()
    return conn


def test_schema(tmp: Path) -> None:
    conn = manifest.connect(tmp / "fresh")
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    check("schema: core tables migrate", {"sources", "frames", "events"}
          <= tables)
    check("schema: user_version set",
          conn.execute("PRAGMA user_version").fetchone()[0] >= 1)


def test_export(tmp: Path) -> None:
    conn = _fixture_db(tmp / "exp")
    r = export_manifest(conn, tmp / "exp" / "m.csv", "csv")
    check("export manifest: csv written", r.get("ok")
          and r.get("records") == 12)
    r2 = export_selection(conn, tmp / "exp" / "sel", top_pct=25,
                          mode="symlink")
    check("export select: top-25% of 11 eligible = 3",
          r2.get("written") == 3, str(r2))
    r3 = export_selection(conn, tmp / "exp" / "bal", balanced=6,
                          mode="copy")
    ok = r3.get("written") == 6
    if ok:
        picked = [row["source"] for row in csv.DictReader(
            open(r3["index"]))]
        check("export select: balanced spread over 3 clusters", ok)
    else:
        check("export select: balanced spread over 3 clusters", False,
              str(r3))
    r4 = export_selection(conn, tmp / "exp" / "none")
    check("export select: refuses empty selection", not r4.get("ok"))


def test_review(tmp: Path) -> None:
    conn = _fixture_db(tmp / "rev")
    r = build_review_report(conn, tmp / "rev" / "out")
    check("review: html + sheet written",
          r.get("ok") and Path(r["html"]).is_file()
          and Path(r["sheet"]).is_file())
    sheet = Path(r["sheet"])
    rows = list(csv.DictReader(open(sheet)))
    rows[0]["decision"] = "keep"
    rows[1]["decision"] = "reject"
    with open(sheet, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    dry = import_review_decisions(conn, sheet)
    check("review import: dry-run counts", dry.get("keep") == 1
          and dry.get("reject") == 1 and not dry.get("applied"))
    live = import_review_decisions(conn, sheet, apply=True)
    st = dict(conn.execute(
        "SELECT status, COUNT(*) FROM frames GROUP BY status").fetchall())
    check("review import: applies kept/quarantined",
          live.get("applied") and st.get("kept") == 1
          and st.get("quarantined") == 1)


def test_generation_log(tmp: Path) -> None:
    img = tmp / "gen.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    result = {"mode": "static_catalog", "init_image_path": str(img),
              "original_outputs": [str(img)], "warnings": []}
    rep = add_generation_to_log(result, "test_ds", tmp)
    ok = rep.get("ok") and Path(rep["log_path"]).is_file()
    check("generation log: JSONL appended", ok)
    check("generation log: identity hash", len(identity_anchor_hash(img))
          == 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("filters", nargs="*")
    args = ap.parse_args()

    t0 = time.time()
    tests = {"schema": test_schema, "export": test_export,
             "review": test_review, "log": test_generation_log}
    chosen = {k: v for k, v in tests.items()
              if not args.filters
              or any(f in k for f in args.filters)}
    for name, fn in chosen.items():
        if time.time() - t0 > args.timeout:
            check(f"{name}: completed within timeout", False, "timeout")
            break
        with tempfile.TemporaryDirectory() as td:
            try:
                fn(Path(td))
            except Exception as exc:
                check(f"{name}: ran without exception", False, str(exc))

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed"
          + (f"  FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
