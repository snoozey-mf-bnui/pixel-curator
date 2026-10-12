"""Cross-run evaluation leaderboard.

Ranks candidate LoRAs (epochs, shards, merges) against each other using the
likeness scores stored in ``evals``. The point is fairness: two candidates are
only comparable on prompts they were both evaluated on, so the headline ranking
uses the intersection of (category, seed) prompt cells that every candidate
shares. Overall means are reported alongside for context, and ``head_to_head``
gives a direct win/loss on shared cells.

Read-only. Pure-Python aggregation (no numpy dependency) so it stays testable
and cheap. Pairs naturally with eval.likeness, which populates evals.likeness.
"""
from __future__ import annotations

import sqlite3
from typing import Optional


def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def candidate_labels(conn: sqlite3.Connection,
                     labels: Optional[list] = None) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT label FROM evals WHERE label IS NOT NULL "
        "AND TRIM(label) <> ''").fetchall()
    found = sorted(r[0] for r in rows)
    if labels:
        want = set(labels)
        found = [l for l in found if l in want]
    return found


def _load_rows(conn: sqlite3.Connection, labels: list[str]) -> list[tuple]:
    if not labels:
        return []
    qs = ",".join("?" for _ in labels)
    return conn.execute(
        f"SELECT label, category, seed, likeness FROM evals "
        f"WHERE label IN ({qs})", labels).fetchall()


def build_leaderboard(conn: sqlite3.Connection, *,
                      labels: Optional[list] = None) -> dict:
    """Rank candidates. Headline metric is mean likeness on the set of
    (category, seed) prompt cells shared by all candidates; overall mean is
    also reported. Returns a structured board (read-only)."""
    labels = candidate_labels(conn, labels)
    if not labels:
        return {"rows": [], "categories": [], "n_candidates": 0,
                "shared_cells": 0, "warning": "no labelled evals found"}

    rows = _load_rows(conn, labels)

    # per-label accumulators
    by_label: dict[str, dict] = {l: {"all": [], "by_cat": {}, "cells": {}}
                                 for l in labels}
    categories: set = set()
    for label, cat, seed, lk in rows:
        d = by_label[label]
        categories.add(cat)
        if lk is not None:
            d["all"].append(lk)
            d["by_cat"].setdefault(cat, []).append(lk)
        # cell key for fair comparison (only cells that have a score count)
        if lk is not None:
            d["cells"].setdefault((cat, seed), []).append(lk)

    # shared (category, seed) cells across every candidate that has any scores
    scored_labels = [l for l in labels if by_label[l]["cells"]]
    if scored_labels:
        shared = set.intersection(
            *[set(by_label[l]["cells"].keys()) for l in scored_labels])
    else:
        shared = set()

    board = []
    for label in labels:
        d = by_label[label]
        shared_vals = []
        for key in shared:
            shared_vals.extend(d["cells"].get(key, []))
        board.append({
            "label": label,
            "n_evals": sum(len(v) for v in d["cells"].values())
                        or len([1 for r in rows if r[0] == label]),
            "n_scored": len(d["all"]),
            "overall_likeness": _mean(d["all"]),
            "shared_likeness": _mean(shared_vals) if shared else None,
            "by_category": {c: _mean(v) for c, v in sorted(d["by_cat"].items())},
        })

    # rank: shared metric first when it exists, else overall; unscored last
    def _key(r):
        primary = r["shared_likeness"] if shared else r["overall_likeness"]
        return (primary is not None, primary if primary is not None else -1.0)

    board.sort(key=_key, reverse=True)
    for i, r in enumerate(board, 1):
        r["rank"] = i

    return {"rows": board, "categories": sorted(c for c in categories if c),
            "n_candidates": len(labels), "shared_cells": len(shared),
            "ranked_on": "shared" if shared else "overall", "warning": None}


def head_to_head(conn: sqlite3.Connection, label_a: str, label_b: str) -> dict:
    """Direct comparison on the (category, seed) prompt cells both candidates
    were scored on. Returns win/loss/tie counts and the mean likeness delta."""
    rows = _load_rows(conn, [label_a, label_b])
    cells = {label_a: {}, label_b: {}}
    for label, cat, seed, lk in rows:
        if lk is not None:
            cells[label].setdefault((cat, seed), []).append(lk)
    shared = set(cells[label_a]) & set(cells[label_b])
    wins_a = wins_b = ties = 0
    deltas = []
    for key in shared:
        a = _mean(cells[label_a][key])
        b = _mean(cells[label_b][key])
        deltas.append(a - b)
        if abs(a - b) < 1e-6:
            ties += 1
        elif a > b:
            wins_a += 1
        else:
            wins_b += 1
    winner = (label_a if wins_a > wins_b else
              label_b if wins_b > wins_a else "tie")
    return {"a": label_a, "b": label_b, "shared_cells": len(shared),
            "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
            "mean_delta_a_minus_b": _mean(deltas), "winner": winner}


def render_leaderboard(board: dict) -> str:
    if board.get("warning"):
        return f"Leaderboard: {board['warning']}"
    cats = board["categories"]
    lines = ["", "Cross-run eval leaderboard",
             "=" * 64,
             f"{board['n_candidates']} candidates  |  ranked on "
             f"{board['ranked_on']} likeness  |  {board['shared_cells']} "
             "shared prompt cell(s)"]
    header = f"{'#':>2}  {'candidate':<26} {'shared':>7} {'overall':>8} {'n':>5}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in board["rows"]:
        sh = "-" if r["shared_likeness"] is None else f"{r['shared_likeness']:.3f}"
        ov = "-" if r["overall_likeness"] is None else f"{r['overall_likeness']:.3f}"
        lines.append(f"{r['rank']:>2}  {r['label']:<26} {sh:>7} {ov:>8} "
                     f"{r['n_scored']:>5}")
    if board["shared_cells"] == 0 and board["n_candidates"] > 1:
        lines.append("\nNote: candidates share no common (category, seed) "
                     "prompt cells - ranking falls back to overall mean, which "
                     "is not a like-for-like comparison. Run the same eval "
                     "matrix (same seeds) on each candidate for a fair board.")
    return "\n".join(lines)
