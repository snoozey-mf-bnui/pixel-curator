"""Scorer registry structural tests (import probes only - offline)."""

from __future__ import annotations

from pixel_curator.curate.scorer_registry import SCORERS, list_scorers, render_scorers


def test_registry_keys_unique_and_mapped():
    assert len(SCORERS) >= 25
    for key, entry in SCORERS.items():
        assert entry.key == key
        assert entry.module.startswith("pixel_curator.")
        assert entry.purpose
        assert entry.stage


def test_list_scorers_covers_registry():
    rows = list_scorers()
    assert {r["key"] for r in rows} == set(SCORERS)
    for r in rows:
        assert isinstance(r["available"], bool)
        assert r["available"] or r["reason"]


def test_core_scorers_always_available():
    rows = {r["key"]: r for r in list_scorers()}
    # pure-numpy/PIL scorers must never be unavailable
    assert rows["sharpness"]["available"]
    assert rows["duplicates"]["available"]


def test_render_scorers_groups_by_stage():
    text = render_scorers()
    assert "[core]" in text and "sharpness" in text
