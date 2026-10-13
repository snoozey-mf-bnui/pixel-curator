"""Model asset registry tests (offline - no downloads)."""

from __future__ import annotations

import pytest

from pixel_curator.model_assets import (
    MODEL_REGISTRY,
    asset_path,
    ensure_model,
    is_installed,
    status,
)


def test_registry_keys_unique_and_complete():
    assert len(MODEL_REGISTRY) >= 7
    for key, asset in MODEL_REGISTRY.items():
        assert asset.key == key
        assert asset.kind in ("hf", "open_clip", "insightface", "url")
        assert asset.purpose
        if asset.kind == "url":
            assert asset.url.startswith("https://")
            assert asset.dest
        if asset.kind == "hf":
            assert "/" in asset.repo_id


def test_status_is_offline_and_complete():
    rows = status()
    assert {r["key"] for r in rows} == set(MODEL_REGISTRY)
    for r in rows:
        assert isinstance(r["installed"], bool)
        assert isinstance(r["size_mb"], int)


def test_url_assets_resolve_under_models_dir():
    from pixel_curator.model_assets import MODELS_DIR
    for asset in MODEL_REGISTRY.values():
        if asset.kind == "url":
            p = asset_path(asset)
            assert p is not None and str(p).startswith(str(MODELS_DIR))


def test_unknown_key_raises_keyerror():
    with pytest.raises(KeyError, match="unknown model"):
        ensure_model("does-not-exist", progress=lambda *_: None)


def test_installed_url_asset_roundtrip(tmp_path, monkeypatch):
    """A file already in place is reported installed without network."""
    from pixel_curator import model_assets
    fake = tmp_path / "realesrgan" / "RealESRGAN_x4plus.pth"
    fake.parent.mkdir(parents=True)
    fake.write_bytes(b"weights")
    monkeypatch.setattr(model_assets, "MODELS_DIR", tmp_path)
    asset = MODEL_REGISTRY["realesrgan-x4plus"]
    assert is_installed(asset)
    assert asset_path(asset) == fake
    # ensure_model on an installed asset returns immediately, no network
    assert ensure_model("realesrgan-x4plus",
                        progress=lambda *_: None) == fake
