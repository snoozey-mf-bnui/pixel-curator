"""Unit tests for project config template/load round-trip."""

from __future__ import annotations

from pixel_curator.config import DB_NAME, DEFAULT_QUOTA, UI_PORT, load_project, write_template


def test_template_roundtrip(tmp_path):
    path = tmp_path / "proj.toml"
    write_template(path)
    assert path.exists()
    prj = load_project(str(path))
    assert prj is not None
    assert hasattr(prj, "output_base")


def test_load_project_missing_raises_clear_error(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError, match="Project file not found"):
        load_project(str(tmp_path / "nope.toml"))


def test_constants_sane():
    assert DB_NAME.endswith(".sqlite3")
    assert isinstance(UI_PORT, int) and 1024 < UI_PORT < 65535
    quotas = [float(p.split("=")[1]) for p in DEFAULT_QUOTA.split(",")]
    assert abs(sum(quotas) - 1.0) < 1e-6
