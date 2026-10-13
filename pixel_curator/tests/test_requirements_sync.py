"""Requirements-file sync: requirements_*.txt must cover pyproject deps.

The two pinned environment files (CoreML / CUDA) are curated, not
generated - this test keeps them honest by asserting every direct
dependency declared in pyproject.toml is present in both files
(modulo the explicit skip list below).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[2]

# direct deps intentionally absent from the curated pin files
# (optional extras the reference environment does not install)
SKIPPED = {"pandas", "pyarrow", "accelerate", "diffusers",
           "hdbscan", "umap-learn"}

# pip name -> requirement-file name where they differ
ALIASES = {
    "Pillow": "pillow",
    "huggingface-hub": "huggingface_hub",
    "open-clip-torch": "open_clip_torch",
    "pillow-heif": "pillow_heif",
    "opencv-python-headless": "opencv-python-headless",
}


def _pyproject_deps() -> set[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = set(data["project"]["dependencies"])
    for name, extra in data["project"]["optional-dependencies"].items():
        if name in ("dev", "all"):   # tooling + aggregate, not runtime pins
            continue
        deps |= set(extra)
    names = set()
    for d in deps:
        name = re.split(r"[<>=!~;\[]", d)[0].strip()
        if name:
            names.add(ALIASES.get(name, name).lower())
    return names - {s.lower() for s in SKIPPED}


def _req_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        names.add(re.split(r"[<>=!~;\[]", line)[0].strip().lower())
    return names


@pytest.mark.parametrize("req", ["requirements_coreml.txt",
                                 "requirements_cuda.txt"])
def test_requirements_cover_pyproject_deps(req):
    req_path = ROOT / "pixel_curator" / req
    if not req_path.exists():
        pytest.skip(f"{req} not present")
    have = _req_names(req_path)
    missing = set()
    for dep in _pyproject_deps():
        if dep in have:
            continue
        # CUDA variant legitimately swaps these two
        if req.endswith("cuda.txt") and dep == "onnxruntime" \
                and "onnxruntime-gpu" in have:
            continue
        missing.add(dep)
    assert not missing, f"{req} missing pyproject deps: {sorted(missing)}"
