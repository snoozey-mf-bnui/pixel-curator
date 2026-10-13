"""Run the legacy stdlib smoke suite as a pytest test."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "run_all.py"


def test_smoke_suite():
    proc = subprocess.run(
        [sys.executable, str(RUNNER)],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "checks passed" in proc.stdout
