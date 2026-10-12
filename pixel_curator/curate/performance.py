"""Priority 5 - curation performance: device detection, det_size tuning and
lightweight stage profiling.

All helpers are local-first, torch-optional and safe on CPU-only machines:
nothing here imports torch at module load, forces MPS, or changes scoring. The
profiler is a near-zero-overhead accumulator that does nothing unless a caller
explicitly enables it.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

# Face-detector det_size values that the SCRFD backend handles cleanly. Larger
# = more accurate on small/distant faces but slower; smaller = faster passes for
# large libraries. 640 is the existing default and stays the default.
ALLOWED_DET_SIZES = (320, 480, 512, 640, 768, 1024)
DEFAULT_DET_SIZE = 640

# Conservative thread/worker bounds so a misconfigured value can't oversubscribe.
MIN_WORKERS = 1
MAX_WORKERS = 32


def detect_compute_device(prefer_mps: bool = True) -> dict:
    """Report the best available compute device without requiring torch.

    Returns ``{"torch_available", "mps_available", "cuda_available",
    "selected_device", "warnings"}``. Never raises; if torch is missing the
    selected device is ``"cpu"`` and a warning explains why.
    """
    info = {"torch_available": False, "mps_available": False,
            "cuda_available": False, "selected_device": "cpu",
            "warnings": []}
    try:
        from .. import runtime
    except Exception as exc:                              # pragma: no cover
        info["warnings"].append(f"runtime import failed: {exc}")
        return info
    try:
        info["torch_available"] = runtime._torch() is not None
        info["mps_available"] = bool(runtime.is_mps_available())
        info["cuda_available"] = bool(runtime.is_cuda_available())
    except Exception as exc:                              # pragma: no cover
        info["warnings"].append(f"device probe failed: {exc}")
    if not info["torch_available"]:
        info["warnings"].append(
            "torch not installed; curation runs on CPU/onnxruntime providers")
        info["selected_device"] = "cpu"
        return info
    if info["cuda_available"]:
        info["selected_device"] = "cuda"
    elif prefer_mps and info["mps_available"]:
        info["selected_device"] = "mps"
    else:
        info["selected_device"] = "cpu"
        if prefer_mps and not info["mps_available"]:
            info["warnings"].append("MPS unavailable; using CPU")
    return info


def validate_det_size(value, *, default: int = DEFAULT_DET_SIZE) -> tuple:
    """Coerce/validate a face-detector ``det_size``.

    Accepts an int or a square ``(w, h)`` tuple. Returns ``(size_int, warnings)``
    - an unknown value falls back to ``default`` with a warning rather than
    raising, so an odd CLI value can never abort a curation run.
    """
    warnings: list = []
    if value is None:
        return default, warnings
    if isinstance(value, (tuple, list)):
        if len(value) == 2 and value[0] == value[1]:
            value = value[0]
        else:
            warnings.append(f"non-square det_size {tuple(value)} not supported; "
                            f"using {default}")
            return default, warnings
    try:
        size = int(value)
    except (TypeError, ValueError):
        warnings.append(f"invalid det_size {value!r}; using {default}")
        return default, warnings
    if size not in ALLOWED_DET_SIZES:
        warnings.append(
            f"det_size {size} is not one of {ALLOWED_DET_SIZES}; using {default}")
        return default, warnings
    return size, warnings


def clamp_workers(value, *, default: int = 1) -> tuple:
    """Clamp a worker count into a safe range. Returns ``(workers, warnings)``."""
    warnings: list = []
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default, [f"invalid worker count {value!r}; using {default}"]
    if n < MIN_WORKERS:
        warnings.append(f"worker count {n} < {MIN_WORKERS}; using {MIN_WORKERS}")
        n = MIN_WORKERS
    elif n > MAX_WORKERS:
        warnings.append(f"worker count {n} > {MAX_WORKERS}; using {MAX_WORKERS}")
        n = MAX_WORKERS
    return n, warnings


# ---------------------------------------------------------------------------
# Stage profiler
# ---------------------------------------------------------------------------
STAGE_KEYS = ("image_load_ms", "orientation_ms", "face_detection_ms",
              "quality_signals_ms", "advanced_quality_ms", "manifest_write_ms")


@dataclass
class PerfProfiler:
    """Near-zero-overhead per-stage timing accumulator.

    When ``enabled`` is False every call is a no-op, so the profiler can be
    threaded through hot loops unconditionally. Timings are accumulated in
    milliseconds; ``summary`` reports totals, per-frame averages and the
    slowest stage.
    """
    enabled: bool = False
    frames: int = 0
    totals_ms: dict = field(default_factory=lambda:
                            {k: 0.0 for k in STAGE_KEYS})
    total_frame_ms: float = 0.0
    _t0: float = 0.0

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - start) * 1000.0
            self.totals_ms[name] = self.totals_ms.get(name, 0.0) + dt

    def add(self, name: str, ms: float) -> None:
        if self.enabled:
            self.totals_ms[name] = self.totals_ms.get(name, 0.0) + float(ms)

    def frame_begin(self) -> None:
        if self.enabled:
            self._t0 = time.perf_counter()

    def frame_end(self) -> None:
        if self.enabled:
            self.total_frame_ms += (time.perf_counter() - self._t0) * 1000.0
            self.frames += 1

    def per_frame(self) -> dict:
        """Latest accumulated totals reshaped as the ``performance`` block."""
        out = {k: round(self.totals_ms.get(k, 0.0), 3) for k in STAGE_KEYS}
        out["total_frame_ms"] = round(self.total_frame_ms, 3)
        return out

    def summary(self, *, det_size=None, workers=None, batch_size=None,
                device=None) -> dict:
        n = max(1, self.frames)
        total_s = self.total_frame_ms / 1000.0
        avgs = {f"avg_{k.replace('_ms', '')}_ms": round(self.totals_ms.get(k, 0.0)
                / n, 3) for k in STAGE_KEYS}
        slowest = max(self.totals_ms.items(), key=lambda kv: kv[1],
                      default=(None, 0.0))[0] if any(self.totals_ms.values()) \
            else None
        return {
            "frames_processed": self.frames,
            "total_seconds": round(total_s, 3),
            "frames_per_second": round(self.frames / total_s, 3)
            if total_s > 0 else 0.0,
            "avg_face_detection_ms": avgs.get("avg_face_detection_ms", 0.0),
            "avg_quality_ms": avgs.get("avg_quality_signals_ms", 0.0),
            "avg_advanced_quality_ms": avgs.get("avg_advanced_quality_ms", 0.0),
            "slowest_stage": slowest,
            "stage_totals_ms": {k: round(v, 3)
                                for k, v in self.totals_ms.items()},
            "det_size": det_size,
            "workers": workers,
            "batch_size": batch_size,
            "selected_device": device,
        }


__all__ = [
    "ALLOWED_DET_SIZES", "DEFAULT_DET_SIZE", "STAGE_KEYS",
    "detect_compute_device", "validate_det_size", "clamp_workers",
    "PerfProfiler",
]
