"""Phase 9: consistent progress reporting for long-running stages.

One lightweight tracker, zero dependencies. Generators keep yielding
strings; progress lines are a ``str`` subclass (:class:`ProgressLine`) so
the CLI can render them live (in-place ``\\r`` rewrite on a TTY, dropped
under ``--quiet``) while every other consumer - pipes, log files, the
pipeline orchestrator's ``[stage]`` prefixing - sees ordinary text.

Typical use inside a ``*_generator``::

    prog = ProgressTracker(len(rows), "scoring")
    for row in rows:
        ...work...
        line = prog.tick()
        if line:
            yield line
    yield prog.summary()
"""
from __future__ import annotations

import time
from typing import Optional


class ProgressLine(str):
    """A progress-class update. Purely a type marker: renderers may
    rewrite these in place or suppress them; any plain-string consumer
    treats them as normal text."""
    __slots__ = ()


def fmt_duration(seconds: float) -> str:
    """65 -> '1m05s'; 4000 -> '1h06m'; 12 -> '12s'."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def fmt_rate(rate: float) -> str:
    if rate >= 100:
        return f"{rate:.0f}/s"
    if rate >= 1:
        return f"{rate:.1f}/s"
    return f"{rate:.2f}/s" if rate > 0 else "-/s"


class ProgressTracker:
    """Throttled percent / rate / ETA progress for a known total.

    ``tick(n)`` returns a :class:`ProgressLine` when it is time to emit
    (every ``every`` items or ``min_interval`` seconds, whichever comes
    first, and always on the final item) - otherwise ``None``.
    """

    def __init__(self, total: int, label: str = "progress", *,
                 every: int = 200, min_interval: float = 5.0,
                 already_done: int = 0):
        self.total = max(0, int(total))
        self.label = label
        self.every = max(1, int(every))
        self.min_interval = float(min_interval)
        self.already_done = max(0, int(already_done))
        self.done = 0
        self.t0 = time.time()
        self._last_emit_t = self.t0
        self._last_emit_n = 0

    # -- formatting ---------------------------------------------------

    def _line(self) -> ProgressLine:
        elapsed = max(1e-3, time.time() - self.t0)
        rate = self.done / elapsed
        pct = (100.0 * self.done / self.total) if self.total else 0.0
        eta = ""
        if rate > 0 and self.total:
            remaining = max(0, self.total - self.done)
            eta = f" | ETA {fmt_duration(remaining / rate)}"
        return ProgressLine(
            f"{self.label} {self.done}/{self.total} ({pct:.1f}%) | "
            f"{fmt_rate(rate)}{eta}")

    # -- API ------------------------------------------------------------

    def tick(self, n: int = 1) -> Optional[ProgressLine]:
        self.done += int(n)
        now = time.time()
        due = (self.done - self._last_emit_n >= self.every
               or now - self._last_emit_t >= self.min_interval
               or (self.total and self.done >= self.total))
        if not due:
            return None
        self._last_emit_t = now
        self._last_emit_n = self.done
        return self._line()

    def resume_banner(self) -> Optional[str]:
        """'Resuming: N already done - M to go.' when prior work exists."""
        if not self.already_done:
            return None
        return (f"Resuming: {self.already_done} of "
                f"{self.already_done + self.total} already done - "
                f"{self.total} to go.")

    def summary(self, extra: str = "") -> str:
        elapsed = max(1e-3, time.time() - self.t0)
        rate = self.done / elapsed
        line = (f"{self.label} complete: {self.done} in "
                f"{fmt_duration(elapsed)} ({fmt_rate(rate)})")
        return line + (f" | {extra}" if extra else "")


__all__ = ["ProgressLine", "ProgressTracker", "fmt_duration", "fmt_rate"]
