"""Live progress emitter for the CLI (TTY-aware, quiet mode, color)."""

from __future__ import annotations

import sys

from ..progress import ProgressLine

_ANSI = {"red": "\033[31m", "yellow": "\033[33m",
         "green": "\033[32m", "dim": "\033[2m", "reset": "\033[0m"}


class UpdateEmitter:
    """Prints generator updates.

    * :class:`ProgressLine` updates rewrite one live line on a TTY,
      print normally when piped, and vanish under ``--quiet``.
    * Warning / error / completion lines get minimal ANSI color on a
      TTY (suppressed by ``--no-color`` or the NO_COLOR env var).
    """

    def __init__(self, quiet: bool = False, color: bool | None = None,
                 stream=None):
        import os
        self.out = stream or sys.stdout
        self.quiet = quiet
        tty = bool(getattr(self.out, "isatty", lambda: False)())
        self.tty = tty
        self.color = (tty and not os.environ.get("NO_COLOR")
                      if color is None else color)
        self._live = False       # a \r progress line is on screen

    def _colorize(self, text: str) -> str:
        if not self.color:
            return text
        probe = text.lstrip()
        if probe.startswith(("FATAL", "ERROR")):
            return _ANSI["red"] + text + _ANSI["reset"]
        if probe.startswith(("WARNING", "!")):
            return _ANSI["yellow"] + text + _ANSI["reset"]
        if ("COMPLETE" in probe[:60] or "COMPLETED" in probe[:60]
                or probe.startswith(("DONE", "OK "))
                or " complete:" in probe[:80]):
            return _ANSI["green"] + text + _ANSI["reset"]
        return text

    def emit(self, update) -> None:
        if isinstance(update, ProgressLine):
            if self.quiet:
                return
            if self.tty:
                print("\r\033[2K  " + str(update), end="",
                      file=self.out, flush=True)
                self._live = True
                return
            print(str(update), file=self.out)
            return
        if self._live:
            print(file=self.out)      # finish the live line first
            self._live = False
        print(self._colorize(str(update)), file=self.out)

    def close(self) -> None:
        if self._live:
            print(file=self.out)
            self._live = False


_EMITTER = UpdateEmitter()


def stream(gen) -> None:
    """Drain a stage generator through the configured emitter."""
    try:
        for update in gen:
            _EMITTER.emit(update)
    finally:
        _EMITTER.close()


def get_emitter() -> UpdateEmitter:
    return _EMITTER


def set_emitter(e: UpdateEmitter) -> None:
    global _EMITTER
    _EMITTER = e
