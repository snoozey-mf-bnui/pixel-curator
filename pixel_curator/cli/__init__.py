"""Command-line interface (package).

The CLI is split into:

* :mod:`.parser`    - argument parser construction
* :mod:`.emitter`   - live progress output
* :mod:`.dispatch`  - context building and routing
* :mod:`.commands`  - command handlers, one module per domain
"""

from __future__ import annotations

from .. import manifest
from .dispatch import run as _run
from .emitter import get_emitter


def main(argv=None) -> None:
    """Entry point with professional handling for a busy manifest.

    While a training run is writing to the manifest, commands surface a
    clear, actionable message instead of a raw sqlite traceback."""
    import sqlite3 as _sqlite3
    try:
        _run(argv)
    except KeyboardInterrupt:
        # The heavy stages (extract/photos/curate/caption/quality) commit in
        # batches and skip finished work on rerun, so an interrupt only ever
        # costs the current batch.
        get_emitter().close()
        print("\nInterrupted - progress through the last batch is saved "
              "in the manifest.\nRerun the same command to resume where "
              "you left off (pipelines: pixel-curator pipeline resume).")
        raise SystemExit(130)
    except manifest.ManifestLocked as exc:
        print(f"\n{exc}")
        raise SystemExit(2)
    except _sqlite3.OperationalError as exc:
        if "locked" not in str(exc).lower():
            raise
        print("\nThe manifest database is temporarily locked - an active "
              "process (typically a training run) is writing to it.\n"
              "Read-only commands (runs, status, insights) work during "
              "training; retry this command in a moment or after the "
              "training run finishes.")
        raise SystemExit(2)


__all__ = ["main"]
