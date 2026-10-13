"""Central logging configuration for pixel-curator.

Modules obtain loggers with::

    from .logutil import get_logger
    logger = get_logger(__name__)

The CLI configures the root ``pixel_curator`` logger once in
:func:`pixel_curator.cli.dispatch.run`. Library use (importing modules
directly) stays silent unless the host application configures logging.

Level resolution order: ``--verbose`` / ``PIXEL_CURATOR_DEBUG=1`` -> DEBUG,
otherwise INFO (WARNING under ``--quiet``).
"""

from __future__ import annotations

import logging
import os

_ROOT = "pixel_curator"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger under the pixel_curator namespace."""
    if name == "__main__":
        name = _ROOT
    return logging.getLogger(name)


def configure_logging(quiet: bool = False, verbose: bool = False) -> None:
    """Configure the package root logger (idempotent)."""
    global _configured
    if verbose or os.environ.get("PIXEL_CURATOR_DEBUG") == "1":
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    root = logging.getLogger(_ROOT)
    root.setLevel(level)
    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
        root.propagate = False
        _configured = True
    for h in root.handlers:
        h.setLevel(level)
