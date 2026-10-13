"""Shared imports and dispatch context for CLI command handlers.

Command modules import from here so every handler sees the same names the
old monolithic ``cli.py`` dispatch scope provided.
"""

from __future__ import annotations

import base64  # noqa: F401  (used by some handlers)
import json  # noqa: F401
import sys  # noqa: F401
from collections.abc import Callable  # noqa: F401
from dataclasses import dataclass
from pathlib import Path  # noqa: F401

from .. import manifest  # noqa: F401
from ..caption import caption_generator  # noqa: F401
from ..config import (  # noqa: F401
    DB_NAME,
    DEFAULT_FPS_TIERS,
    DEFAULT_QUOTA,
    UI_PORT,
    CaptionConfig,
    ClusterConfig,
    CurateConfig,
    ExtractConfig,
    PackageConfig,
    PrefilterConfig,
    SmartCurateConfig,
    load_project,
    write_template,
)
from ..curate import (  # noqa: F401
    build_anchor,
    curate_generator,
    smart_curate_generator,
)
from ..curate.diversity import cluster_generator  # noqa: F401
from ..extract import pipeline_generator  # noqa: F401
from ..packager import package_generator  # noqa: F401
from ..progress import ProgressLine  # noqa: F401
from ..util import check_dependencies  # noqa: F401
from .emitter import UpdateEmitter, get_emitter, set_emitter, stream  # noqa: F401


@dataclass
class Context:
    """What every command handler receives."""

    args: object          # argparse.Namespace
    prj: object           # loaded project config
    out_base: Callable    # resolve --output flag against project default
