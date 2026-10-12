"""Phase 6: Studio Orchestration and Production Control Center.

A unified, read-only control layer over the Phase 1-5 workflow:
dataset readiness, run readiness, batch readiness, a unified asset
registry, a production state machine, automated next-step
recommendations and a production handoff exporter. Every read path uses
read-only manifest connections and degrades gracefully when metadata or
optional backends are missing - safe to run during active training.
"""

from .readiness import (batch_readiness, dataset_readiness,
                        readiness_level, run_readiness)
from .registry import build_asset_registry
from .state import build_studio_state, recommend_next_actions
from .handoff import export_studio_handoff
from .learning import (aggregate_preferences, ingest_directory,
                       load_preferences, rate_generation,
                       recommend_from_preferences, record_generation,
                       render_profile, reward, select_training_candidates,
                       stage_harvest)

__all__ = [
    "dataset_readiness", "run_readiness", "batch_readiness",
    "readiness_level", "build_asset_registry", "build_studio_state",
    "recommend_next_actions", "export_studio_handoff",
    "record_generation", "rate_generation", "aggregate_preferences",
    "load_preferences", "recommend_from_preferences", "render_profile",
    "reward", "select_training_candidates", "stage_harvest",
    "ingest_directory",
]
