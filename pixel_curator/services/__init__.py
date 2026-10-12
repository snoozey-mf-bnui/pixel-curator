"""Shared runtime services for the CLI pipeline.

These modules back the ``pixel_curator`` subcommands (training jobs, status
writers, Florence captioning, merge services, ...) so no single command
owns the logic.  Everything here is import-safe: no torch, no HEIF, no
network at import time.
"""
from __future__ import annotations

__all__ = [
    "model_architecture", "media_support", "training_jobs", "log_streaming",
    "training_runtime", "model_registry", "merge_service", "status_writer",
]
