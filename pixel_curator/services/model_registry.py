"""Shared service - model registry with architecture awareness.

Discovers/annotates known models (LoRAs, checkpoints, merges) and tags each with
its architecture (sdxl / unknown). Both UI surfaces read this. Pure
filesystem + metadata; never downloads or opens model weights.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from . import model_architecture as arch

MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth")
LORA_HINT = ("lora", "lyco", "locon")
MERGED_HINT = ("merge", "merged")


@dataclass
class ModelRecord:
    model_id: str
    name: str = ""
    path: str = ""
    model_type: str = "unknown"   # lora | checkpoint | dreambooth | merged | unknown
    architecture: str = "unknown"
    base_model: str = ""
    training_type: str = ""
    created_at: str = ""
    source: str = "local"
    tags: list = field(default_factory=list)
    notes: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _infer_model_type(name: str) -> str:
    n = name.lower()
    if any(h in n for h in MERGED_HINT):
        return "merged"
    if any(h in n for h in LORA_HINT):
        return "lora"
    if "dreambooth" in n or n.endswith("_db.safetensors"):
        return "dreambooth"
    if n.endswith((".ckpt", ".safetensors")):
        return "checkpoint"
    return "unknown"


def record_from_path(path) -> ModelRecord:
    p = Path(path)
    mt = _infer_model_type(p.name)
    archk = arch.detect_model_architecture(p.name)
    try:
        created = datetime.fromtimestamp(
            p.stat().st_mtime).isoformat(timespec="seconds") \
            if p.exists() else ""
    except Exception:
        created = ""
    return ModelRecord(
        model_id=p.stem, name=p.name, path=str(p), model_type=mt,
        architecture=archk, created_at=created,
        metadata={"size": (p.stat().st_size if p.exists() else 0)})


def discover_models(*dirs, limit: int = 500) -> list:
    """Scan one or more directories for model files. Read-only; never raises."""
    out, seen = [], set()
    for d in dirs:
        try:
            root = Path(d).expanduser()
            if not root.is_dir():
                continue
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.suffix.lower() in MODEL_EXTS \
                        and str(p) not in seen:
                    seen.add(str(p))
                    out.append(record_from_path(p))
                    if len(out) >= limit:
                        return out
        except Exception:
            continue
    return out


def annotate_architecture(models: list) -> list:
    """Ensure every model dict/record carries an ``architecture`` key."""
    out = []
    for m in models:
        d = m.to_dict() if isinstance(m, ModelRecord) else dict(m)
        if not d.get("architecture") or d["architecture"] == "unknown":
            hint = d.get("path") or d.get("base_model") or d.get("name", "")
            d["architecture"] = arch.detect_model_architecture(hint)
        out.append(d)
    return out


def filter_models(models: list, *, architecture: str | None = None,
                  model_type: str | None = None) -> list:
    def _arch(m):
        return (m.architecture if isinstance(m, ModelRecord)
                else m.get("architecture", "unknown"))

    def _type(m):
        return (m.model_type if isinstance(m, ModelRecord)
                else m.get("model_type", "unknown"))
    out = list(models)
    if architecture and architecture not in ("", "any", "all"):
        ak = arch.normalize_architecture(architecture)
        out = [m for m in out if arch.normalize_architecture(_arch(m)) == ak]
    if model_type and model_type not in ("", "any", "all"):
        out = [m for m in out if _type(m) == model_type]
    return out


def registry_rows(models: list) -> list:
    rows = []
    for m in models:
        d = m.to_dict() if isinstance(m, ModelRecord) else dict(m)
        rows.append([d.get("model_id", ""), d.get("name", ""),
                     d.get("model_type", ""), d.get("architecture", ""),
                     d.get("created_at", "")])
    return rows


def write_registry_json(models: list, path) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() if isinstance(m, ModelRecord) else m
                for m in models]
        p.write_text(json.dumps({"models": data}, indent=2), encoding="utf-8")
    except Exception:
        pass


__all__ = ["ModelRecord", "MODEL_EXTS", "record_from_path", "discover_models",
           "annotate_architecture", "filter_models", "registry_rows",
           "write_registry_json"]
