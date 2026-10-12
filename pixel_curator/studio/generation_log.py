"""Generation logging - local JSONL dataset log (CLI-first, no GUI).

Replaces the retired FiftyOne integration with a plain append-only JSONL
log: every generated catalog image / sequence frame is recorded as one
JSON line with stable metadata fields. The log can be inspected with any
text tool, re-imported by scripts, or exported via the CLI. No optional
dependencies; never raises for ordinary I/O problems.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("redresser.generation_log")

DEFAULT_DATASET = "personal_catalog_dataset"

# Metadata fields attached to every record (kept stable for tooling).
METADATA_FIELDS = (
    "source_image_path", "output_path", "mode", "wardrobe_preset",
    "lora_stack", "validated_lora_names", "prompt", "negative_prompt",
    "identity_anchor_hash", "post_processed_path", "sequence_id",
    "frame_index", "garment_tags", "controlnet_enabled",
    "ip_adapter_enabled", "post_processing_enabled", "created_at", "warnings",
)


def identity_anchor_hash(image_path: Path | str) -> str:
    """Safe LOCAL file hash of the personal identity reference image (sha256,
    first 16 hex). Returns '' if the file is unreadable."""
    try:
        p = Path(image_path).expanduser()
        if not p.is_file():
            return ""
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return ""


def build_sample_metadata(result: dict, output_path: str = "",
                          frame_index: Optional[int] = None) -> dict:
    """Assemble the per-record metadata dict from an orchestrator result."""
    from ..util import now_iso
    payload = result.get("payload") or {}
    vl = result.get("validated_loras") or {}
    valid_names = [stem for stem, _ in vl.get("valid_pairs", [])]
    return {
        "source_image_path": str(result.get("init_image_path", "")),
        "output_path": str(output_path),
        "mode": result.get("mode", ""),
        "wardrobe_preset": result.get("wardrobe_preset", ""),
        "lora_stack": result.get("lora_stack", []),
        "validated_lora_names": valid_names,
        "prompt": payload.get("prompt", "") if isinstance(payload, dict)
        else "",
        "negative_prompt": result.get("payload", {}).get("negative_prompt", "")
        if isinstance(result.get("payload"), dict) else "",
        "identity_anchor_hash": result.get("identity_anchor_hash", ""),
        "post_processed_path": "",
        "sequence_id": result.get("sequence_id", ""),
        "frame_index": frame_index if frame_index is not None else -1,
        "garment_tags": result.get("garment_tags", []),
        "controlnet_enabled": bool(result.get("controlnet_enabled")),
        "ip_adapter_enabled": bool(result.get("ip_adapter_enabled")),
        "post_processing_enabled": bool(result.get("post_processing_enabled")),
        "created_at": now_iso(),
        "warnings": list(result.get("warnings", [])),
    }


def default_log_path(output_base: Path | str = "outputs",
                     dataset_name: str = DEFAULT_DATASET) -> Path:
    """Location of the JSONL log for a dataset name."""
    return Path(output_base).expanduser() / "dataset_logs" / \
        f"{dataset_name}.jsonl"


def add_generation_to_log(result: dict,
                          dataset_name: str = DEFAULT_DATASET,
                          output_base: Path | str = "outputs") -> dict:
    """Append a generation result's outputs (images + sequence frames) to the
    local JSONL dataset log. Returns a structured report; never raises for
    ordinary I/O problems."""
    outputs = list(result.get("post_processed_outputs")
                   or result.get("original_outputs") or [])
    seq = list(result.get("sequence_outputs") or [])
    all_paths = [p for p in outputs + seq if p]
    metadata = [build_sample_metadata(
        result, output_path=str(p),
        frame_index=(i if seq and p in seq else None))
        for i, p in enumerate(all_paths)]

    path = default_log_path(output_base, dataset_name)
    added = 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for meta in metadata:
                fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
                added += 1
    except Exception as exc:
        log.warning("Could not write dataset log %s: %s", path, exc)
        return {"ok": False, "added": 0, "dataset": dataset_name,
                "log_path": str(path), "metadata": metadata,
                "warning": f"Dataset log write failed: {exc}"}
    return {"ok": True, "added": added, "dataset": dataset_name,
            "log_path": str(path), "metadata": metadata}
