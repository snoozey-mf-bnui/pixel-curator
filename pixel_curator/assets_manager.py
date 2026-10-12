"""Feature 6B - verified asset manager (allowlist-only, local-first).

Manages a small, explicit allowlist of verified SFW fashion / catalog /
garment LoRA assets. It will only ever place a file that is named in the
allowlist, and only from the source configured for that entry. It never
scrapes, never crawls, never resolves unknown assets, and never stores new
credentials - it relies on any Hugging Face / Git authentication already
present in the environment.

Network access happens only when the user explicitly requests a download of
an allowlisted asset that has a configured source. Placeholder entries
(``OWNER/REPO``) and ``local_only`` entries never trigger a download; they
return a professional instruction to place the file manually.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .lora_validator import DEFAULT_LORA_DIR, LoraStackValidator

log = logging.getLogger("redresser.assets_manager")

_MODEL_EXT = ".safetensors"
_PLACEHOLDER_REPOS = {"OWNER/REPO", "owner/repo", ""}

# Verified, SFW fashion asset allowlist. Repo IDs are placeholders until the
# user configures verified sources; placeholder entries never download.
VERIFIED_FASHION_ASSETS: dict[str, dict] = {
    "professional_identity_reference_v1.safetensors": {
        "role": "personal_identity_anchor",
        "source_type": "local_only",
        "path_hint": None,
        "description": "Personal identity anchor for private self-modeling "
                       "workflows.",
    },
    "structured_garment_layer_v2.safetensors": {
        "role": "garment_rendering",
        "source_type": "huggingface",
        "hf_repo_id": "OWNER/REPO",
        "filename": "structured_garment_layer_v2.safetensors",
        "description": "Verified garment layering asset for SFW fashion "
                       "catalog styling.",
    },
    "sheer_fabric_styling_v1.safetensors": {
        "role": "fabric_interaction",
        "source_type": "huggingface",
        "hf_repo_id": "OWNER/REPO",
        "filename": "sheer_fabric_styling_v1.safetensors",
        "description": "Verified fabric interaction asset for SFW catalog "
                       "styling.",
    },
    "fabric_detail_enhancement_v3.safetensors": {
        "role": "fabric_detail_enhancement",
        "source_type": "local_only",
        "path_hint": None,
        "description": "Verified fabric detail enhancement asset for SFW "
                       "catalog styling.",
    },
}


@dataclass
class AssetResult:
    name: str
    status: str                     # available | placed | skipped | rejected
    path: Optional[str] = None
    message: str = ""


class VerifiedAssetManager:
    """Allowlist-bounded asset placement into the configured LoRA directory."""

    def __init__(self, lora_dir: Path | str | None = None,
                 allowlist_path: Path | str | None = None) -> None:
        self.lora_dir = Path(lora_dir or DEFAULT_LORA_DIR).expanduser()
        self.allowlist = dict(VERIFIED_FASHION_ASSETS)
        if allowlist_path:
            self._merge_allowlist(Path(allowlist_path).expanduser())

    def _merge_allowlist(self, path: Path) -> None:
        try:
            extra = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(extra, dict):
                self.allowlist.update(extra)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read allowlist %s: %s", path, exc)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _norm(name: str) -> str:
        n = (name or "").strip()
        return n if n.lower().endswith(_MODEL_EXT) else n + _MODEL_EXT

    def is_verified(self, name: str) -> bool:
        return self._norm(name) in self.allowlist

    def local_path(self, name: str) -> Optional[Path]:
        p = self.lora_dir / self._norm(name)
        return p if p.is_file() else None

    def refresh_index(self) -> int:
        """Refresh the LoRA index after a placement; returns file count."""
        idx = LoraStackValidator(self.lora_dir).refresh_index()
        return len(idx.get("files", []))

    # -- core ----------------------------------------------------------------
    def ensure_lora_available(self, lora_name: str,
                              hf_repo_id: Optional[str] = None,
                              github_url: Optional[str] = None
                              ) -> Optional[Path]:
        """Ensure an allowlisted verified asset is present in ``lora_dir``.

        Returns the local path if available/placed, else ``None`` with a
        professional warning. Downloads only allowlisted assets that have a
        configured (non-placeholder) source.
        """
        name = self._norm(lora_name)

        existing = self.local_path(name)
        if existing:
            log.info("Verified asset available locally: %s", name)
            return existing

        entry = self.allowlist.get(name)
        if entry is None:
            log.warning("Asset is not in the verified fashion asset "
                        "allowlist; download skipped: %s", name)
            return None

        source_type = str(entry.get("source_type", "local_only"))
        if source_type == "local_only":
            log.warning("Verified asset has no configured source; please "
                        "place it in the LoRA directory manually: %s", name)
            return None

        repo_id = hf_repo_id or entry.get("hf_repo_id")
        if source_type == "huggingface":
            if not repo_id or repo_id in _PLACEHOLDER_REPOS:
                log.warning("Verified asset has a placeholder source; "
                            "configure a repository or place it manually: "
                            "%s", name)
                return None
            return self._download_hf(name, repo_id,
                                     entry.get("filename", name))

        log.warning("Verified asset source type '%s' is not supported; "
                    "place it manually: %s", source_type, name)
        return None

    def _download_hf(self, name: str, repo_id: str,
                     filename: str) -> Optional[Path]:
        try:
            from huggingface_hub import hf_hub_download
        except Exception:
            log.warning("Hugging Face helper unavailable; download skipped: "
                        "%s", name)
            return None
        try:
            self.lora_dir.mkdir(parents=True, exist_ok=True)
            cached = hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as exc:                   # pragma: no cover - network
            log.warning("Verified download did not complete for %s: %s",
                        name, exc)
            return None
        src = Path(cached)
        if src.suffix.lower() != _MODEL_EXT:
            log.warning("Downloaded asset is not a %s file; rejected: %s",
                        _MODEL_EXT, name)
            return None
        dest = self.lora_dir / name
        try:
            shutil.copy2(src, dest)
        except OSError as exc:                      # pragma: no cover
            log.warning("Could not place verified asset %s: %s", name, exc)
            return None
        if not dest.is_file():
            log.warning("Verified asset placement could not be confirmed: "
                        "%s", name)
            return None
        self.refresh_index()
        log.info("Verified asset placed and index refreshed: %s", name)
        return dest

    def download_missing(self, names: Optional[list[str]] = None
                         ) -> list[AssetResult]:
        """Check (and, where configured, place) verified assets. Without an
        explicit list, all allowlist entries are checked. Reports per asset;
        placeholder/local-only entries are reported as skipped, not errors."""
        targets = [self._norm(n) for n in (names or self.allowlist.keys())]
        results: list[AssetResult] = []
        for name in targets:
            local = self.local_path(name)
            if local:
                results.append(AssetResult(name, "available", str(local),
                                           "Verified asset available "
                                           "locally."))
                continue
            if not self.is_verified(name):
                results.append(AssetResult(name, "rejected", None,
                                           "Not in the verified fashion "
                                           "asset allowlist."))
                continue
            placed = self.ensure_lora_available(name)
            if placed:
                results.append(AssetResult(name, "placed", str(placed),
                                           "Verified asset placed."))
            else:
                results.append(AssetResult(
                    name, "skipped", None,
                    "No configured/verified source; place manually."))
        return results


def ensure_lora_available(lora_name: str, hf_repo_id: Optional[str] = None,
                          github_url: Optional[str] = None,
                          lora_dir: Path | str | None = None
                          ) -> Optional[Path]:
    """Module-level wrapper around :class:`VerifiedAssetManager`."""
    return VerifiedAssetManager(lora_dir).ensure_lora_available(
        lora_name, hf_repo_id, github_url)
