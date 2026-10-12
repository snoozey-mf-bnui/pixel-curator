"""LoRA stack preflight validation - Feature 5.

The Test tab and wardrobe presets collect LoRA stack entries as free text,
one per line, in ``display_name:weight`` form. Display names may carry a
readability category prefix such as ``[SDXL-LORA-CLOTHING] ``. Forge/reForge
resolves a ``<lora:NAME:weight>`` tag by the on-disk file stem, so an entry
whose prefixed label does not correspond to a real file makes reForge resolve
the network to ``None`` and raise ``AttributeError: 'NoneType' object has no
attribute 'filename'`` during activation.

This module normalizes and validates a stack against the configured LoRA
directory BEFORE a payload is built, so only resolvable ``<lora:...>``
fragments are ever sent.

Important matching rule: some legitimate files genuinely contain a bracket
prefix in their filename (e.g. ``[SDXL-LORA-CLOTHING] See-Through
Clothing.safetensors``). The validator therefore tries each entry
**as-typed first**, then the bracket-stripped form - it never blindly strips
a prefix. The first candidate that matches a real file wins.

Pure standard library. Non-destructive (read-only scan + a JSON index
cache). Designed never to raise for ordinary validation problems - it skips
and warns instead.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("redresser.lora_validator")

DEFAULT_LORA_DIR = Path(
    "~/FORGE_REPO/stable-diffusion-webui-reForge/models/Lora").expanduser()
INDEX_FILENAME = ".redresser_lora_index.json"

# Absent weight uses this; a malformed weight is reset to MALFORMED_WEIGHT
# with a warning. Both documented for the tests.
DEFAULT_WEIGHT = 1.0
MALFORMED_WEIGHT = 0.5

_BRACKET_PREFIX = re.compile(r"^\s*\[[^\]]*\]\s*")
_MODEL_EXT = ".safetensors"
_INVALID_EXTS = (".ckpt", ".pt", ".pth", ".bin")


@dataclass
class LoraItem:
    display_name: str
    clean_name: str = ""          # resolved real filename (with extension)
    stem: str = ""                # filename without extension (what Forge uses)
    weight: float = DEFAULT_WEIGHT
    path: str = ""
    valid: bool = False


@dataclass
class _Parsed:
    base: str                     # name without the trailing :weight
    weight: float
    weight_warning: Optional[str] = None


def _split_weight(entry: str) -> _Parsed:
    """Split only the FINAL ``:<number>`` as the weight; everything before is
    the name (so paths/odd names with colons stay intact)."""
    text = entry.strip()
    head, sep, tail = text.rpartition(":")
    if sep and head.strip():
        try:
            return _Parsed(head.strip(), float(tail))
        except ValueError:
            # Trailing token after ':' is not a number. If it looks like an
            # attempted weight (short, no path/extension), treat it as a
            # malformed weight; otherwise the colon is part of the name.
            t = tail.strip()
            if t and len(t) <= 8 and "/" not in t and "." not in t \
                    and _MODEL_EXT not in t.lower():
                return _Parsed(head.strip(), MALFORMED_WEIGHT,
                               f"Invalid LoRA weight defaulted to "
                               f"{MALFORMED_WEIGHT:.2f}: {head.strip()}")
    return _Parsed(text, DEFAULT_WEIGHT)


def _candidate_names(base: str) -> list[str]:
    """Ordered candidate filenames to look for: as-typed first, then the
    bracket-stripped form; each coerced to a ``.safetensors`` filename.
    De-duplicated, order preserved."""
    out: list[str] = []
    for raw in (base.strip(), _BRACKET_PREFIX.sub("", base).strip()):
        if not raw:
            continue
        name = raw if raw.lower().endswith(_MODEL_EXT) else raw + _MODEL_EXT
        if name not in out:
            out.append(name)
    return out


class LoraStackValidator:
    """Validate LoRA stacks against a directory, with a JSON index cache."""

    def __init__(self, lora_dir: Path | str | None = None,
                 index_path: Path | str | None = None) -> None:
        self.lora_dir = Path(lora_dir or DEFAULT_LORA_DIR).expanduser()
        self.index_path = (Path(index_path).expanduser() if index_path
                           else self.lora_dir / INDEX_FILENAME)
        self._by_name: dict[str, dict] = {}        # filename -> record
        self._by_name_lower: dict[str, dict] = {}  # lowercase filename -> rec
        self._by_stripped: dict[str, dict] = {}    # prefix-stripped lower -> rec

    # -- index ---------------------------------------------------------------
    def _scan(self) -> list[dict]:
        files: list[dict] = []
        if not self.lora_dir.is_dir():
            log.warning("LoRA directory not found: %s", self.lora_dir)
            return files
        for p in sorted(self.lora_dir.rglob("*" + _MODEL_EXT)):
            # skip our own helper / quarantine folders if any remain
            if any(part.startswith(("_", ".")) for part in
                   p.relative_to(self.lora_dir).parts[:-1]):
                continue
            try:
                st = p.stat()
                files.append({"name": p.name, "stem": p.stem,
                              "path": str(p), "size": st.st_size,
                              "mtime": int(st.st_mtime)})
            except OSError:
                continue
        return files

    def refresh_index(self) -> dict:
        """Rescan the directory and write the JSON index cache."""
        files = self._scan()
        index = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "lora_dir": str(self.lora_dir), "files": files}
        try:
            self.index_path.write_text(json.dumps(index, indent=2),
                                       encoding="utf-8")
        except OSError as exc:
            log.warning("Could not write LoRA index %s: %s",
                        self.index_path, exc)
        self._load_maps(files)
        return index

    def _load_maps(self, files: list[dict]) -> None:
        self._by_name = {f["name"]: f for f in files}
        self._by_name_lower = {f["name"].lower(): f for f in files}
        # prefix-stripped index: lets an un-prefixed entry resolve to a real
        # file that carries a bracket prefix (first occurrence wins).
        self._by_stripped = {}
        for f in files:
            key = _BRACKET_PREFIX.sub("", f["name"]).strip().lower()
            self._by_stripped.setdefault(key, f)

    def _ensure_index(self) -> None:
        """Load the cache; fall back to a fresh scan on miss/corruption.
        Never raises - a corrupt cache must not break payload building."""
        if self._by_name:
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if (isinstance(data, dict)
                    and data.get("lora_dir") == str(self.lora_dir)
                    and isinstance(data.get("files"), list)):
                self._load_maps(data["files"])
                return
            log.info("LoRA index stale or for a different directory - "
                     "refreshing.")
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Unreadable LoRA index (%s) - rescanning.", exc)
        self.refresh_index()

    # -- validation ----------------------------------------------------------
    def _match(self, base: str) -> tuple[Optional[dict], str]:
        """Resolve a parsed base name to an on-disk record. Returns
        ``(record, how)`` where how is 'exact' (matched the name as-typed),
        'normalized' (matched after bracket-strip / case / extension /
        prefix coercion), or '' (no match)."""
        cands = _candidate_names(base)
        # 1) exact filename, as-typed (first candidate is the as-typed form)
        for i, cand in enumerate(cands):
            rec = self._by_name.get(cand)
            if rec:
                return rec, ("exact" if i == 0 else "normalized")
        # 2) case-insensitive filename
        for cand in cands:
            rec = self._by_name_lower.get(cand.lower())
            if rec:
                return rec, "normalized"
        # 3) symmetric prefix-strip: un-prefixed entry vs prefixed real file
        for cand in cands:
            key = _BRACKET_PREFIX.sub("", cand).strip().lower()
            rec = self._by_stripped.get(key)
            if rec:
                return rec, "normalized"
        return None, ""

    def validate_lora_stack(
            self, stack_list: list[str]) -> tuple[dict, list[str]]:
        """Normalize + validate a list of ``display_name:weight`` strings.

        Returns ``(cleaned, warnings)`` where ``cleaned`` has ``items``
        (every parsed entry, with a ``valid`` flag), ``prompt_fragment``
        (only valid entries), and ``total_weight`` (sum of valid weights).

        De-duplication policy: when the same resolved stem appears more than
        once, the HIGHEST weight is kept and a warning is emitted.
        """
        self._ensure_index()
        warnings: list[str] = []
        items: list[LoraItem] = []
        by_stem: dict[str, int] = {}            # stem -> index in items

        for entry in stack_list or []:
            if not str(entry).strip():
                continue
            parsed = _split_weight(str(entry))
            if parsed.weight_warning:
                warnings.append(parsed.weight_warning)
                log.warning(parsed.weight_warning)
            item = LoraItem(display_name=str(entry).strip(),
                            weight=parsed.weight)

            # reject a non-safetensors model extension outright
            low = _BRACKET_PREFIX.sub("", parsed.base).strip().lower()
            if low.endswith(_INVALID_EXTS):
                msg = f"Invalid LoRA extension skipped: {parsed.base.strip()}"
                warnings.append(msg)
                log.warning(msg)
                items.append(item)
                continue

            rec, how = self._match(parsed.base)
            if not rec:
                clean = _candidate_names(parsed.base)
                msg = (f"Skipped missing asset: "
                       f"{clean[-1] if clean else parsed.base.strip()}")
                warnings.append(msg)
                log.warning(msg)
                items.append(item)
                continue

            item.clean_name = rec["name"]
            item.stem = rec["stem"]
            item.path = rec["path"]
            item.valid = True
            # warn only when real normalization happened (prefix/case/ext),
            # not merely because the entry carried a ':weight' suffix
            if how == "normalized":
                msg = (f"Normalized label: {item.display_name} -> "
                       f"{rec['name']}")
                warnings.append(msg)
                log.info(msg)

            # de-duplicate on resolved stem, keeping the highest weight
            if item.stem in by_stem:
                prev = items[by_stem[item.stem]]
                kept = max(prev.weight, item.weight)
                msg = (f"Duplicate LoRA skipped: {item.clean_name} "
                       f"(kept weight {kept:g})")
                warnings.append(msg)
                log.warning(msg)
                prev.weight = kept
                item.valid = False
                items.append(item)
                continue
            by_stem[item.stem] = len(items)
            items.append(item)

        valid = [i for i in items if i.valid]
        fragment = " ".join(f"<lora:{i.stem}:{i.weight:g}>" for i in valid)
        cleaned = {
            "items": [i.__dict__ for i in items],
            "valid_pairs": [(i.stem, i.weight) for i in valid],
            "prompt_fragment": fragment,
            "total_weight": round(sum(i.weight for i in valid), 4),
        }
        return cleaned, warnings


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def validate_lora_stack(
        stack_list: list[str],
        lora_dir: Path | str | None = None,
        index_path: Path | str | None = None) -> tuple[dict, list[str]]:
    """Module-level wrapper around :class:`LoraStackValidator`."""
    return LoraStackValidator(lora_dir, index_path).validate_lora_stack(
        stack_list)


def normalize_stack_text(stack_text: str) -> list[str]:
    """Split a Test-tab stack field (newline and/or comma separated) into
    individual entries."""
    parts = [s.strip() for s in (stack_text or "").replace(",", "\n")
             .splitlines()]
    return [s for s in parts if s]
