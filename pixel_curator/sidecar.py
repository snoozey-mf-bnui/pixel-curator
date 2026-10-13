"""Google Photos Takeout sidecar-metadata reader.

Takeout exports a JSON sidecar per media file (``*.supplemental-metadata.json``
and truncated variants). The sidecars carry two signals worth using in the
curation pipeline:

  * ``people``          - Google's own face-recognition tags (e.g. the subject)
  * ``photoTakenTime``  - the true capture timestamp (diversity / val-split)

This module is intentionally dependency-free and defensive: every public
function tolerates missing files, malformed JSON, and the messy Takeout naming
(``...supplemental-meta.json`` truncation, trailing ``..json`` with an empty
suffix, and ``(1)`` duplicate markers). It never raises on bad input; it
returns structured results and accumulates warnings instead, so an optional
subsystem can never crash the pipeline.

Matching strategy (most-reliable first):
  1. the ``title`` field *inside* each JSON is the original media filename;
  2. the media name recovered by stripping the sidecar filename suffix.
Both are indexed (case-folded), with a ``(n)``-duplicate-stripped variant, so a
media file resolves even when Takeout renamed its sidecar.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Sidecar filename suffix: ".supplemental-metadata", truncated forms like
# ".supplemental-meta" / ".supplemental-met", an optional "(1)" duplicate
# marker, then ".json". Also covers the degenerate "<media>..json" case where
# the suffix was truncated to nothing.
_SIDECAR_SUFFIX_RE = re.compile(
    r"\.supplemental-met[a-z]*(?:\(\d+\))?\.json$", re.IGNORECASE
)
_DUP_MARKER_RE = re.compile(r"\s*\(\d+\)(?=\.[^.]+$|$)")


def media_name_from_sidecar(json_name: str) -> str:
    """Recover the media filename a sidecar JSON describes, from its name.

    Examples:
      IMG_3288.JPG.supplemental-metadata.json      -> IMG_3288.JPG
      IMG_3288.JPG.supplemental-metadata(1).json   -> IMG_3288.JPG
      RPReplay_Final17.mp4.supplemental-meta.json  -> RPReplay_Final17.mp4
      sstg-88D9....mp4..json                        -> sstg-88D9....mp4
    """
    name = json_name
    stripped = _SIDECAR_SUFFIX_RE.sub("", name)
    if stripped != name:
        return stripped
    # Degenerate truncation: "<media>..json" or "<media>.json" with no token.
    if name.lower().endswith(".json"):
        base = name[: -len(".json")]
        return base.rstrip(".")
    return name


def _norm(name: str) -> str:
    """Case-folded basename key for indexing/lookup."""
    return Path(name).name.casefold()


def _dedup_variant(name: str) -> str:
    """Drop a Google ``(n)`` duplicate marker: 'IMG (1).JPG' -> 'IMG.JPG'."""
    return _DUP_MARKER_RE.sub("", name)


def _norm_person(name: str) -> str:
    """Normalize a person name for tolerant comparison.

    'Michelle_Huang', 'Michelle Huang', 'michelle-huang' all collapse equal.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


@dataclass
class SidecarRecord:
    title: str
    people: tuple[str, ...] = ()
    taken_time: int | None = None        # epoch seconds, if known
    sources: tuple[str, ...] = ()           # sidecar json filenames merged in

    def merge(self, other: SidecarRecord) -> SidecarRecord:
        people = tuple(dict.fromkeys((*self.people, *other.people)))
        # earliest known capture time wins (most likely the original)
        times = [t for t in (self.taken_time, other.taken_time) if t]
        taken = min(times) if times else None
        return SidecarRecord(
            title=self.title or other.title, people=people, taken_time=taken,
            sources=tuple(dict.fromkeys((*self.sources, *other.sources))),
        )


@dataclass
class SidecarIndex:
    by_name: dict[str, SidecarRecord] = field(default_factory=dict)
    count: int = 0
    parsed: int = 0
    warnings: list[str] = field(default_factory=list)

    def _put(self, key: str, rec: SidecarRecord) -> None:
        key = key.casefold()
        if not key:
            return
        existing = self.by_name.get(key)
        self.by_name[key] = existing.merge(rec) if existing else rec

    def lookup(self, media_path) -> SidecarRecord | None:
        """Resolve a media file (path or name) to its sidecar record."""
        name = Path(media_path).name
        for key in (name, _dedup_variant(name)):
            rec = self.by_name.get(key.casefold())
            if rec:
                return rec
        return None


def parse_sidecar(json_path: Path) -> SidecarRecord | None:
    """Parse one sidecar JSON. Returns None if it is not a usable sidecar."""
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    title = str(data.get("title") or "").strip()

    people_raw = data.get("people") or []
    people: list[str] = []
    if isinstance(people_raw, list):
        for entry in people_raw:
            if isinstance(entry, dict) and entry.get("name"):
                people.append(str(entry["name"]).strip())

    taken = None
    pt = data.get("photoTakenTime") or {}
    if isinstance(pt, dict) and pt.get("timestamp"):
        try:
            taken = int(pt["timestamp"])
        except (TypeError, ValueError):
            taken = None

    # A sidecar with neither a title nor people nor a timestamp is useless.
    if not title and not people and taken is None:
        return None
    return SidecarRecord(
        title=title, people=tuple(people), taken_time=taken,
        sources=(Path(json_path).name,),
    )


def build_sidecar_index(metadata_dir) -> SidecarIndex:
    """Index every sidecar JSON under ``metadata_dir`` (recursive).

    Keyed (case-folded) by the JSON ``title`` and by the media name recovered
    from the sidecar filename, plus ``(n)``-stripped variants. Never raises.
    """
    idx = SidecarIndex()
    if not metadata_dir:
        return idx
    root = Path(metadata_dir).expanduser()
    if not root.exists():
        idx.warnings.append(f"metadata_dir not found: {root}")
        return idx

    for jp in root.rglob("*.json"):
        if not jp.is_file():
            continue
        idx.count += 1
        rec = parse_sidecar(jp)
        if rec is None:
            continue
        idx.parsed += 1
        keys = set()
        if rec.title:
            keys.add(_norm(rec.title))
            keys.add(_norm(_dedup_variant(rec.title)))
        fname = media_name_from_sidecar(jp.name)
        if fname:
            keys.add(_norm(fname))
            keys.add(_norm(_dedup_variant(fname)))
        for k in keys:
            idx._put(k, rec)
    return idx


def subject_present(rec: SidecarRecord | None, subject_names) -> bool:
    """True if any configured subject name appears in the record's people tags."""
    if rec is None or not subject_names:
        return False
    wanted = {_norm_person(s) for s in subject_names if s}
    if not wanted:
        return False
    return any(_norm_person(p) in wanted for p in rec.people)
