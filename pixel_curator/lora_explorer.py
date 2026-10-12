"""Visual LoRA Explorer - professional browser for specialized LoRAs.

Scans configured LoRA folders, reads .safetensors metadata (stdlib header
parse - no dependency on the safetensors package), associates preview
thumbnails, and maintains per-LoRA *visual influence profiles* in JSON
sidecars (`<stem>.concept.json`) mirrored into the manifest (schema v8,
`lora_influence_profiles`) for fast UI queries.

Non-destructive: model files are only ever read.  Heavy analysis (image
embedding, aesthetics) is intentionally out of scope here - sidecars are
the extension point.
"""
from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .config import Project
from .util import now_iso

CONCEPT_FAMILIES = (
    "identity", "character", "wardrobe", "fashion", "lighting", "pose",
    "style", "texture", "detail", "environment", "composition", "camera",
    "refinement",
)

INFLUENCE_TAGS = (
    "identity_anchor", "silhouette", "garment_style", "fabric_texture",
    "lighting_mood", "pose_energy", "facial_consistency", "anatomy_balance",
    "color_palette", "scene_context", "detail_density", "composition_flow",
    "camera_perspective",
)

# conservative default weight ranges per family (weight intelligence)
FAMILY_WEIGHT_RANGES: dict[str, tuple[float, float]] = {
    "identity": (0.65, 0.85), "character": (0.65, 0.85),
    "wardrobe": (0.25, 0.55), "fashion": (0.25, 0.55),
    "style": (0.15, 0.35), "pose": (0.15, 0.35),
    "texture": (0.10, 0.30), "detail": (0.10, 0.30),
    "refinement": (0.10, 0.30),
    "lighting": (0.10, 0.35), "camera": (0.10, 0.35),
    "environment": (0.15, 0.35), "composition": (0.15, 0.35),
}

# identity risk by family: how strongly this concept tends to pull
# facial/character consistency when overdriven
FAMILY_IDENTITY_RISK: dict[str, str] = {
    "identity": "none", "character": "none",
    "wardrobe": "low", "fashion": "low", "detail": "low",
    "refinement": "low", "lighting": "low", "camera": "low",
    "texture": "medium", "pose": "medium", "environment": "medium",
    "composition": "medium", "style": "high",
}

# filename-keyword -> family inference (first match wins; sidecar overrides)
_NAME_HINTS: list[tuple[str, str]] = [
    ("character", "identity"), ("identity", "identity"),
    ("figure", "pose"), ("pose", "pose"),
    ("outfit", "wardrobe"), ("wardrobe", "wardrobe"),
    ("fashion", "fashion"), ("lingerie", "fashion"), ("garment", "fashion"),
    ("light", "lighting"), ("style", "style"), ("texture", "texture"),
    ("detail", "detail"), ("refine", "refinement"), ("camera", "camera"),
    ("env", "environment"), ("scene", "environment"),
    ("comp", "composition"),
]

_FAMILY_TAGS: dict[str, list[str]] = {
    "identity": ["identity_anchor", "facial_consistency"],
    "character": ["identity_anchor", "facial_consistency"],
    "wardrobe": ["garment_style", "silhouette"],
    "fashion": ["garment_style", "fabric_texture", "silhouette"],
    "lighting": ["lighting_mood"],
    "pose": ["pose_energy", "anatomy_balance"],
    "style": ["color_palette", "composition_flow"],
    "texture": ["fabric_texture", "detail_density"],
    "detail": ["detail_density"],
    "environment": ["scene_context"],
    "composition": ["composition_flow"],
    "camera": ["camera_perspective"],
    "refinement": ["detail_density", "anatomy_balance"],
}

SIDECAR_SUFFIX = ".concept.json"
LORA_EXTENSIONS = (".safetensors", ".pt", ".ckpt")
PREVIEW_LEVELS = ("default", "low", "medium", "high")
_PREVIEW_SUFFIXES = (".preview.png", ".preview.jpg", ".png", ".jpg", ".jpeg")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LoraInfluenceProfile:
    """Visual influence profile - what this LoRA tends to affect."""
    family: str = "style"
    influence_tags: list[str] = field(default_factory=list)
    weight_min: float = 0.15
    weight_max: float = 0.35
    weight_default: float = 0.25
    identity_risk: str = "medium"            # none | low | medium | high
    base_models: list[str] = field(default_factory=lambda: ["PONY"])
    compatible_families: list[str] = field(default_factory=list)
    known_conflicts: list[str] = field(default_factory=list)
    conflict_families: list[str] = field(default_factory=list)
    priority_hint: str = "normal"      # anchor|primary|normal|supporting|experimental
    notes: str = ""


@dataclass
class LoraCard:
    """UI-ready record for one LoRA."""
    lora_id: str
    path: str
    preview: Optional[str] = None
    modified_at: Optional[str] = None
    size_mb: float = 0.0
    network_dim: Optional[int] = None
    sd_metadata: dict = field(default_factory=dict)
    profile: LoraInfluenceProfile = field(default_factory=LoraInfluenceProfile)
    display_name: str = ""
    description: str = ""
    preview_levels: dict = field(default_factory=dict)   # level -> image path
    has_preview: bool = False
    metadata_source: Optional[str] = None    # sidecar path | "inferred"
    concept_meta: dict = field(default_factory=dict)     # normalized v2 metadata

    def to_json(self) -> dict:
        d = asdict(self)
        d["profile"] = asdict(self.profile)
        return d


# ---------------------------------------------------------------------------
# safetensors header (stdlib): 8-byte LE length + JSON header
# ---------------------------------------------------------------------------

def read_safetensors_metadata(path: Path,
                              max_header: int = 32 * 1024 * 1024) -> dict:
    """Read the `__metadata__` block of a .safetensors file safely.
    Returns {} on any irregularity - never raises, never loads tensors."""
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if not 0 < n <= max_header:
                return {}
            header = json.loads(fh.read(n))
        meta = header.get("__metadata__") or {}
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _network_dim(meta: dict) -> Optional[int]:
    for k in ("ss_network_dim", "network_dim"):
        try:
            return int(meta.get(k))
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Profiles: sidecar load/save + inference
# ---------------------------------------------------------------------------

def infer_profile(stem: str, meta: dict) -> LoraInfluenceProfile:
    """Baseline profile from filename + training metadata (sidecar wins)."""
    n = stem.lower()
    family = "style"
    for hint, fam in _NAME_HINTS:
        if hint in n:
            family = fam
            break
    lo, hi = FAMILY_WEIGHT_RANGES.get(family, (0.15, 0.35))
    return LoraInfluenceProfile(
        family=family,
        influence_tags=list(_FAMILY_TAGS.get(family, [])),
        weight_min=lo, weight_max=hi,
        weight_default=round((lo + hi) / 2, 2),
        identity_risk=FAMILY_IDENTITY_RISK.get(family, "medium"),
        compatible_families=[f for f in CONCEPT_FAMILIES
                             if f not in (family,)],
        notes=(f"auto-profiled from filename; trained dim "
               f"{_network_dim(meta)}" if _network_dim(meta) else
               "auto-profiled from filename"),
    )


def sidecar_path(lora_path: Path) -> Path:
    return lora_path.with_suffix("").with_name(
        lora_path.stem + SIDECAR_SUFFIX)


def _sidecar_candidates(lora_path: Path,
                        previews_root: Optional[Path] = None) -> list[Path]:
    cands = [sidecar_path(lora_path),
             lora_path.with_suffix(".json")]
    if previews_root:
        cands.append(Path(previews_root) / lora_path.stem / "meta.json")
    return cands


def parse_sidecar(data: dict, stem: str,
                  meta: Optional[dict] = None) -> tuple:
    """Tolerant parser for all sidecar schemas (v2 rich, v1 native, legacy).

    Returns (profile, display_name, description, preview_images_map).
    The full normalized object is produced by `normalized_metadata()`.
    """
    from .concept_metadata import normalize_concept_metadata
    cm = normalize_concept_metadata(data, stem)
    base = infer_profile(stem, meta or {})
    known = LoraInfluenceProfile.__dataclass_fields__
    native = {k: v for k, v in data.items() if k in known}
    prof_dict = {**{k: getattr(base, k) for k in known}, **native}
    # rich-schema mappings (only applied when the native key is absent)
    if "category" in data and "family" not in data:
        fam = str(data["category"]).lower()
        if fam in CONCEPT_FAMILIES:
            prof_dict["family"] = fam
            lo, hi = FAMILY_WEIGHT_RANGES.get(fam, (0.15, 0.35))
            prof_dict.setdefault("weight_min", lo)
            prof_dict.setdefault("weight_max", hi)
            prof_dict["identity_risk"] = FAMILY_IDENTITY_RISK.get(
                fam, prof_dict.get("identity_risk", "medium"))
    if "concept_tags" in data and "influence_tags" not in data:
        prof_dict["influence_tags"] = [str(t) for t in data["concept_tags"]]
    if "recommended_weight" in data and "weight_default" not in data:
        try:
            prof_dict["weight_default"] = float(data["recommended_weight"])
        except (TypeError, ValueError):
            pass
    if "description" in data and not prof_dict.get("notes"):
        prof_dict["notes"] = str(data["description"])
    # v2 fields flow into the profile (conservative, additive)
    if "priority_hint" in data:
        prof_dict["priority_hint"] = cm.priority_hint
    if "identity_risk_level" in data:
        prof_dict["identity_risk"] = cm.identity_risk_level
    if "recommended_weight_range" in data:
        lo2, hi2 = cm.recommended_weight_range
        prof_dict["weight_min"], prof_dict["weight_max"] = lo2, hi2
        prof_dict.setdefault("weight_default", round((lo2 + hi2) / 2, 2))
    if cm.known_conflicts:
        prof_dict["known_conflicts"] = [
            c.lora_id for c in cm.known_conflicts if c.lora_id]
        prof_dict["conflict_families"] = [
            c.concept_family for c in cm.known_conflicts
            if c.concept_family]
    if cm.concept_tags and "influence_tags" not in data \
            and "concept_tags" not in data:
        prof_dict["influence_tags"] = cm.concept_tags
    profile = LoraInfluenceProfile(
        **{k: v for k, v in prof_dict.items() if k in known})
    previews = cm.preview_images
    return (profile, str(data.get("display_name") or ""),
            str(data.get("description") or cm.notes or ""),
            previews if isinstance(previews, dict) else {})


def load_sidecar_raw(lora_path: Path,
                     previews_root: Optional[Path] = None
                     ) -> tuple[Optional[dict], Optional[str]]:
    """First readable sidecar wins.  Returns (data, source_path)."""
    for sp in _sidecar_candidates(lora_path, previews_root):
        if not sp.exists():
            continue
        try:
            data = json.loads(sp.read_text())
            if isinstance(data, dict):
                return data, str(sp)
        except Exception:
            continue          # tolerant: a broken sidecar never crashes scan
    return None, None


def load_sidecar(lora_path: Path) -> Optional[LoraInfluenceProfile]:
    data, _src = load_sidecar_raw(lora_path)
    if data is None:
        return None
    profile, _dn, _desc, _pv = parse_sidecar(data, lora_path.stem)
    return profile


def save_sidecar(lora_path: Path, profile: LoraInfluenceProfile) -> Path:
    sp = sidecar_path(lora_path)
    sp.write_text(json.dumps(asdict(profile), indent=2))
    return sp


def _find_preview(lora_path: Path) -> Optional[str]:
    for suffix in _PREVIEW_SUFFIXES:
        cand = lora_path.with_name(lora_path.stem + suffix)
        if cand.exists():
            return str(cand)
    return None


def find_preview_images(lora_path: Path, metadata: Optional[dict] = None,
                        previews_root: Optional[Path] = None) -> dict:
    """Discover strength-preview images for a LoRA.

    Precedence per level: sidecar-declared file > `<stem>.<level>.<ext>`
    beside the model > `<previews_root>/<stem>/<level>.<ext>`.  The
    "default" level also accepts `<stem>.preview.<ext>` / `<stem>.<ext>`.
    Returns {level: path} for levels that exist - missing levels degrade
    gracefully (placeholder state in the UI).
    """
    declared = (metadata or {}).get("preview_images") or {}
    out: dict[str, str] = {}
    for level in PREVIEW_LEVELS:
        # 1) sidecar-declared (relative to the model's folder)
        name = declared.get(level)
        if name:
            cand = (lora_path.parent / str(name)).expanduser()
            if cand.exists():
                out[level] = str(cand)
                continue
        # 2) beside the model
        if level == "default":
            beside = _find_preview(lora_path)
            if beside:
                out[level] = beside
                continue
        else:
            for ext in _IMAGE_EXTS:
                cand = lora_path.with_name(f"{lora_path.stem}.{level}{ext}")
                if cand.exists():
                    out[level] = str(cand)
                    break
            if level in out:
                continue
        # 3) dedicated previews folder
        if previews_root:
            for ext in _IMAGE_EXTS:
                cand = Path(previews_root) / lora_path.stem / f"{level}{ext}"
                if cand.exists():
                    out[level] = str(cand)
                    break
    return out


# ---------------------------------------------------------------------------
# Scan + index
# ---------------------------------------------------------------------------

def lora_dirs(prj: Project) -> list[Path]:
    """All configured LoRA folders (studio output + Forge Lora dir)."""
    dirs = []
    if prj.lora_output_dir:
        dirs.append(Path(prj.lora_output_dir).expanduser())
    elif prj.output_base:
        dirs.append(Path(prj.output_base).expanduser() / "LORA_OUTPUT")
    if prj.forge_root:
        dirs.append(Path(prj.forge_root).expanduser() / "models" / "Lora")
    return [d for d in dirs if d.exists()]


def default_previews_root(prj: Project) -> Optional[Path]:
    """Dedicated previews folder: `<output_base>/previews/lora/<stem>/`."""
    if prj.output_base:
        root = Path(prj.output_base).expanduser() / "previews" / "lora"
        if root.exists():
            return root
    return None


def scan_loras(prj: Project, dirs: Optional[list[Path]] = None,
               read_metadata: bool = True,
               previews_root: Optional[Path] = None) -> list[LoraCard]:
    """Build the LoRA index.  Sidecar profile > inferred profile.
    Tolerant by design: a LoRA with no metadata, no sidecar, or no preview
    still yields a complete card (placeholder preview state)."""
    import datetime
    if previews_root is None:
        previews_root = default_previews_root(prj)
    cards: dict[str, LoraCard] = {}
    for d in (dirs if dirs is not None else lora_dirs(prj)):
        files = [p for ext in LORA_EXTENSIONS for p in d.glob(f"*{ext}")]
        for p in sorted(files):
            if p.stem in cards:
                continue          # first dir wins (studio output preferred)
            meta = (read_safetensors_metadata(p)
                    if read_metadata and p.suffix == ".safetensors" else {})
            raw, source = load_sidecar_raw(p, previews_root)
            from .concept_metadata import normalize_concept_metadata
            if raw is not None:
                profile, display, desc, declared = parse_sidecar(
                    raw, p.stem, meta)
                cm = normalize_concept_metadata(raw, p.stem, source or "")
            else:
                profile, display, desc, declared = (
                    infer_profile(p.stem, meta), "", "", {})
                source = "inferred"
                cm = normalize_concept_metadata(
                    {"concept_family": profile.family,
                     "concept_tags": profile.influence_tags,
                     "identity_risk_level": profile.identity_risk,
                     "recommended_weight_range": [profile.weight_min,
                                                  profile.weight_max],
                     "notes": profile.notes}, p.stem, "inferred")
            levels = find_preview_images(
                p, {"preview_images": declared}, previews_root)
            try:
                st = p.stat()
                modified = datetime.datetime.fromtimestamp(
                    st.st_mtime).isoformat(timespec="seconds")
                size_mb = round(st.st_size / 1024 / 1024, 1)
            except OSError:
                modified, size_mb = None, 0.0
            cards[p.stem] = LoraCard(
                lora_id=p.stem, path=str(p),
                preview=levels.get("default"),
                modified_at=modified, size_mb=size_mb,
                network_dim=_network_dim(meta),
                sd_metadata={k: meta[k] for k in
                             ("ss_base_model_version", "ss_output_name",
                              "ss_num_epochs", "ss_learning_rate")
                             if k in meta},
                profile=profile,
                display_name=display or p.stem,
                description=desc or profile.notes,
                preview_levels=levels,
                has_preview=bool(levels),
                metadata_source=source,
                concept_meta=cm.to_payload(),
            )
    return list(cards.values())


_SCAN_CACHE: dict = {"key": None, "ts": 0.0, "cards": []}


def scan_loras_cached(prj: Project, max_age: float = 10.0,
                      force: bool = False) -> list[LoraCard]:
    """Cached scan for UI responsiveness with large libraries.

    The cache key is each LoRA folder's (path, mtime, entry count), so
    adding/removing/replacing files invalidates immediately; metadata-only
    sidecar edits are picked up within `max_age` seconds or on force.
    """
    import time
    dirs = lora_dirs(prj)
    try:
        key = tuple((str(d), d.stat().st_mtime, len(list(d.iterdir())))
                    for d in dirs)
    except OSError:
        key = tuple(str(d) for d in dirs)
    now = time.time()
    if (not force and _SCAN_CACHE["key"] == key
            and now - _SCAN_CACHE["ts"] < max_age):
        return _SCAN_CACHE["cards"]
    cards = scan_loras(prj, dirs=dirs)
    _SCAN_CACHE.update(key=key, ts=now, cards=cards)
    return cards


def discover_loras(roots: list[str],
                   previews_root: Optional[str] = None) -> list[LoraCard]:
    """Spec-parity API: scan explicit roots without a Project."""
    prj = Project()
    return scan_loras(prj,
                      dirs=[Path(r).expanduser() for r in roots
                            if Path(r).expanduser().exists()],
                      previews_root=Path(previews_root).expanduser()
                      if previews_root else None)


def build_explorer_payload(items: list[LoraCard],
                           preview_url_base: str =
                           "/api/concept/lora_preview") -> dict:
    """UI-ready payload: cards + index-backed preview URLs (no raw
    filesystem paths leak into preview serving)."""
    out = []
    for c in items:
        d = c.to_json()
        d["preview_levels_available"] = [
            lv for lv in PREVIEW_LEVELS if lv in c.preview_levels]
        d["preview_urls"] = {
            lv: f"{preview_url_base}/{c.lora_id}/{lv}"
            for lv in d["preview_levels_available"]}
        out.append(d)
    return {"items": out, "count": len(out)}


def filter_cards(cards: list[LoraCard], family: str = "",
                 tag: str = "", search: str = "",
                 sort: str = "name") -> list[LoraCard]:
    out = cards
    if family:
        out = [c for c in out if c.profile.family == family]
    if tag:
        out = [c for c in out if tag in c.profile.influence_tags]
    if search:
        s = search.lower()
        out = [c for c in out if s in c.lora_id.lower()
               or s in c.profile.notes.lower()]
    risk_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
    keys = {
        "name": lambda c: c.lora_id.lower(),
        "modified": lambda c: c.modified_at or "",
        "family": lambda c: (c.profile.family, c.lora_id.lower()),
        "identity_risk": lambda c: (risk_rank.get(c.profile.identity_risk, 2),
                                    c.lora_id.lower()),
    }
    return sorted(out, key=keys.get(sort, keys["name"]),
                  reverse=(sort == "modified"))


def sync_profiles_to_manifest(conn, cards: list[LoraCard]) -> int:
    """Mirror influence profiles into `lora_influence_profiles` (v8) so the
    UI and stack intelligence can query without re-scanning disks."""
    n = 0
    for c in cards:
        conn.execute(
            "INSERT INTO lora_influence_profiles (lora_id, path, family, "
            "influence_tags, weight_min, weight_max, weight_default, "
            "identity_risk, known_conflicts, notes, preview, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(lora_id) DO UPDATE SET path=excluded.path, "
            "family=excluded.family, influence_tags=excluded.influence_tags, "
            "weight_min=excluded.weight_min, weight_max=excluded.weight_max, "
            "weight_default=excluded.weight_default, "
            "identity_risk=excluded.identity_risk, "
            "known_conflicts=excluded.known_conflicts, "
            "notes=excluded.notes, preview=excluded.preview, "
            "updated_at=excluded.updated_at",
            (c.lora_id, c.path, c.profile.family,
             json.dumps(c.profile.influence_tags),
             c.profile.weight_min, c.profile.weight_max,
             c.profile.weight_default, c.profile.identity_risk,
             json.dumps(c.profile.known_conflicts), c.profile.notes,
             c.preview, now_iso()))
        n += 1
    conn.commit()
    return n
