"""Project configuration: constants, stage configs, project-file loader.

A *project file* (TOML on py3.11+, JSON everywhere) replaces editing
constants in code. Generate a starter with:  pixel-curator init
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib  # py3.11+
    HAVE_TOML = True
except Exception:
    HAVE_TOML = False

# -----------------------------
# Constants
# -----------------------------

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff"}

DB_NAME = "extractor_manifest.sqlite3"
LOG_NAME = "extractor.log"
UI_PORT = 7861  # 7860 is usually taken by Forge/A1111

# Subject domains: "human" runs the full face/pose/anatomy/identity signal
# set; anything else runs domain-agnostic signals only.
VALID_DOMAINS = ("human", "object", "animal", "general")

# Neutral defaults: a project file (or explicit flags) is expected to set
# real paths.  `pixel-curator init` writes them into the starter TOML.
DEFAULT_VIDEO_DIRS: list[str] = []
DEFAULT_PHOTOS_DIR = ""
DEFAULT_OUTPUT_BASE = "outputs"

# Forge/reForge install whose already-downloaded models we can reuse
# (BLIP captioner, DeepBooru tagger, GFPGAN, YuNet) in later phases.
# Empty = not configured; Forge-backed commands explain how to set it.
DEFAULT_FORGE_ROOT = ""


# -----------------------------
# Stage configs
# -----------------------------

# Duration-keyed sampling ladder. Each tier is
#   (max_duration_seconds, fps, max_frames)
# and the FIRST tier whose max_duration is >= the clip duration wins; the
# final tier (max_duration = inf) is the catch-all for the longest clips.
# Sampling is keyed on *duration* — that is what controls how many seconds
# pass between frames ("every 1 second counts" == ~1 fps). Larger files are
# generally longer, so they naturally fall into the higher tiers.
#
# Names map to the size/length buckets the studio thinks in:
#   live   : Live Photo motion clips (a few seconds)        -> densest
#   short  : short-to-medium clips up to ~1 min
#   medium : medium clips ~1-5 min   (~1-10 GB territory)
#   long   : long clips ~5-20 min    (~10-50 GB territory)
#   xlong  : extra-long ~20 min+      (50 GB+; e.g. a 45-min capture)
DEFAULT_FPS_TIERS: tuple[tuple[float, float, int], ...] = (
    (6.0,           10.0, 120),    # live   : ~10 fps, up to 120 frames
    (60.0,           4.0, 480),    # short  : ~4 fps,  up to 480 frames
    (300.0,          2.0, 1200),   # medium : ~2 fps,  up to 1200 frames
    (1200.0,         1.0, 2400),   # long   : ~1 fps,  up to 2400 frames
    (float("inf"),   1.0, 6000),   # xlong  : ~1 fps,  up to 6000 frames
)


@dataclass
class PrefilterConfig:
    """Early pre-filtering gates (Phase 5). Fast, decode-light rejection of
    files that would never survive curation: thumbnails, social-media
    re-compressions, truncated downloads, caption-bar screenshots. All
    rejections are status-based ('rejected_prefilter' + reason) and fully
    reversible via `prefilter --restore`.
    """
    output_base: Path
    min_bytes: int = 12_000            # below this a photo is thumbnail junk
    min_side: int = 256                # shortest image side (header read only)
    min_bpp: float = 0.05              # bytes/pixel floor for JPEG/WebP;
                                       # heavy social re-compression sits below
    check_truncation: bool = True      # JPEG missing EOI marker (2-byte tail)
    text_check: bool = True            # Snapchat-style caption-bar detector
                                       # (needs one small gray decode)
    extreme_blur: float = 5.0          # Laplacian-var floor on 96x96 gray;
                                       # far below curate's min_sharpness=35
    decode_checks: bool = True         # standalone runs include text/blur;
                                       # ingestion tier stays zero-decode
    dry_run: bool = False
    workers: int = 8


@dataclass
class ExtractConfig:
    output_base: Path
    fps: float = 0.25
    jpeg_quality: int = 2
    segment_seconds: int = 300
    use_personfromvid: bool = False
    import_photos: bool = True
    photo_import_mode: str = "hardlink"
    resume: bool = True
    dry_run: bool = False
    limit_videos: int = 0
    overwrite: bool = False
    max_side: int = 0
    # Adaptive, duration-aware sampling. When enabled, the per-video frame
    # rate and frame ceiling are chosen from the clip's duration via the
    # tier ladder below, so short Live Photo clips are sampled densely while
    # longer videos taper toward ~1 fps. Disable with adaptive=False to use a
    # uniform `fps` for every clip (legacy behaviour).
    adaptive: bool = True
    fps_tiers: tuple[tuple[float, float, int], ...] = field(
        default_factory=lambda: DEFAULT_FPS_TIERS
    )
    # If > 0, this overrides EVERY tier's frame ceiling. 0 = use per-tier caps.
    max_frames_per_video: int = 0
    min_frames_per_video: int = 4      # floor so very short clips still yield frames
    # Phase 5: run the zero-decode prefilter battery on imported photos so
    # thumbnail junk / heavy re-compressions never enter curation as 'new'.
    # Rejections are reversible ('rejected_prefilter'). --no-prefilter opts out.
    prefilter: bool = True
    # Phase G2 (Curator Studio): optional per-video trim window in seconds.
    # 0.0 for both keeps the legacy whole-video behaviour; trim_end of 0
    # means "until the end". Segments are laid out inside the window only.
    trim_start: float = 0.0
    trim_end: float = 0.0


@dataclass
class CurateConfig:
    output_base: Path
    hamming_threshold: int = 4
    min_sharpness: float = 35.0
    min_brightness: float = 18.0
    max_brightness: float = 242.0
    workers: int = 8
    rescore: bool = False
    # Phase 5: text-overlay/caption-bar detection piggybacks the 96x96 gray
    # that scoring already decodes (zero extra I/O). Flagged frames become
    # 'rejected_prefilter' (reason text_overlay) instead of 'scored'.
    prefilter_text: bool = True


@dataclass
class SmartCurateConfig:
    output_base: Path
    anchor_dir: Optional[Path] = None      # reference images of the subject
    identity_threshold: float = 0.35       # cosine sim vs anchor (ArcFace space)
    min_face_area: float = 0.015           # face bbox area / image area floor
    det_size: int = 640
    batch_log_every: int = 100
    rescan: bool = False                   # re-run detection on already-scanned frames
    auto_rotate: bool = True               # try 90/180/270 when no face found;
                                           # rotate the frame file upright in place
    # Orientation gates (Phase 3): mirror orient.OrientConfig defaults so
    # `curate --smart` and the standalone `orient` command decide identically.
    orient_min_det_score: float = 0.55     # confidence floor for rotations
    orient_min_axis_ratio: float = 1.25    # eyes->mouth axis ambiguity gate
    # Optional Google Takeout sidecar assist: when the source's people tag
    # confirms the subject, relax the identity threshold so good frames at a
    # bad angle are not falsely rejected as wrong-identity.
    metadata_dir: Optional[Path] = None
    subject_names: tuple = ()
    sidecar_identity_relax: float = 0.10
    # Priority-5 performance options (opt-in; defaults preserve behavior).
    profile_performance: bool = False      # collect per-stage timings + summary


@dataclass
class ClusterConfig:
    output_base: Path
    k: int = 0                 # 0 = auto: sqrt(n/2), clamped 2..40
    batch_size: int = 16
    max_iter: int = 30
    reembed: bool = False      # recompute existing CLIP embeddings


@dataclass
class CaptionConfig:
    output_base: Path
    trigger: str = "ohwx"
    class_word: str = "person"
    threshold: float = 0.35           # general-tag confidence floor
    char_threshold: float = 0.85      # character-tag confidence floor
    pony_prefix: bool = False         # prepend score_9, score_8_up, ...
    blacklist: str = ""               # comma-separated tags to drop
    remap: str = ""                   # "old:new,old2:new2"
    prune: str = ""                   # permanent-trait tags to absorb into trigger
    max_tags: int = 30
    force: bool = False               # re-caption frames with edited=0
    batch_log_every: int = 50
    repo_id: str = "SmilingWolf/wd-swinv2-tagger-v3"
    # Tagging backend (Phase 1): wd14 (ONNX, existing behavior) |
    # mps (torch+timm, WD14-v3 weights on the Apple Silicon GPU) |
    # florence (Florence-2 prose -> pseudo-tags). Unavailable backends
    # fall back to wd14 with a logged reason.
    tagger: str = "wd14"


DEFAULT_QUOTA = "closeup=0.30,portrait=0.30,upper_body=0.25,full_body=0.15"


@dataclass
class PackageConfig:
    output_base: Path
    token: str = "ohwx"
    class_word: str = "person"
    repeats: int = 10
    max_per_video: int = 40
    max_total: int = 0
    link_mode: str = "hardlink"
    write_captions: bool = True
    caption_text: str = ""
    dataset_name: str = "DATASET"
    quota: str = DEFAULT_QUOTA        # framing quotas; "" disables
    use_caption_table: bool = True    # per-frame WD14 captions when present
    training_config: bool = True      # Phase 10: write training_config.*
                                      # beside the package (suggestion only)


# -----------------------------
# Project file
# -----------------------------

@dataclass
class Project:
    name: str = "PIXEL_CURATOR"
    video_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_VIDEO_DIRS))
    photos_dir: str = DEFAULT_PHOTOS_DIR
    output_base: str = DEFAULT_OUTPUT_BASE
    trigger_token: str = "ohwx"
    class_word: str = "person"
    # Subject domain: which curation signals apply. "human" enables the full
    # face/pose/anatomy/identity signal set; other domains run only generic
    # signals (sharpness, exposure, duplicates, resolution, diversity).
    # human | object | animal | general
    domain: str = "human"
    anchor_dir: str = ""                   # subject reference images for identity curation
    metadata_dir: str = ""                 # Google Takeout sidecar JSON folder (optional)
    subject_names: list[str] = field(default_factory=list)  # e.g. ["Michelle_Huang"]
    forge_root: str = DEFAULT_FORGE_ROOT
    ui_port: int = UI_PORT
    recipes: dict = field(default_factory=dict)   # [recipes.NAME] sections
    # Phase 5 training
    sd_scripts_dir: str = ""               # kohya sd-scripts checkout
    sd_scripts_python: str = ""            # python inside sd-scripts' own venv
    base_model: str = ""                   # Pony Diffusion V6 XL .safetensors path
    lora_output_dir: str = ""              # default: {output_base}/LORA_OUTPUT
    # [runtime] section - device/precision policy (additive; absent = defaults)
    # keys: device=auto|cuda|mps|cpu, precision=auto|float16|bfloat16|float32,
    #       allow_tf32=true, cuda_empty_cache=true, mps_fallback=true
    runtime: dict = field(default_factory=dict)
    # Default captioning backend (Phase 1): wd14 | mps | florence.
    # Overridden per-run by `caption --tagger`. Absent -> wd14 (unchanged).
    default_tagger: str = "wd14"
    # [dreambooth_selector] section (Phase 6) - profile, thresholds, signal
    # weights, quotas and rules for curate/dreambooth_nsfw_selector. Absent
    # section = built-in 'balanced' profile (identical to legacy behavior).
    dreambooth_selector: dict = field(default_factory=dict)
    # [pipelines.NAME] sections (Phase 7) - declarative curation/selection
    # pipeline recipes: stages, gates, per-stage options. Built-in templates
    # (curation/selection/complete) exist even with no sections defined.
    pipelines: dict = field(default_factory=dict)

    @property
    def output_path(self) -> Path:
        return Path(self.output_base).expanduser()


PROJECT_TEMPLATE = """# pixel-curator project file
name = "PIXEL_CURATOR"

# Source media (never modified)
video_dirs = [
{video_dirs}
]
photos_dir = "{photos_dir}"

# All derived data (frames, manifest, datasets) goes here
output_base = "{output_base}"

# Training identity
trigger_token = "ohwx"
class_word = "person"

# Subject domain: human | object | animal | general.
# "human" = full face/pose/anatomy/identity signal set; any other value runs
# domain-agnostic signals only (sharpness, exposure, duplicates, diversity).
domain = "human"

# 5-15 clear reference images of the subject, for identity curation
anchor_dir = ""

# Existing Forge/reForge install (models reused in later phases)
forge_root = "{forge_root}"

ui_port = 7861

# ---------------------------------------------------------------
# Compute (optional): pick a dedicated accelerator provider.
# device   = torch:   auto | cuda | mps | cpu
# provider = onnx:    auto | cuda | coreml | cpu   (WD14 tagger,
#            insightface likeness, orientation, quality signals)
# "auto" picks CUDA -> MPS/CoreML -> CPU. Unavailable choices fall back
# to CPU with a note in `pixel-curator doctor`.
# ---------------------------------------------------------------

[runtime]
device = "auto"
provider = "auto"
precision = "auto"     # auto | float32 | float16 | bfloat16
allow_tf32 = true

# Phase 5 training (kohya sd-scripts)
sd_scripts_dir = ""        # e.g. /Users/you/sd-scripts
base_model = ""            # e.g. /Volumes/.../ponyDiffusionV6XL.safetensors
lora_output_dir = ""       # default: {output_base}/LORA_OUTPUT

# ---------------------------------------------------------------
# Dataset recipes (Phase 4): build with  pixel-curator build --recipe NAME
# Filters: framing="closeup,portrait" | include_clusters="2,5" |
#          exclude_clusters="3" | min_identity=0.45 | min_sharpness=50
# ---------------------------------------------------------------

[recipes.character_v1]
type = "character"
repeats = 10
max_total = 400
max_per_video = 40
quota = "closeup=0.30,portrait=0.30,upper_body=0.25,full_body=0.15"
val_fraction = 0.05

[recipes.style_v1]
type = "style"
repeats = 4
max_total = 600
max_per_video = 20
quota = ""

[recipes.outfit_v1]
type = "outfit"
repeats = 8
max_total = 150
framing = "upper_body,full_body"
include_clusters = ""        # set after clustering, e.g. "2,5"
token = "outfitx"
class_word = "outfit"

[recipes.pose_v1]
type = "pose"
repeats = 6
max_total = 200
framing = "full_body"
quota = ""

[recipes.detail_v1]
type = "detail"
repeats = 12
max_total = 120
framing = "closeup"
min_sharpness = 80

# multi-concept build: each listed recipe becomes its own folder
[recipes.combo_v1]
concepts = "character_v1,outfit_v1"
val_fraction = 0.05
"""


def write_template(path: Path) -> None:
    body = PROJECT_TEMPLATE.format(
        video_dirs="\n".join(f'    "{d}",' for d in DEFAULT_VIDEO_DIRS),
        photos_dir=DEFAULT_PHOTOS_DIR,
        output_base=DEFAULT_OUTPUT_BASE,
        forge_root=DEFAULT_FORGE_ROOT,
    )
    path.write_text(body, encoding="utf-8")


def _parse_flat_toml(text: str) -> dict:
    """Minimal TOML subset parser (Python 3.10, no tomllib): scalar keys,
    multi-line string arrays, and [table.subtable] section headers. Only
    intended for files written by `pixel-curator init` / the Settings tab."""
    data: dict = {}
    current: dict = data
    key: Optional[str] = None
    in_array = False
    items: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip() if not raw.strip().startswith('"') else raw.strip()
        if not line:
            continue
        if not in_array and line.startswith("[") and line.endswith("]"):
            current = data
            for part in line[1:-1].strip().split("."):
                part = part.strip()
                if part:
                    current = current.setdefault(part, {})
            continue
        if in_array:
            if line.startswith("]"):
                current[key] = items
                in_array = False
                key = None
                items = []
            else:
                item = line.rstrip(",").strip()
                if item.startswith('"') and item.endswith('"'):
                    items.append(item[1:-1])
            continue
        if "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        if v == "[":
            key = k
            in_array = True
            items = []
        elif v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            current[k] = [
                s.strip().strip('"') for s in inner.split(",") if s.strip()
            ] if inner else []
        elif v.startswith('"') and v.endswith('"'):
            current[k] = v[1:-1]
        elif v in ("true", "false"):
            current[k] = v == "true"
        else:
            try:
                current[k] = int(v)
            except ValueError:
                try:
                    current[k] = float(v)
                except ValueError:
                    current[k] = v
    return data


def dumps_toml(prj: Project) -> str:
    """Serialize a Project to TOML (flat schema, no external deps)."""
    lines = [f'name = "{prj.name}"', "", "video_dirs = ["]
    lines += [f'    "{d}",' for d in prj.video_dirs]
    lines += [
        "]",
        f'photos_dir = "{prj.photos_dir}"',
        f'output_base = "{prj.output_base}"',
        f'trigger_token = "{prj.trigger_token}"',
        f'class_word = "{prj.class_word}"',
        f'anchor_dir = "{prj.anchor_dir}"',
        f'metadata_dir = "{prj.metadata_dir}"',
        "subject_names = ["
        + ", ".join(f'"{s}"' for s in prj.subject_names) + "]",
        f'forge_root = "{prj.forge_root}"',
        f"ui_port = {prj.ui_port}",
        f'sd_scripts_dir = "{prj.sd_scripts_dir}"',
        f'sd_scripts_python = "{prj.sd_scripts_python}"',
        f'base_model = "{prj.base_model}"',
        f'lora_output_dir = "{prj.lora_output_dir}"',
        "",
    ]
    for rname in sorted(prj.recipes or {}):
        lines.append(f"[recipes.{rname}]")
        for k, v in (prj.recipes[rname] or {}).items():
            if isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            else:
                lines.append(f'{k} = "{v}"')
        lines.append("")
    return "\n".join(lines)


def save_project(prj: Project, path: Path) -> None:
    if path.suffix.lower() == ".json":
        from dataclasses import asdict
        path.write_text(json.dumps(asdict(prj), indent=2), encoding="utf-8")
    else:
        path.write_text(dumps_toml(prj), encoding="utf-8")


def load_project(path: Optional[str]) -> Project:
    """Load project file (TOML or JSON). Missing path -> defaults."""
    if not path:
        return Project()
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Project file not found: {p}")
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
    elif p.suffix.lower() in (".toml", ".tml"):
        text = p.read_text(encoding="utf-8")
        if HAVE_TOML:
            data = tomllib.loads(text)
        else:
            # py3.10 fallback: our project schema is flat (str/int/list[str])
            data = _parse_flat_toml(text)
    else:
        raise ValueError(f"Unsupported project file type: {p.suffix}")
    known = {f for f in Project.__dataclass_fields__}
    # Tolerate organizational section headers ([project], [paths], ...):
    # flatten unknown one-level sections so their keys still map to fields.
    flat: dict = {}
    for k, v in data.items():
        if k in known:
            flat[k] = v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if k2 in known and k2 not in flat:
                    flat[k2] = v2
    return Project(**flat)
