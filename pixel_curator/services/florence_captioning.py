"""Florence-2 captioning - import-safe orchestration service.

Florence-2 runs as an OPTIONAL external VLM backend in a dedicated Python
environment (default: the ``lora-vlm`` conda env), invoked via subprocess. This
module is import-safe: it NEVER imports torch / transformers / Florence code at
import time, and never loads a model on its own - it only builds payloads, shells
out to ``florence2_runner``, reads structured JSON back, and post-processes the
natural-language caption into professional, SFW dataset metadata.

Captions complement (never overwrite) the existing WD14 / tag captions - results
are stored separately (sidecars + JSONL). Neutral visual language only; the
service never infers age/identity/sensitive attributes and never emits explicit
content (Florence output is sanitized).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VLM_PYTHON = ("/opt/homebrew/Caskroom/miniconda/base/envs/"
                      "lora-vlm/bin/python")
_ALT_VLM_PYTHONS = (
    str(Path.home() / "miniconda3/envs/lora-vlm/bin/python"),
    str(Path.home() / "miniconda/envs/lora-vlm/bin/python"),
)
RUNNER_MODULE = "pixel_curator.services.florence2_runner"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic", ".heif")

RESULTS_NAME = "florence_captions.jsonl"
SUMMARY_NAME = "florence_caption_summary.json"
CARD_NAME = "florence_caption_card.md"

VALID_TASK_PROMPTS = ("<MORE_DETAILED_CAPTION>", "<DETAILED_CAPTION>", "<CAPTION>")

# conservative SFW guard - if Florence emits any of these, the caption is
# neutralized and flagged (we never produce explicit/sexualized captions).
_UNSAFE_TOKENS = ("nude", "naked", "nsfw", "explicit", "nipple", "genital",
                  "porn", "sexual", "topless", "lingerie-clad")

ROUTING_TAGS = ("portrait_reference", "full_body_reference", "pose_reference",
                "outfit_reference", "texture_reference", "lighting_reference",
                "composition_reference", "captioning_reference",
                "figure_study_composition", "close_crop_study", "review_required")


# ============================ dataclasses ===================================
@dataclass
class FlorenceRuntimeConfig:
    python_executable: str = DEFAULT_VLM_PYTHON
    model_path: str = ""
    model_variant: str = "base-ft"     # base-ft | large-ft | custom
    device: str = "auto"               # auto | mps | cuda | cpu
    dtype: str = "float32"             # float32 | float16 | bfloat16 | auto
    local_files_only: bool = True
    allow_downloads: bool = False
    trust_remote_code: bool = True
    max_new_tokens: int = 256
    num_beams: int = 3
    timeout_seconds: int = 300
    cpu_fallback: bool = True


@dataclass
class FlorenceCaptionRequest:
    image_path: str
    task_prompt: str = "<MORE_DETAILED_CAPTION>"
    mode: str = "caption"              # caption | dense_caption | analysis | routing
    extra_prompt: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class FlorenceCaptionResult:
    ok: bool
    image_path: str
    model_path: str = ""
    model_variant: str = ""
    caption: str = ""
    raw_text: str = ""
    parsed: dict = field(default_factory=dict)
    visual_attributes: dict = field(default_factory=dict)
    routing_tags: list = field(default_factory=list)
    quality_signals: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    runtime: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ============================ runtime / paths ===============================
def default_florence_model_paths(project_root: Optional[Path] = None) -> dict:
    root = Path(project_root).expanduser() if project_root else REPO_ROOT
    base = root / "models" / "vlm" / "florence2"
    return {"base-ft": str(base / "Florence-2-base-ft"),
            "large-ft": str(base / "Florence-2-large-ft")}


def _resolve_python(cfg: FlorenceRuntimeConfig) -> Optional[str]:
    cands = [cfg.python_executable, *_ALT_VLM_PYTHONS]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def resolve_model_path(cfg: FlorenceRuntimeConfig,
                       project_root: Optional[Path] = None) -> str:
    if cfg.model_path:
        return str(Path(cfg.model_path).expanduser())
    paths = default_florence_model_paths(project_root)
    return paths.get(cfg.model_variant, paths.get("base-ft", ""))


def detect_florence_runtime(config: Optional[FlorenceRuntimeConfig] = None,
                            project_root: Optional[Path] = None) -> dict:
    """Report readiness without importing or loading anything heavy. Never
    raises."""
    cfg = config or FlorenceRuntimeConfig()
    py = _resolve_python(cfg)
    paths = default_florence_model_paths(project_root)
    model_status = {k: os.path.isdir(v) for k, v in paths.items()}
    chosen = resolve_model_path(cfg, project_root)
    warnings = []
    if py is None:
        warnings.append(
            "VLM Python not found - set --vlm-python or create the lora-vlm env; "
            "Florence is optional and the rest of the pipeline is unaffected")
    if chosen and not os.path.isdir(chosen):
        warnings.append(f"model path not found: {chosen} (no auto-download)")
    return {
        "available": bool(py) and bool(chosen) and os.path.isdir(chosen),
        "python_executable": py, "python_exists": bool(py),
        "configured_python": cfg.python_executable,
        "model_variant": cfg.model_variant, "model_path": chosen,
        "model_path_exists": bool(chosen) and os.path.isdir(chosen),
        "models_detected": model_status, "runner_module": RUNNER_MODULE,
        "warnings": warnings,
    }


def build_florence_subprocess_payload(request: FlorenceCaptionRequest,
                                      config: FlorenceRuntimeConfig,
                                      project_root: Optional[Path] = None) -> dict:
    task = request.task_prompt if request.task_prompt in VALID_TASK_PROMPTS \
        else "<MORE_DETAILED_CAPTION>"
    return {
        "image_path": str(request.image_path),
        "task_prompt": task, "mode": request.mode,
        "extra_prompt": request.extra_prompt or "",
        "model_path": resolve_model_path(config, project_root),
        "model_variant": config.model_variant,
        "device": config.device, "dtype": config.dtype,
        "local_files_only": config.local_files_only,
        "allow_downloads": config.allow_downloads,
        "trust_remote_code": config.trust_remote_code,
        "max_new_tokens": config.max_new_tokens, "num_beams": config.num_beams,
        "cpu_fallback": config.cpu_fallback,
    }


# ============================ SFW post-processing ===========================
def _sanitize_caption(text: str) -> tuple:
    """Return (clean_text, flags). Neutralizes explicit content; never emits it."""
    flags = []
    low = (text or "").lower()
    if any(tok in low for tok in _UNSAFE_TOKENS):
        flags.append("sanitized_explicit_content")
        return ("Figure-study composition flagged for human review.", flags)
    # strip score-style prefixes if any leaked in
    clean = (text or "").strip()
    return clean, flags


_FOCUS_RULES = [
    ("full_body", ("full body", "full-body", "head to toe", "standing figure")),
    ("upper_body", ("upper body", "waist up", "torso")),
    ("outfit", ("outfit", "fashion", "garment", "dress", "clothing", "jacket")),
    ("detail_texture", ("close-up", "close up", "macro", "texture", "fabric detail")),
    ("portrait", ("portrait", "headshot", "face", "head and shoulders")),
    ("composition", ("composition", "scene", "environment", "background")),
]
_POSE_RULES = [
    ("seated", ("seated", "sitting", "sits")),
    ("reclined", ("reclining", "lying", "lying down", "reclined")),
    ("dynamic", ("dynamic", "walking", "running", "jumping", "motion", "mid-stride")),
    ("gesture", ("gesturing", "reaching", "pointing", "raised arm")),
    ("close_crop", ("close-up", "close up", "cropped")),
    ("neutral", ("standing", "neutral", "still")),
]
_LIGHT_RULES = [
    ("excellent", ("studio lighting", "clean lighting", "well-lit", "soft light")),
    ("good", ("natural light", "balanced lighting", "daylight")),
    ("mixed", ("backlit", "harsh", "mixed lighting", "shadow")),
    ("poor", ("dark", "underexposed", "dim", "low light", "overexposed")),
]


def _first_match(text_low: str, rules, default="unknown") -> str:
    for label, keys in rules:
        if any(k in text_low for k in keys):
            return label
    return default


def extract_visual_attributes(caption: str,
                              existing_metadata: Optional[dict] = None) -> dict:
    """Rule-based, SFW extraction of structured attributes from a Florence
    natural caption + existing quality metadata. Never infers age/identity."""
    md = existing_metadata or {}
    low = (caption or "").lower()
    focus = _first_match(low, _FOCUS_RULES)
    pose = _first_match(low, _POSE_RULES)
    light = _first_match(low, _LIGHT_RULES)
    detail = "unknown"
    if any(k in low for k in ("fine detail", "fine texture", "high detail",
                              "intricate", "sharp texture")):
        detail = "high"
    elif any(k in low for k in ("texture", "fabric", "detail")):
        detail = "moderate"
    elif low:
        detail = "general"
    comp = {"full_body": "full_body", "portrait": "portrait",
            "detail_texture": "close_crop"}.get(focus, "unknown")
    if "studio" in low:
        comp = "studio"
    elif "environment" in low or "outdoor" in low:
        comp = "environmental"
    arch = md.get("architecture_target", "unknown")

    targets, tags = [], []
    fmap = {
        "portrait": ("portrait_reference", "portrait_dataset"),
        "upper_body": ("portrait_reference", "outfit_dataset"),
        "full_body": ("full_body_reference", "full_body_pose_dataset"),
        "outfit": ("outfit_reference", "outfit_dataset"),
        "detail_texture": ("texture_reference", "detail_texture_dataset"),
        "composition": ("composition_reference", "lighting_composition_dataset"),
    }
    if focus in fmap:
        tags.append(fmap[focus][0])
        targets.append(fmap[focus][1])
    if pose in ("dynamic", "gesture"):
        tags.append("pose_reference")
        targets.append("gesture_pose_dataset")
    if light in ("good", "excellent"):
        tags.append("lighting_reference")
    if caption:
        tags.append("captioning_reference")
    flags = []
    if not caption:
        flags.append("empty_caption")
        tags = ["review_required"]
    # AGE SAFETY: only from an explicit human/metadata flag, never inferred here.
    if md.get("age_uncertain") is True:
        flags.append("review_required_age_uncertain")
        tags = ["review_required"]
        targets = []
    # dedupe
    def _ded(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return {
        "natural_caption": caption, "caption_style": "florence_natural",
        "primary_focus": focus, "pose_type": pose, "detail_level": detail,
        "lighting_quality": light, "composition_type": comp,
        "architecture_target": arch, "recommended_targets": _ded(targets),
        "routing_tags": _ded([t for t in tags if t in ROUTING_TAGS] + tags),
        "flags": flags,
    }


def _finalize_result(raw: dict, request: FlorenceCaptionRequest,
                     cfg: FlorenceRuntimeConfig) -> FlorenceCaptionResult:
    """Turn a runner's raw JSON dict into a post-processed result."""
    ok = bool(raw.get("ok"))
    raw_text = raw.get("text") or raw.get("caption") or ""
    clean, sflags = _sanitize_caption(raw_text)
    attrs = extract_visual_attributes(clean, request.metadata)
    flags = sorted(set(sflags + attrs.get("flags", []) + raw.get("flags", [])))
    if "review_required_age_uncertain" in flags:
        clean = ""
    return FlorenceCaptionResult(
        ok=ok and "review_required_age_uncertain" not in flags,
        image_path=str(request.image_path),
        model_path=raw.get("model_path", resolve_model_path(cfg)),
        model_variant=cfg.model_variant,
        caption=clean, raw_text=raw_text, parsed=raw.get("parsed", {}),
        visual_attributes=attrs, routing_tags=attrs.get("routing_tags", []),
        quality_signals=request.metadata.get("quality_signals", {})
        if isinstance(request.metadata.get("quality_signals"), dict) else {},
        flags=flags, warnings=list(raw.get("warnings", [])),
        errors=list(raw.get("errors", [])),
        runtime={"device": raw.get("device"), "dtype": raw.get("dtype"),
                 "elapsed_s": raw.get("elapsed_s"), "mode": request.mode})


def _error_result(request, cfg, msg, warnings=None) -> FlorenceCaptionResult:
    return FlorenceCaptionResult(
        ok=False, image_path=str(request.image_path),
        model_variant=cfg.model_variant, model_path=resolve_model_path(cfg),
        errors=[msg], warnings=list(warnings or []),
        visual_attributes=extract_visual_attributes("", request.metadata),
        flags=["florence_unavailable"])


# ============================ run (subprocess) ==============================
def run_florence_caption(request: FlorenceCaptionRequest,
                         config: Optional[FlorenceRuntimeConfig] = None,
                         project_root: Optional[Path] = None
                         ) -> FlorenceCaptionResult:
    """Caption one image via the external runner. Never raises; returns a clean
    error result when Florence is unavailable."""
    cfg = config or FlorenceRuntimeConfig()
    rt = detect_florence_runtime(cfg, project_root)
    if not rt["python_exists"]:
        return _error_result(request, cfg, "VLM Python executable not found",
                             rt["warnings"])
    if not rt["model_path_exists"]:
        return _error_result(request, cfg,
                             f"Florence model not found: {rt['model_path']}",
                             rt["warnings"])
    if not Path(request.image_path).exists():
        return _error_result(request, cfg,
                             f"image not found: {request.image_path}")
    payload = build_florence_subprocess_payload(request, cfg, project_root)
    tmpdir = Path(tempfile.mkdtemp(prefix="florence_"))
    pj, rj = tmpdir / "payload.json", tmpdir / "result.json"
    try:
        pj.write_text(json.dumps(payload), encoding="utf-8")
        cmd = [rt["python_executable"], "-m", RUNNER_MODULE,
               "--payload-json", str(pj), "--result-json", str(rj)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=cfg.timeout_seconds, cwd=str(REPO_ROOT))
        if rj.exists():
            raw = json.loads(rj.read_text(encoding="utf-8"))
        else:
            return _error_result(
                request, cfg,
                f"runner produced no result (rc={proc.returncode})",
                [proc.stderr.strip()[:300]] if proc.stderr else [])
        raw.setdefault("elapsed_s", round(time.time() - t0, 2))
        return _finalize_result(raw, request, cfg)
    except subprocess.TimeoutExpired:
        return _error_result(request, cfg,
                             f"Florence timed out after {cfg.timeout_seconds}s")
    except json.JSONDecodeError as exc:
        return _error_result(request, cfg, f"invalid runner JSON: {exc}")
    except Exception as exc:
        return _error_result(request, cfg, f"runner invocation failed: {exc}")
    finally:
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def run_florence_batch(image_paths, output_jsonl, config=None, mode="caption",
                       limit=None, project_root=None,
                       caption_fn: Optional[Callable] = None,
                       resume: bool = True) -> dict:
    """Caption many images, streaming to JSONL. Resumable. ``caption_fn`` lets
    tests inject a fake captioner (defaults to ``run_florence_caption``)."""
    cfg = config or FlorenceRuntimeConfig()
    fn = caption_fn or run_florence_caption
    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if resume and out.exists():
        try:
            for line in out.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line).get("image_path"))
        except Exception:
            done = set()
    paths = list(image_paths)
    if limit:
        paths = paths[:limit]
    ok = fail = 0
    with out.open("a", encoding="utf-8") as fh:
        for p in paths:
            sp = str(p)
            if sp in done:
                continue
            req = FlorenceCaptionRequest(image_path=sp, mode=mode)
            res = fn(req, cfg) if caption_fn else fn(req, cfg, project_root)
            fh.write(json.dumps(res.to_dict()) + "\n")
            fh.flush()
            ok += 1 if res.ok else 0
            fail += 0 if res.ok else 1
    return {"output_jsonl": str(out), "processed": ok + fail, "ok": ok,
            "failed": fail, "summary": summarize_florence_results(out)}


# ============================ load / summarize ==============================
def load_florence_results(jsonl_path) -> list:
    p = Path(jsonl_path)
    if p.is_dir():
        p = p / RESULTS_NAME
    out = []
    if not p.exists():
        return out
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    except Exception:
        pass
    return out


def summarize_florence_results(jsonl_path) -> dict:
    results = (jsonl_path if isinstance(jsonl_path, list)
               else load_florence_results(jsonl_path))
    total = len(results)
    ok = sum(1 for r in results if r.get("ok"))

    def dist(getter):
        d = {}
        for r in results:
            try:
                k = getter(r)
            except Exception:
                k = "unknown"
            d[k] = d.get(k, 0) + 1
        return dict(sorted(d.items(), key=lambda kv: -kv[1]))
    return {
        "total": total, "ok": ok, "failed": total - ok,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_focus_distribution": dist(
            lambda r: r["visual_attributes"]["primary_focus"]),
        "pose_distribution": dist(
            lambda r: r["visual_attributes"]["pose_type"]),
        "detail_distribution": dist(
            lambda r: r["visual_attributes"]["detail_level"]),
        "flag_distribution": dist(
            lambda r: (r.get("flags") or ["none"])[0]),
    }


def render_florence_card(summary: dict) -> str:
    lines = ["# Florence-2 Caption Card", "",
             f"Captioned: **{summary.get('ok', 0)}/{summary.get('total', 0)}**", ""]
    for title, key in (("Primary focus", "primary_focus_distribution"),
                       ("Pose", "pose_distribution"),
                       ("Detail", "detail_distribution"),
                       ("Flags", "flag_distribution")):
        lines.append(f"## {title}")
        for k, v in (summary.get(key) or {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def write_florence_sidecars(result: FlorenceCaptionResult, out_dir) -> dict:
    """Write <stem>.florence.json + .txt into out_dir (separate from WD14)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image_path).stem
    written = {}
    try:
        jp = out / f"{stem}.florence.json"
        jp.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        written["json"] = str(jp)
        if result.caption:
            tp = out / f"{stem}.florence.txt"
            tp.write_text(result.caption + "\n", encoding="utf-8")
            written["txt"] = str(tp)
    except Exception:
        pass
    return written


def results_by_filename(jsonl_path) -> dict:
    """basename -> compact fields for Dataset Explorer merge."""
    out = {}
    for r in load_florence_results(jsonl_path):
        name = Path(r.get("image_path", "")).name
        if not name:
            continue
        va = r.get("visual_attributes", {})
        out[name] = {
            "caption_natural": r.get("caption", ""),
            "caption_natural_source": "florence2" if r.get("ok") else "",
            "florence_model_variant": r.get("model_variant", ""),
            "florence_processed": bool(r.get("ok")),
            "primary_focus": va.get("primary_focus"),
            "pose_type": va.get("pose_type"),
            "detail_level": va.get("detail_level"),
            "lighting_quality": va.get("lighting_quality"),
            "composition_type": va.get("composition_type"),
            "routing_tags": va.get("routing_tags", []),
            "florence_flags": r.get("flags", []),
        }
    return out


__all__ = [
    "FlorenceRuntimeConfig", "FlorenceCaptionRequest", "FlorenceCaptionResult",
    "default_florence_model_paths", "detect_florence_runtime",
    "resolve_model_path", "build_florence_subprocess_payload",
    "extract_visual_attributes", "run_florence_caption", "run_florence_batch",
    "load_florence_results", "summarize_florence_results", "render_florence_card",
    "write_florence_sidecars", "results_by_filename", "RESULTS_NAME",
    "SUMMARY_NAME", "CARD_NAME", "VALID_TASK_PROMPTS",
]
