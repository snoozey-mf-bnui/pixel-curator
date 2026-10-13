# pixel-curator

**Local-first dataset curation for DreamBooth likeness training and LoRA training.**

You have hundreds or thousands of images of a face, a character, a product, or a style. You want to train a model that reproduces it. The hard part isn't the training — it's everything before it: deduping, culling bad frames, fixing skewed distribution, captioning, and exporting in the folder structure your trainer actually expects.

`pixel-curator` is a terminal-based CLI that does that entire pipeline. No GUI, no accounts, no cloud — every byte stays on your machine.

It is built primarily for **subject / identity training** — the case where the goal is model likeness: a DreamBooth fine-tune or a LoRA that reliably reproduces one specific target from a large personal image or video archive. The curation core is generic enough to be repurposed, but the default workflow, scoring signals, and export targets are optimized for likeness training.

---

## Why this exists

If you've collected 1,000+ images of a subject, the failures happen before the first training step:

- A large fraction are near-duplicates — burst shots, adjacent video frames, minor crops
- Some are technically unusable — low resolution, soft focus, compression artifacts
- The distribution is skewed — hundreds of shots from one angle, almost none from another
- Captioning is a second project on its own
- Handing a raw folder of 1,792 images to a Colab notebook is a fast route to disconnects, VRAM exhaustion, and daily-limit lockouts

A well-curated 200–500 image subset reliably outperforms a messy 2,000 on likeness training. Most of the work is figuring out which subset that is. That's what this tool is for.

---

## Pipeline

Every stage is resumable, dry-run by default where mutations are possible, and tracked in a SQLite manifest.

| Stage | Command | What happens |
|---|---|---|
| Set up | `init`, `doctor` | Scaffold a project file, verify dependencies (ffmpeg, models, backends) |
| Ingest | `extract`, `photos`, `smart-extract` | Frame extraction from video (HDR-aware, adaptive FPS, resumable) and photo import (JPEG/PNG/WebP/HEIC) |
| Filter | `prefilter` | Fast multiprocessed pass to drop technically unusable frames |
| Score | `curate` | Optical quality scoring, perceptual-hash dedup, per-frame status assignment |
| Cluster | `cluster`, `viz` | CLIP embeddings cached in the manifest, k-means clustering, embedding maps |
| Caption | `caption`, `florence` | WD14 tagger (auto-downloads) or Florence-2 natural-language captions |
| Semantic filter | `filter` | Keep/drop frames matching a CLIP text query ("beach photos", "images with watermarks") |
| Review | `review export` / `review import` | Static HTML report + spreadsheet keep/reject round-trip |
| Export | `export manifest`, `export select`, `package` | Flat CSV/JSONL/Parquet manifest; top-N% / threshold / balanced selections; **Kohya LoRA and DreamBooth instance packages** |
| Train & eval | `train`, `eval`, `likeness`, `pick-epoch` | Optional LoRA training and identity-likeness evaluation |
| Operate | `stats`, `health`, `backup`, `restore`, `selftest` | Manifest stats, pipeline health, safety nets, smoke tests |

---

## What "training-ready" means here

Two export paths, both first-class:

**DreamBooth** — `package kohya --mode dreambooth` produces a numbered `instance/` folder matching the standard Colab DreamBooth notebooks:

```
instance/
  000001.jpg
  000001.txt
  000002.jpg
  000002.txt
dataset_config.toml
```

Point any DreamBooth trainer at the folder and train. Captions are assembled from the manifest's tags and quality metadata at packaging time — no model calls needed during export.

**LoRA (multi-concept)** — `package kohya --mode lora` produces `concept_<target>/` folders with a `dataset_config.toml` ready for in-house Kohya `sd-scripts` setups.

Before packaging, `export select` materializes a curated slice into a folder:

- `--top-pct 10` — best 10% by unified score, symlinks by default (cull an 80 GB library without copying a byte)
- `--min-score 0.8` — everything above a threshold, real copies with caption sidecars
- `--balanced 500` — 500 frames spread evenly across CLIP clusters (run `cluster` first)

Every selection folder includes a `selection_manifest.csv` mapping outputs back to their sources.

The unified score is face-aware quality (0–1) when available, otherwise the percentile rank of sharpness — so thresholds mean the same thing regardless of which scorers ran.

---

## Install

Requires **Python 3.10+** and **ffmpeg** (for video extraction).

```bash
git clone https://github.com/snoozey-mf-bnui/pixel-curator.git
cd pixel-curator
pip install -e .            # core: stdlib + Pillow + numpy
```

Optional extras, installed only if you need them:

```bash
pip install -e '.[caption]'   # WD14 tagging (onnxruntime, huggingface-hub)
pip install -e '.[cluster]'   # CLIP embeddings / clustering / text filter (torch, open-clip)
pip install -e '.[vlm]'       # Florence-2 captions (transformers)
pip install -e '.[parquet]'   # parquet manifest export
pip install -e '.[cv]'        # OpenCV-based quality scorers
pip install -e '.[face]'      # identity / face analysis (insightface)
pip install -e '.[train]'     # LoRA / DreamBooth training + eval (torch, diffusers)
pip install -e '.[restore]'   # restoration post-processing (Real-ESRGAN, GFPGAN)
pip install -e '.[heif]'      # HEIC/HEIF photo ingest
pip install -e '.[viz]'       # UMAP / HDBSCAN embedding maps
pip install -e '.[all]'       # everything
pip install -e '.[dev]'       # pytest, ruff
```

Run as an installed command or as a module:

```bash
pixel-curator --help
python -m pixel_curator --help
```

---

## Quickstart (likeness training)

```bash
# 1. scaffold a project file and check your machine
pixel-curator init
pixel-curator -p myproject.toml doctor

# 2. ingest: video frames and/or photos of your subject
pixel-curator -p myproject.toml extract
pixel-curator -p myproject.toml photos

# 3. cull and score
pixel-curator -p myproject.toml prefilter
pixel-curator -p myproject.toml curate

# 4. captions + CLIP clustering (recommended for likeness training)
pixel-curator -p myproject.toml caption
pixel-curator -p myproject.toml cluster

# 5. inspect what survived
pixel-curator -p myproject.toml stats
pixel-curator -p myproject.toml review export --out review_round_1
open review_round_1/index.html
```

Then pick a slice and package it:

```bash
# DreamBooth-ready instance/ folder
pixel-curator -p myproject.toml export select --out curated --top-pct 20 --mode copy
pixel-curator -p myproject.toml package kohya \
  --dataset curated \
  --output dreambooth_pkg \
  --mode dreambooth --apply

# Or LoRA-ready concept_/ folders
pixel-curator -p myproject.toml package kohya \
  --dataset curated \
  --output lora_pkg \
  --mode lora --apply
```

---

## Semantic text filtering

Uses CLIP text encoding against cached image embeddings (run `cluster` once first). Dry-run by default:

```bash
# keep only beach shots
pixel-curator -p myproject.toml filter --text "beach photos" --threshold 0.25

# drop anything with visible text/watermarks
pixel-curator -p myproject.toml filter --text "image with text or watermark" --invert

# commit the changes (default is a report only)
pixel-curator -p myproject.toml filter --text "beach photos" --apply
```

---

## Visual review round-trip

```bash
pixel-curator -p myproject.toml review export --out review1
open review1/index.html          # thumbnail grid, score histogram, status chips
# fill the 'decision' column in review1/review_sheet.csv: keep / reject
pixel-curator -p myproject.toml review import --sheet review1/review_sheet.csv           # dry-run
pixel-curator -p myproject.toml review import --sheet review1/review_sheet.csv --apply   # commit
```

`reject` quarantines a frame (recoverable); `keep` marks it as kept. The HTML references originals via `file://` URLs — nothing is copied, so it works on very large libraries.

This is also the workflow for trimming class / regularization images, which is one of the more effective fixes for weak likeness results.

---

## Models

Core scoring needs no models. Optional capabilities download or locate their own:

| Feature | Model | Setup |
|---|---|---|
| WD14 captions | `SmilingWolf/wd-swinv2-tagger-v3` (~450 MB) | auto-downloads from HuggingFace on first run |
| CLIP clustering / text filter | ViT-B-32 (open_clip, laion2b) | auto-downloads on first run |
| Identity scoring | InsightFace `buffalo_l` (+ optional `inswapper_128.onnx`) | auto-downloads via `insightface` |
| Florence-2 captions | `microsoft/Florence-2-base` or local fine-tunes | place fine-tunes in `pixel_curator/models/vlm/florence2/` |
| Face restoration | `GFPGANv1.4.pth` | place in `pixel_curator/models/gfpgan/` |
| Upscaling | `RealESRGAN_x4plus.pth` | place in `pixel_curator/models/realesrgan/` |

Model weights are gitignored — the repo never carries them.

---

## Compute providers (CUDA / CoreML / CPU)

Everything auto-detects: `auto` picks CUDA → MPS/CoreML → CPU depending on the machine, so the same repo runs on an NVIDIA/Linux box, a Colab A100, or Apple Silicon unchanged. On CUDA, TF32 and bf16/fp16 autocast are enabled automatically.

Pin a dedicated provider in the project file:

```toml
[runtime]
device = "cuda"        # torch:      auto | cuda | mps | cpu
provider = "coreml"    # onnxruntime: auto | cuda | coreml | cpu
precision = "auto"     # auto | float32 | float16 | bfloat16
allow_tf32 = true
```

`device` drives torch workloads (CLIP, Florence-2); `provider` drives ONNX workloads (WD14 tagging, InsightFace likeness, orientation, quality signals). An unavailable choice degrades to CPU with a note — verify with:

```bash
pixel-curator -p myproject.toml doctor    # shows device, dtype, providers
```

---

## Project files

`pixel-curator init` writes a starter TOML (JSON on Python 3.10). A minimal project:

```toml
name = "my-archive"
output_dir = "./outputs"
video_dirs = ["~/Videos/shoots"]
photo_dirs = ["~/Pictures/library"]

[extract]
min_frames_per_video = 4

[curate]
# scoring/filter thresholds...

[cluster]
k = 0            # 0 = auto
batch_size = 32
```

Every command takes `-p/--project`; most also accept `--json` for scripting.

---

## Design guarantees

- **Source media is never modified** — extraction, scoring, and export only read originals; selection exports write to a new folder.
- **Resumable** — the SQLite manifest tracks what's done; re-running a stage picks up where it stopped.
- **Graceful degradation** — every optional model/backend falls back with a clear warning instead of crashing.
- **Dry-run first** — status-changing commands report what they would do unless you pass `--apply`.

---

## Tests

```bash
pixel-curator selftest            # smoke checks: schema, export, review, logging
python -m pytest                  # unit tests (pip install -e '.[dev]')
ruff check pixel_curator          # lint
```

---

## Scope note

The curation core — extract, score, dedupe, cluster, caption, filter, review, export — is content-agnostic and will work on any image archive. But every default in the tool is chosen with likeness training in mind: face-aware quality scoring, identity-likeness evaluation, class-image trimming via the review round-trip, and DreamBooth / LoRA export as the two first-class output formats. If you're training a subject or a style model, you're the target user.

## License

MIT — see [LICENSE](LICENSE).
