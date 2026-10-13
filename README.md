# pixel-curator

**Local-first CLI for turning raw photo/video archives into curated, scored, exportable datasets.** Strictly terminal-based: no GUI, no accounts, no cloud — every byte stays on your machine.

It was built to prepare LoRA training datasets, but the curation core is content-agnostic: extract frames, score quality, dedupe, caption, cluster by visual similarity, filter with natural language, review in a browser, and export the result in whatever shape you need — a flat manifest for a spreadsheet, a symlinked "best 10%" folder, or a training-ready Kohya/DreamBooth package.

## What it does

| Stage | Command | What happens |
|---|---|---|
| Set up | `init`, `doctor` | Scaffold a project file, verify dependencies (ffmpeg, models, backends) |
| Ingest | `extract`, `photos`, `smart-extract` | Frame extraction from video (HDR-aware, adaptive FPS, segmented + resumable) and photo import (JPEG/PNG/WebP/HEIC) |
| Filter | `prefilter` | Fast multiprocessed pass to drop technically unusable frames |
| Score | `curate` | Optical quality scoring, dedup (perceptual hash), status assignment |
| Cluster | `cluster`, `viz` | CLIP embeddings cached in the manifest, k-means clustering, embedding maps |
| Caption | `caption`, `florence` | WD14 tagger (auto-downloads), Florence-2 natural-language captions |
| Semantic filter | `filter` | Keep/drop frames matching a text query ("beach photos", "images with watermarks") |
| Review | `review export` / `review import` | Static HTML report + spreadsheet keep/reject round-trip |
| Export | `export manifest`, `export select`, `package` | Flat CSV/JSONL/Parquet manifest; top-N%/threshold/balanced selections (copy or symlink); Kohya LoRA and DreamBooth instance packages |
| Train & eval | `train`, `eval`, `likeness`, `pick-epoch` | Optional LoRA training and identity-likeness evaluation |
| Operate | `stats`, `health`, `backup`, `restore`, `selftest` | Manifest stats, pipeline health, safety nets, smoke tests |

Everything is **dry-run by default** where mutations are possible, and every long stage is **resumable** through a SQLite manifest that tracks sources, frames, scores, captions, embeddings, clusters, dataset versions, and full lineage.

## Install

Requires Python 3.10+ and ffmpeg (for video extraction).

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
pip install -e '.[parquet]'   # parquet manifest export (pandas, pyarrow)
pip install -e '.[cv]'        # OpenCV-based quality scorers (opencv, scikit-learn)
pip install -e '.[face]'      # identity / face analysis (insightface)
pip install -e '.[train]'     # LoRA / DreamBooth training + eval (torch, diffusers)
pip install -e '.[restore]'   # restoration post-processing (Real-ESRGAN, GFPGAN)
pip install -e '.[heif]'      # HEIC/HEIF photo ingest (pillow-heif)
pip install -e '.[viz]'       # UMAP / HDBSCAN embedding maps
pip install -e '.[all]'       # everything
pip install -e '.[dev]'       # contributor tooling (pytest, ruff)
```

The package version is read dynamically from `pixel_curator.__version__`, so
`pip show pixel-curator` and `pixel-curator --version` always agree.

Run either as an installed command or as a module:

```bash
pixel-curator --help
python -m pixel_curator --help
```

## Quickstart

```bash
# 1. scaffold a project file and check your machine
pixel-curator init
pixel-curator -p myproject.toml doctor

# 2. ingest: video frames and/or photos
pixel-curator -p myproject.toml extract
pixel-curator -p myproject.toml photos

# 3. cull and score
pixel-curator -p myproject.toml prefilter
pixel-curator -p myproject.toml curate

# 4. (optional) captions + CLIP clustering
pixel-curator -p myproject.toml caption
pixel-curator -p myproject.toml cluster

# 5. look at what survived
pixel-curator -p myproject.toml stats
pixel-curator -p myproject.toml review export --out review_round_1
open review_round_1/index.html
```

## Getting selections out (framework-agnostic)

**Flat manifest** — every frame, every signal, one row each. Open in any spreadsheet or dataframe tool:

```bash
pixel-curator -p myproject.toml export manifest --out manifest.csv
pixel-curator -p myproject.toml export manifest --out manifest.jsonl --format jsonl
pixel-curator -p myproject.toml export manifest --out manifest.parquet --format parquet
```

**Selection export** — materialize a slice into a folder. Symlink mode culls an 80 GB library without copying a byte:

```bash
# best 10% by unified score, as symlinks (zero-copy)
pixel-curator -p myproject.toml export select --out best10 --top-pct 10

# everything at or above score 0.8, real copies, with caption sidecars
pixel-curator -p myproject.toml export select --out heroes --min-score 0.8 --mode copy

# 500 frames spread evenly across CLIP clusters (run 'cluster' first)
pixel-curator -p myproject.toml export select --out balanced500 --balanced 500
```

The unified `score` is `integrity` (0–1 face-aware quality) when present, otherwise the percentile rank of sharpness — so thresholds mean the same thing regardless of which scorers ran. Every selection folder includes a `selection_manifest.csv` index mapping outputs back to sources.

**Training packages** — LoRA and DreamBooth are both first-class:

```bash
# LoRA (multi-concept): concept_<target>/ folders with dataset_config.toml
pixel-curator -p myproject.toml package kohya --dataset <export_dir> --output <pkg_dir> --mode lora --apply

# DreamBooth: a ready-to-train instance/ folder (000001.jpg + 000001.txt
# captions, dataset_config.toml) matching the standard Colab notebooks
pixel-curator -p myproject.toml package kohya --dataset <export_dir> --output <pkg_dir> --mode dreambooth --apply
```

The DreamBooth mode exports your curated selection as a numbered **instance image set** with matching caption files and a Kohya-style `dataset_config.toml` — point any DreamBooth trainer (Colab notebook, kohya sd-scripts, ai-toolkit) at the folder and train. Captions are assembled from the manifest's tags/quality metadata; no model calls needed at packaging time.

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

## Visual review round-trip

```bash
pixel-curator -p myproject.toml review export --out review1
open review1/index.html          # thumbnail grid, score histogram, status chips
# fill the 'decision' column in review1/review_sheet.csv: keep / reject
pixel-curator -p myproject.toml review import --sheet review1/review_sheet.csv           # dry-run
pixel-curator -p myproject.toml review import --sheet review1/review_sheet.csv --apply   # commit
```

`reject` sets a frame's status to `quarantined` (recoverable), `keep` sets `kept`. The HTML references your original files via `file://` URLs — nothing is copied, so it works on very large libraries.

## Models

Core scoring needs no models. Optional capabilities download or locate their own:

| Feature | Model | Setup |
|---|---|---|
| WD14 captions | `SmilingWolf/wd-swinv2-tagger-v3` (~450 MB) | auto-downloads from HuggingFace on first run |
| CLIP clustering / text filter | ViT-B-32 (open_clip, laion2b) | auto-downloads on first run |
| Identity scoring | InsightFace `buffalo_l` (+ optional `inswapper_128.onnx`) | auto-downloads via the `insightface` package |
| Florence-2 captions | `microsoft/Florence-2-base` or local fine-tunes | place fine-tunes in `pixel_curator/models/vlm/florence2/` |
| Face restoration | GFPGANv1.4.pth | place in `pixel_curator/models/gfpgan/` |
| Upscaling | RealESRGAN_x4plus.pth | place in `pixel_curator/models/realesrgan/` |

Model weights are gitignored — the repo never carries them.

## Compute providers (CUDA / CoreML / CPU)

Everything auto-detects: `auto` picks **CUDA → MPS/CoreML → CPU** depending on the machine, so the same repo runs on an NVIDIA/Linux box, a Colab A100, or Apple Silicon unchanged. On CUDA, TF32 and bf16/fp16 autocast are enabled automatically.

To pin a dedicated provider, set it once in your project file:

```toml
[runtime]
device = "cuda"        # torch:      auto | cuda | mps | cpu
provider = "coreml"    # onnxruntime: auto | cuda | coreml | cpu
precision = "auto"     # auto | float32 | float16 | bfloat16
allow_tf32 = true
```

`device` drives torch workloads (CLIP, Florence-2); `provider` drives ONNX workloads (WD14 tagging, InsightFace likeness, orientation, quality signals). An unavailable choice degrades to CPU with a note — check what was selected with:

```bash
pixel-curator -p myproject.toml doctor    # shows device, dtype, providers
```

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

## Tests

```bash
pixel-curator selftest            # smoke checks: schema, export, review, logging
python -m pytest                  # unit tests (pip install -e '.[dev]')
ruff check pixel_curator          # lint
```

## Design guarantees

- **Source media is never modified** — extraction, scoring, and export only read originals; selection exports write to a new folder.
- **Resumable** — the SQLite manifest tracks what's done; re-running a stage picks up where it stopped.
- **Graceful degradation** — every optional model/backend falls back with a clear warning instead of crashing.
- **Dry-run first** — status-changing commands report what they *would* do unless you pass `--apply`.

## License

MIT — see [LICENSE](LICENSE).
