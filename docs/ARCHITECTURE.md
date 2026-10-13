# Architecture

One pipeline, one manifest. Everything is a CLI command that reads/writes a
SQLite manifest; there is no daemon, no GUI, no cloud.

## The pipeline DAG

```
                raw video / photo archives
                          |
                     extract / photos            (ingest; ffmpeg, HEIC-aware)
                          |
                      prefilter                  (technical rejects: tiny,
                          |                       corrupt, text overlays)
                     curate  ──────────────┐     (score: sharpness/brightness/
                          |                │      dHash + dedup + status)
              curate --smart (optional)    │     (InsightFace identity filter)
                          |                │
                quality (optional)         │     (deep signals; --workers N
                          |                │      parallel CPU path)
                   caption                 │     (WD14 tagger; ONNX or MPS)
                          |                │
         cluster / viz / filter (optional) ┘     (CLIP embeddings, k-means,
                          |                       text queries)
                  review export/import           (browser round-trip)
                          |
        export manifest / select / package       (CSV/JSONL, top-N% picks,
                          |                       Kohya / DreamBooth)
                train / eval (optional)          (LoRA training + likeness)
```

Every stage is **resumable**: the manifest records what's done, so a rerun
continues where the last one stopped. Every mutation is **dry-run first**:
`--apply` commits.

## Module map

| Package | Role |
|---|---|
| `cli/` | Thin entry (`__init__.py`), `dispatch.py` routing table, `commands/` handlers by domain, `parsers/` argparse registration by domain, `emitter.py` live progress |
| `manifest.py` | SQLite schema + migrations (`user_version`), stats, locking |
| `config.py` | Project TOML/JSON loading, dataclass configs per stage |
| `curate/` | 40+ scorers; catalog in `curate/scorer_registry.py` |
| `caption/` | WD14 taggers (ONNX `wd14.py`, MPS/timm `taggers.py`), rules engine `enhance.py` |
| `extract.py` / `prefilter.py` | Ingest |
| `export/` | Manifest export, selections, review round-trip |
| `train/` / `eval/` | Recipe builders, sd-scripts launchers, likeness eval |
| `studio/` | Generation workflows (artistic batches, figure study, learning) |
| `services/` | Longer-running orchestration (dataset explorer, training jobs, status writers) |
| `model_assets.py` | Model registry + first-run auto-download |
| `runtime.py` | Device policy (torch device, tf32, onnxruntime providers) |
| `logutil.py` | Package logging configuration (`--verbose`) |

## Design invariants

1. **Source media is never modified** - only read.
2. **Graceful degradation** - optional models/backends probe with
   import-only checks and fall back with a logged warning.
3. **Offline by default** - the only network calls are model downloads,
   always via `model_assets.ensure_model` (sha256-verified where direct).
4. **The manifest is the truth** - sidecars and JSONL outputs are derived
   views; the SQLite DB decides resume/status.
5. **stdlib-first** - heavy deps (torch, cv2, insightface, ...) live behind
   `pyproject.toml` extras; the core CLI runs on Pillow + numpy.

## CLI anatomy

```
pixel-curator <cmd>
   └─ cli/__init__.py: main()         error handling (locks, interrupts)
   └─ cli/parser.py: parse_args()     top-level flags + parsers/*.register()
   └─ cli/dispatch.py: run()          logging, runtime policy, Context
   └─ cli/commands/<group>.py         handler(ctx) per command
```

Adding a command touches three places (parser, handler, `HANDLERS` table);
`tests/test_cli.py::test_dispatch_covers_every_subcommand` enforces it.
