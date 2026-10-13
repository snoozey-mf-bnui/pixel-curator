# Changelog

All notable changes to pixel-curator. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions come from `pixel_curator.__version__`.

## [Unreleased]

### Added
- `model_assets.py` - registry of all pipeline models with offline
  `status()` and idempotent first-run downloads; Real-ESRGAN x4 and
  GFPGAN v1.4 now auto-download (sha256-verified) instead of requiring
  manual SSD paths.
- `pixel-curator models` command (status / download / --only / --json).
- `curate --list-scorers` - scorer registry with import-probe availability
  (`curate/scorer_registry.py`).
- `quality --workers N` - process-pool parallel scoring for the plain
  CPU path (face-detector and advanced-signal passes stay sequential).
- Global `--verbose` flag + `logutil.py`; previously silent exception
  handlers now debug-log with tracebacks.
- pytest suite: scorer invariant tests on synthetic images, CLI
  structure tests, model-asset tests, requirements-sync test.
- CI: GitHub Actions (ruff, pytest, CLI smoke on 3.10/3.12), coverage
  reporting, release workflow, dependabot.
- `requirements_coreml.txt` / `requirements_cuda.txt` pinned environments.
- CONTRIBUTING.md, pre-commit config, docs/ARCHITECTURE.md.

### Changed
- CLI monolith split: `cli.py` (4,959 lines) is now a package -
  `parser` became `cli/parsers/` (7 modules), dispatch became
  `cli/commands/` (7 modules) behind a routing table.
- Caption-record consumers speak generic `captions` metadata instead of
  Florence-shaped records (visual focus classifiers, dataset explorer,
  status writer, classify command).
- 91 dead-variable sites cleaned up; ruff `F841` is now enforced.

### Removed
- **Florence-2 support** (`florence` and `vlm-caption` commands,
  `services/florence_*`, `caption/vlm.py`, local weights). Captioning is
  WD14-only (ONNX or MPS backend).
- Byte-identical duplicate modules (`aesthetic_scoring`,
  `dynamic_score_scorer`).
- Hardcoded external-SSD model paths.
- Name collisions resolved by rename: `curate/visual_focus_cv.py`,
  `studio/asset_registry.py`, `artistic/batch_review.py`.

## [3.31.0] - pre-refactor

Legacy history (pre-changelog): extraction, curation, clustering,
captioning, review, export/packaging, LoRA training + eval, studio
workflows. See git history for details.

[Unreleased]: https://github.com/snoozey-mf-bnui/pixel-curator/compare/v3.31.0...HEAD
[3.31.0]: https://github.com/snoozey-mf-bnui/pixel-curator/releases/tag/v3.31.0
