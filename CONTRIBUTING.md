# Contributing

Thanks for working on pixel-curator. This document is the short version of
how to get a change in cleanly.

## Setup

```bash
git clone https://github.com/snoozey-mf-bnui/pixel-curator.git
cd pixel-curator
pip install -e '.[dev]'
pre-commit install        # optional but recommended
```

## The rules that matter

1. **Source media is never modified.** Extraction, scoring and export only
   read originals; selections write to new folders.
2. **Dry-run first.** Status-changing commands report what they *would* do
   unless `--apply` is passed. Keep that contract.
3. **Graceful degradation.** Every optional model/backend falls back with a
   clear warning instead of crashing. Probe with import-only checks
   (see `curate/scorer_registry.py` for the pattern).
4. **Resumability.** Long stages commit in batches so a rerun continues
   where the last one stopped.
5. **No silent failures.** Catch specific exceptions; if you must swallow,
   `logger.debug("...", exc_info=True)` it (surfaced with `--verbose`).

## Where things live

| Area | Location |
|---|---|
| CLI entry + dispatch | `pixel_curator/cli/` (`dispatch.py`, `commands/`, `parsers/`) |
| Curation scorers | `pixel_curator/curate/` (+ register in `scorer_registry.py`) |
| Manifest (SQLite schema) | `pixel_curator/manifest.py` |
| Model downloads | `pixel_curator/model_assets.py` |
| Config / project files | `pixel_curator/config.py` |

Adding a CLI command = parser block in `cli/parsers/<group>.py` +
handler in `cli/commands/<group>.py` + row in `cli/dispatch.py:HANDLERS`.
`tests/test_cli.py` fails until all three agree.

Adding a scorer = module in `curate/` + entry in
`curate/scorer_registry.py` (`curate --list-scorers` shows it).

## Checks before you commit

```bash
ruff check pixel_curator        # must be clean
python -m pytest                # unit + structural tests
pixel-curator selftest          # smoke suite (schema, export, review)
```

CI runs all of the above on Python 3.10 and 3.12.

## Style

- Python 3.10+, stdlib-first; heavy deps stay behind optional extras.
- `ruff` lints; the rules currently scoped off live in
  `pyproject.toml` with a note to re-enable them one at a time - if a
  change touches one of those areas, fixing the local violations is
  welcome.
- Log via `logging.getLogger(__name__)`; `print` is for deliberate CLI
  output only.
