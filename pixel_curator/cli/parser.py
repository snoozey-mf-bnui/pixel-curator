"""Argument parser construction for the pixel-curator CLI."""

from __future__ import annotations

import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="pixel-curator",
        description="pixel-curator - extraction, curation, dataset packaging",
    )
    parser.add_argument("--project", "-p", default=None,
                        help="Project file (.toml or .json). Default: built-in paths")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="suppress live progress lines (summaries, "
                             "warnings and errors still print)")
    parser.add_argument("--no-color", action="store_true",
                        help="disable ANSI colors (also honors NO_COLOR)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="enable debug logging (also honors "
                             "PIXEL_CURATOR_DEBUG=1)")
    sub = parser.add_subparsers(dest="command")

    from .parsers import captioning, curation, exporting, ingest, ops, studio, training
    for _mod in (ops, ingest, curation, captioning, exporting, training,
                 studio):
        _mod.register(sub)

    return parser.parse_args(argv)



