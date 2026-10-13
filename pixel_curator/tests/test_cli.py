"""CLI structural tests: parser builds, dispatch table covers every command."""

from __future__ import annotations

from unittest import mock

import pytest

from pixel_curator.cli.dispatch import HANDLERS
from pixel_curator.cli.parser import parse_args


def _top_level_commands() -> set[str]:
    """Capture the ArgumentParser inside parse_args and list subcommands."""
    import argparse
    captured = {}
    real = argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        captured["parser"] = self
        raise SystemExit(0)

    with mock.patch.object(argparse.ArgumentParser, "parse_args", spy):
        try:
            parse_args([])
        except SystemExit:
            pass
    parser = captured["parser"]
    for action in parser._actions:
        if hasattr(action, "choices") and isinstance(action.choices, dict):
            return set(action.choices)
    raise AssertionError("no subparsers action found")


def test_parser_builds_and_defaults_to_ui():
    args = parse_args([])
    assert args.command is None or args.command == "ui"


def test_dispatch_covers_every_subcommand():
    assert _top_level_commands() == set(HANDLERS)


@pytest.mark.parametrize("cmd", sorted(HANDLERS))
def test_every_handler_is_parseable(cmd):
    """Each registered command must parse; commands with required args may
    legitimately fail with argparse usage error (exit code 2)."""
    try:
        args = parse_args([cmd])
        assert args.command == cmd
    except SystemExit as exc:
        assert exc.code in (0, 2), f"{cmd}: unexpected exit code {exc.code}"
