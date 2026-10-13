"""Parser registration: ops commands."""

from __future__ import annotations

from ...config import DEFAULT_FPS_TIERS, DEFAULT_QUOTA, UI_PORT  # noqa: F401


def register(sub):
    sub.add_parser("doctor", help="Check dependencies")

    sub.add_parser("stats", help="Show manifest statistics")

    p_mod = sub.add_parser(
        "models", help="List required AI models and download missing ones "
                       "(first-run setup)")
    p_mod.add_argument("models_cmd", nargs="?", default="status",
                       choices=("status", "download"),
                       help="status (default) or download")
    p_mod.add_argument("--only", default=None,
                       help="comma-separated model keys to download "
                            "(default: everything missing)")
    p_mod.add_argument("--json", action="store_true")

    p_ov = sub.add_parser(
        "overview", help="One-screen status: frame funnel, caption/quality "
                         "coverage, pending work + next commands, paused "
                         "pipelines (read-only)")
    p_ov.add_argument("--output", default=None)
    p_ov.add_argument("--json", action="store_true")

    p_val = sub.add_parser(
        "validate", help="Data doctor: read-only manifest/dataset integrity "
                         "checks (caption drift, orphans, dataset mismatch)")
    p_val.add_argument("--output", default=None)
    p_val.add_argument("--json", action="store_true",
                       help="Emit the structured report as JSON")

    p_st = sub.add_parser(
        "selftest", help="Run the standalone test suite (tests/run_all.py) "
                         "and report a single pass/fail board")
    p_st.add_argument("filters", nargs="*",
                      help="substring filters on test filenames")
    p_st.add_argument("--timeout", type=int, default=300)

    p_init = sub.add_parser("init", help="Write a starter project file")
    p_init.add_argument("path", nargs="?", default="project.toml")

    p_hist = sub.add_parser(
        "history", help="Dataset version history: every registered version "
                        "(builds + folder exports) with lineage, change "
                        "summaries and curation snapshots")
    p_hist.add_argument("--dataset", type=int, default=None,
                        help="show the lineage chain of one version")
    p_hist.add_argument("--name", default=None,
                        help="filter by recipe / export name")
    p_hist.add_argument("--curation", action="store_true",
                        help="with --dataset: print the curation snapshot "
                             "captured when that version was created")
    p_hist.add_argument("--json", action="store_true")
    p_hist.add_argument("--output", default=None)

    p_hl = sub.add_parser("health",
                          help="Read-only dataset health report (HTML + JSON)")
    p_hl.add_argument("--output", default=None)
    p_hl.add_argument("--open", action="store_true", help="open the HTML report")

    p_df = sub.add_parser("diff", help="Diff two dataset versions "
                          "(membership + quality/duplicate/cluster deltas)")
    p_df.add_argument("a", type=int, nargs="?", default=None)
    p_df.add_argument("b", type=int, nargs="?", default=None)
    p_df.add_argument("--dataset", type=int, default=0,
                      help="new dataset version (visual diff mode)")
    p_df.add_argument("--vs", type=int, default=0,
                      help="baseline dataset version to compare against")
    p_df.add_argument("--json", action="store_true")
    p_df.add_argument("--export", default="",
                      help="write the structured diff JSON to a file")
    p_df.add_argument("--output", default=None)
    p_df.add_argument("--full", action="store_true", help="list every changed frame")

    sub.add_parser("runs", help="List training runs")

    # Phase 8 - Training Experiment Tracking

    p_su = sub.add_parser("status", help="Live training status: step/ETA/"
                          "loss/disk/checkpoints (reads the manifest; safe "
                          "to run while training)")
    p_su.add_argument("--json", action="store_true")
    p_su.add_argument("--follow", type=float, default=0, metavar="SECONDS",
                      help="poll every N seconds; macOS-notifies on new "
                           "epoch checkpoints, low disk, stalls and completion")
    p_su.add_argument("--no-notify", dest="notify", action="store_false",
                      default=True)
    p_su.add_argument("--stall-seconds", type=float, default=900.0,
                      help="alert when a 'running' run records no new step for "
                           "this many seconds (default 900 = 15 min)")
    p_su.add_argument("--output", default=None)

    p_wa = sub.add_parser("watch", help="Watch source folders; auto-ingest new media")
    p_wa.add_argument("--interval", type=int, default=300)
    p_wa.add_argument("--auto-curate", action="store_true")
    p_wa.add_argument("--once", action="store_true")
    p_wa.add_argument("--output", default=None)

    p_gc = sub.add_parser("gc", help="Disk cleanup (dry-run by default)")
    p_gc.add_argument("--apply", action="store_true")
    p_gc.add_argument("--keep-builds", type=int, default=2)
    p_gc.add_argument("--output", default=None)

    sub.add_parser("space", help="Disk usage dashboard")

    p_bk = sub.add_parser("backup", help="Archive manifest + presets + "
                                         "project file + LoRA sidecars")
    p_bk.add_argument("--out", default="")

    p_rs = sub.add_parser("restore", help="Extract a backup for manual "
                                          "placement (non-destructive)")
    p_rs.add_argument("archive")
    p_rs.add_argument("--dest", default="restored_backup")

    # Concept Control Layer

