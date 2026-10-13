"""Phase 5.1/5.3: kohya sd-scripts adapter - command builder, live loss
parsing, run tracking in the manifest.

The studio drives `sdxl_train_network.py` as a subprocess, parses its tqdm
progress lines for steps/epoch/loss, and records everything in the `runs`
and `run_metrics` tables. Checkpoints land as
  {lora_output_dir}/{token}_{preset}_v{dataset:03d}.safetensors  (+ per-epoch)
with a run-metadata JSON next to each.

Stop semantics: the UI queue's Stop closes this generator; the subprocess is
terminated in the finally block (current epoch's last saved checkpoint
remains on disk - sd-scripts saves every epoch).

Adapter interface: `trainer_cmd` override lets tests (or alternative
backends like OneTrainer) substitute the executable while keeping queue,
parsing and bookkeeping identical.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path

from .. import manifest
from ..config import Project
from ..util import now_iso, safe_slug, setup_logging
from .presets import COMMON_ARGS, COMMON_FLAGS, MPS_ENV, PRESETS

logger = logging.getLogger(__name__)

# tqdm-ish progress: "steps:  3%|..| 80/2500 [00:41<20:00, loss=0.0911]"
RE_STEP = re.compile(r"(\d+)\s*/\s*(\d+)")
RE_LOSS = re.compile(r"(?:avr_)?loss[=:]\s*([0-9]*\.?[0-9]+)")
RE_EPOCH = re.compile(r"epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
# sd-scripts checkpoint save lines ("saving checkpoint: /path/x.safetensors")
RE_SAVE = re.compile(r"saving checkpoint:?\s*(\S+\.safetensors)", re.IGNORECASE)

DISK_CHECK_EVERY = 120.0          # mid-run boot-volume probe interval (s)


@dataclass
class TrainConfig:
    output_base: Path
    dataset_version: int
    preset: str = "character"
    name: str = ""                    # auto: {token}_{preset}_v{NNN}
    overrides: dict = field(default_factory=dict)   # arg-name -> value
    dry_run: bool = False
    trainer_cmd: list[str] = field(default_factory=list)  # test/backend hook
    min_free_gb: float = 20.0     # boot-volume preflight floor (0 disables)
    register_experiment: bool = False   # Phase 8: auto-register on success


def train_available(prj: Project) -> tuple[bool, str]:
    if not prj.sd_scripts_dir:
        return False, "sd_scripts_dir not set in project file"
    script = Path(prj.sd_scripts_dir).expanduser() / "sdxl_train_network.py"
    if not script.exists():
        return False, f"sdxl_train_network.py not found in {prj.sd_scripts_dir}"
    if not prj.base_model:
        return False, "base_model not set in project file (Pony V6 XL checkpoint path)"
    return True, "ok"


def _dataset_dir(conn, version: int) -> Path:
    row = conn.execute(
        "SELECT dir FROM datasets WHERE version = ?", (version,)
    ).fetchone()
    if not row:
        raise KeyError(f"dataset v{version} not found - run a build first")
    return Path(row[0])


def build_command(prj: Project, cfg: TrainConfig, dataset_dir: Path,
                  run_name: str, run_dir: Path) -> list[str]:
    preset = PRESETS[cfg.preset]
    lora_out = Path(prj.lora_output_dir or (Path(prj.output_base) / "LORA_OUTPUT"))

    if cfg.trainer_cmd:
        cmd = list(cfg.trainer_cmd)
    else:
        python = prj.sd_scripts_python or "python"   # dedicated venv when set
        cmd = [
            str(Path(python).expanduser()) if "/" in python else python,
            str(Path(prj.sd_scripts_dir).expanduser() / "sdxl_train_network.py"),
        ]

    args: dict[str, object] = {
        "pretrained_model_name_or_path": prj.base_model,
        "train_data_dir": str(dataset_dir / "img"),
        "output_dir": str(lora_out),
        "output_name": run_name,
        "logging_dir": str(run_dir / "logs"),
        "network_module": "networks.lora",
        "network_dim": preset["network_dim"],
        "network_alpha": preset["network_alpha"],
        "learning_rate": preset["unet_lr"],
        "train_batch_size": preset["batch_size"],
        "max_train_epochs": preset["epochs"],
        "min_snr_gamma": preset["min_snr_gamma"],
        **COMMON_ARGS,
    }
    flags = list(COMMON_FLAGS)
    if preset["te_lr"] and float(preset["te_lr"]) > 0:
        args["text_encoder_lr"] = preset["te_lr"]
    else:
        flags.append("network_train_unet_only")
    if preset.get("caption_dropout_rate"):
        args["caption_dropout_rate"] = preset["caption_dropout_rate"]

    # val_img present? sd-scripts has no native val split; noted in metadata.
    args.update(cfg.overrides or {})

    # MPS: HF Accelerate rejects fp16/bf16 *mixed precision* on Apple
    # Silicon ("fp16 mixed precision requires a GPU"). Training math runs
    # fp32 on MPS; save_precision stays fp16 so checkpoints are unchanged.
    # CUDA/Linux keeps the preset value. Explicit --set wins either way.
    if "mixed_precision" not in (cfg.overrides or {}):
        from ..runtime import get_device_name
        if get_device_name("auto") == "mps" and \
                str(args.get("mixed_precision")) in ("fp16", "bf16"):
            args["mixed_precision"] = "no"

    # MPS: native scaled-dot-product attention is faster and far leaner
    # on unified memory than the legacy mem_eff_attn implementation.
    from ..runtime import get_device_name as _gdn
    if _gdn("auto") == "mps" and "mem_eff_attn" in flags:
        flags[flags.index("mem_eff_attn")] = "sdpa"

    for k, v in args.items():
        cmd.extend([f"--{k}", str(v)])
    for f in flags:
        cmd.append(f"--{f}")
    return cmd


def train_generator(prj: Project, cfg: TrainConfig) -> Generator[str, None, None]:
    setup_logging(cfg.output_base)
    conn = manifest.connect(cfg.output_base)

    if cfg.preset not in PRESETS:
        yield f"FATAL: unknown preset '{cfg.preset}'. Options: {', '.join(PRESETS)}"
        return
    if not cfg.trainer_cmd:
        # Disk preflight first - a full boot volume is the most actionable
        # failure (swap + MPSGraph temp starvation SIGKILLed RUN #2/#3).
        if not cfg.dry_run and cfg.min_free_gb > 0:
            from ..monitor import disk_guard
            ok, report = disk_guard(prj, cfg.min_free_gb, cfg.output_base)
            yield report
            if not ok:
                return
        ok, reason = train_available(prj)
        if not ok:
            yield (f"Training unavailable: {reason}\n"
                   "Set sd_scripts_dir / base_model / lora_output_dir in the "
                   "project file (Settings tab or editor).")
            return
    try:
        dataset_dir = _dataset_dir(conn, cfg.dataset_version)
    except KeyError as exc:
        yield f"FATAL: {exc}"
        return

    run_name = cfg.name or (
        f"{safe_slug(prj.trigger_token)}_{cfg.preset}_v{cfg.dataset_version:03d}"
    )
    run_dir = cfg.output_base / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(prj, cfg, dataset_dir, run_name, run_dir)
    cmd_str = " ".join(cmd)

    if cfg.dry_run:
        yield f"[DRY RUN] {cmd_str}\n\nPreset '{cfg.preset}': {PRESETS[cfg.preset]}"
        return

    preset = PRESETS[cfg.preset]
    cur = conn.execute(
        "INSERT INTO runs(name, dataset_version, preset, status, config_json, "
        "command, output_dir, started_at, total_epochs) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (run_name, cfg.dataset_version, cfg.preset, "running",
         json.dumps({"preset": preset, "overrides": cfg.overrides}),
         cmd_str, str(run_dir), now_iso(), int(preset["epochs"])),
    )
    run_id = cur.lastrowid
    conn.commit()

    # MPS env vars only on Apple Silicon - never pollute a CUDA/Linux run
    from ..runtime import is_mps_available
    env = ({**os.environ, **MPS_ENV} if is_mps_available()
           else dict(os.environ))
    log_path = run_dir / "train.log"
    yield (f"RUN #{run_id} '{run_name}' starting\n"
           f"dataset v{cfg.dataset_version:03d} | preset {cfg.preset} "
           f"(dim {preset['network_dim']}/{preset['network_alpha']}, "
           f"lr {preset['unet_lr']}, batch {preset['batch_size']}, "
           f"{preset['epochs']} epochs)\nlog: {log_path}\n")

    proc: subprocess.Popen | None = None
    status = "failed"
    error: str | None = None
    step = total = epoch = 0
    last_loss: float | None = None
    last_yield = 0.0
    last_disk_check = time.time()
    disk_warned = False
    tail: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd, env=env, cwd=prj.sd_scripts_dir or None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        with open(log_path, "w", encoding="utf-8") as logf:
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                line_s = line.strip()
                if not line_s:
                    continue
                tail.append(line_s[-160:])
                tail = tail[-12:]

                m = RE_EPOCH.search(line_s)
                if m:
                    epoch = int(m.group(1))
                m = RE_STEP.search(line_s)
                if m:
                    step, total = int(m.group(1)), int(m.group(2))
                m = RE_LOSS.search(line_s)
                if m:
                    last_loss = float(m.group(1))
                    conn.execute(
                        "INSERT OR REPLACE INTO run_metrics(run_id, step, loss, ts) "
                        "VALUES (?,?,?,?)",
                        (run_id, step, last_loss, now_iso()),
                    )

                m = RE_SAVE.search(line_s)
                if m:
                    ckpt = Path(m.group(1)).name
                    from ..monitor import notify_macos
                    notify_macos("pixel-curator",
                                 f"Checkpoint saved: {ckpt} "
                                 f"(epoch {epoch}, loss {last_loss})")
                    yield (f"*** CHECKPOINT SAVED: {ckpt} "
                           f"(epoch {epoch}/{preset['epochs']})")

                now = time.time()
                if now - last_disk_check >= DISK_CHECK_EVERY:
                    last_disk_check = now
                    from ..monitor import WARN_FREE_GB, free_gb, notify_macos
                    boot = free_gb()
                    if (boot is not None and boot < WARN_FREE_GB
                            and not disk_warned):
                        disk_warned = True
                        warn = (f"WARNING: boot volume down to {boot:.1f} GB "
                                "free - macOS may kill this run. Free space "
                                "NOW (Trash, caches) without stopping training.")
                        notify_macos("pixel-curator", warn, sound="Basso")
                        yield f"*** {warn}"
                if now - last_yield >= 1.5:
                    last_yield = now
                    conn.execute(
                        "UPDATE runs SET current_step=?, total_steps=?, "
                        "current_epoch=?, last_loss=? WHERE run_id=?",
                        (step, total, epoch, last_loss, run_id),
                    )
                    conn.commit()
                    pct = f"{100*step/total:.1f}%" if total else "-"
                    yield (
                        f"RUN #{run_id} '{run_name}' | epoch {epoch}/{preset['epochs']} "
                        f"| step {step}/{total} ({pct}) | loss {last_loss}\n\n"
                        + "\n".join(tail)
                    )
        rc = proc.wait()
        if rc == 0:
            status = "completed"
        else:
            error = f"exit code {rc}; see {log_path}"
    except GeneratorExit:
        status = "stopped"
        raise
    except Exception as exc:
        error = str(exc)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        conn.execute(
            "UPDATE runs SET status=?, finished_at=?, current_step=?, "
            "total_steps=?, current_epoch=?, last_loss=?, error=? WHERE run_id=?",
            (status, now_iso(), step, total, epoch, last_loss, error, run_id),
        )
        conn.commit()
        # run metadata next to checkpoints
        meta = {
            "run_id": run_id, "name": run_name, "status": status,
            "dataset_version": cfg.dataset_version, "preset": cfg.preset,
            "preset_params": preset, "overrides": cfg.overrides,
            "final_loss": last_loss, "steps": step, "finished_at": now_iso(),
            "command": cmd_str,
        }
        try:
            (run_dir / "run.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
            pass

    if status == "completed":
        lora_out = Path(prj.lora_output_dir or (Path(prj.output_base) / "LORA_OUTPUT"))
        yield (
            f"\nRUN #{run_id} COMPLETED.\n"
            f"Final loss: {last_loss} | steps: {step}\n"
            f"Checkpoints: {lora_out}/{run_name}*.safetensors (per epoch)\n"
            f"Metadata: {run_dir / 'run.json'}\n"
            "\nNext: Phase 6 eval - test grids per epoch via Forge API."
        )
        # Phase 8: optional experiment record (pure metadata; a failure
        # here must never taint a completed training run)
        if cfg.register_experiment:
            try:
                from ..experiments import experiment_from_run
                rec, warns = experiment_from_run(conn, run_id,
                                                 lora_dir=lora_out)
                for w in warns:
                    yield f"experiment: {w}"
                if rec:
                    yield (f"Experiment #{rec['exp_id']} registered "
                           f"(dataset v{cfg.dataset_version:03d}). "
                           f"Notes: pixel-curator exp note {rec['exp_id']} "
                           "\"...\"")
            except Exception as exc:                # pragma: no cover
                yield f"experiment registration skipped: {exc}"
    else:
        yield f"\nRUN #{run_id} {status.upper()}. {error or ''}\nLog: {log_path}"


def list_runs(conn) -> list[dict]:
    return [
        {"run_id": r[0], "name": r[1], "dataset_version": r[2], "preset": r[3],
         "status": r[4], "started_at": r[5], "finished_at": r[6],
         "step": r[7], "total_steps": r[8], "epoch": r[9],
         "total_epochs": r[10], "last_loss": r[11], "error": r[12] or ""}
        for r in conn.execute(
            "SELECT run_id, name, dataset_version, preset, status, started_at, "
            "finished_at, current_step, total_steps, current_epoch, total_epochs, "
            "last_loss, error FROM runs ORDER BY run_id"
        )
    ]


def run_metrics(conn, run_id: int, max_points: int = 400) -> list[tuple[int, float]]:
    rows = conn.execute(
        "SELECT step, loss FROM run_metrics WHERE run_id=? ORDER BY step", (run_id,)
    ).fetchall()
    if len(rows) > max_points:           # downsample for the UI
        stride = len(rows) // max_points + 1
        rows = rows[::stride]
    return [(int(s), float(l)) for s, l in rows if l is not None]
