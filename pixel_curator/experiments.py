"""Phase 8: Training Experiment Tracking.

Artifact-lineage records linking trained models (.safetensors) to the
dataset versions (Phase 2 registry) they were trained on, plus training
parameters, outcomes and human notes.

Design notes
------------
* An *experiment* is metadata about a training artifact. It complements
  the ``runs`` table (live process tracking of the internal launcher):
  experiments also cover external trainings (Colab kohya / DreamBooth
  notebooks) whose .safetensors otherwise come back with zero lineage.
* Fully optional and non-intrusive: nothing in the existing pipeline
  requires an experiment record, registration never raises into a
  caller's workflow, and every write is append/update-only metadata.
* Human-readable layer: registration mirrors the Phase 2 ``version.json``
  pattern with a ``<model>.experiment.json`` sidecar next to the model,
  and dataset exports can carry a prefilled ``experiment_template.json``
  so a Colab round-trip closes lineage with one command:

      pixel-curator exp register --from-file <export>/experiment_template.json \\
          --model /path/to/trained.safetensors
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from .util import now_iso

log = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".experiment.json"
TEMPLATE_NAME = "experiment_template.json"

MODEL_TYPES = ("lora", "dreambooth", "checkpoint", "embedding", "other")

_COLS = ("exp_id", "name", "model_path", "model_hash", "model_type",
         "base_model", "dataset_version", "dataset_dir", "run_id",
         "params_json", "outcome", "rating", "notes", "source",
         "trained_at", "created_at", "updated_at")


# ---------------------------------------------------------------------------
# Hashing (models are multi-GB: default is a cheap fingerprint)
# ---------------------------------------------------------------------------

def quick_hash(path: Path | str, full: bool = False) -> Optional[str]:
    """Fingerprint a model file.

    Default 'quick:' mode hashes size + first/last MiB - enough to tell
    checkpoints apart without reading 6 GB. ``full=True`` streams the
    whole file (sha256:). Returns None when the file is unreadable.
    """
    try:
        p = Path(path)
        size = p.stat().st_size
        h = hashlib.sha256()
        with open(p, "rb") as f:
            if full:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
                return "sha256:" + h.hexdigest()[:16]
            h.update(str(size).encode())
            h.update(f.read(1 << 20))
            if size > (2 << 20):
                f.seek(-(1 << 20), 2)
                h.update(f.read(1 << 20))
        return "quick:" + h.hexdigest()[:16]
    except OSError as exc:
        log.debug("hash skipped for %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Row <-> dict
# ---------------------------------------------------------------------------

def _to_record(row) -> dict:
    rec = dict(zip(_COLS, row))
    try:
        rec["params"] = json.loads(rec.pop("params_json") or "{}")
    except Exception:
        rec["params"] = {}
    return rec


def get_experiment(conn: sqlite3.Connection,
                   key: int | str) -> Optional[dict]:
    """Look up by exp_id (int / numeric string) or exact name; falls back
    to a unique name-substring match."""
    cols = ", ".join(_COLS)
    if isinstance(key, int) or str(key).isdigit():
        row = conn.execute(
            f"SELECT {cols} FROM experiments WHERE exp_id = ?",
            (int(key),)).fetchone()
        if row:
            return _to_record(row)
    rows = conn.execute(
        f"SELECT {cols} FROM experiments WHERE name = ?",
        (str(key),)).fetchall()
    if not rows:
        rows = conn.execute(
            f"SELECT {cols} FROM experiments WHERE name LIKE ? "
            "ORDER BY exp_id", (f"%{key}%",)).fetchall()
    if len(rows) == 1:
        return _to_record(rows[0])
    return None


def list_experiments(conn: sqlite3.Connection, *,
                     dataset_version: Optional[int] = None,
                     model: Optional[str] = None,
                     outcome: Optional[str] = None,
                     since: Optional[str] = None,
                     limit: int = 0) -> list[dict]:
    where, params = [], []
    if dataset_version is not None:
        where.append("dataset_version = ?")
        params.append(int(dataset_version))
    if model:
        where.append("(name LIKE ? OR model_path LIKE ?)")
        params.extend([f"%{model}%", f"%{model}%"])
    if outcome:
        where.append("outcome = ?")
        params.append(outcome)
    if since:
        where.append("created_at >= ?")
        params.append(since)
    sql = f"SELECT {', '.join(_COLS)} FROM experiments"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY exp_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [_to_record(r) for r in conn.execute(sql, params)]


# ---------------------------------------------------------------------------
# Registration / update / notes
# ---------------------------------------------------------------------------

def _stamp_note(text: str) -> str:
    return f"[{now_iso()}] {text.strip()}"


def register_experiment(conn: sqlite3.Connection, *,
                        name: str,
                        model_path: Optional[str] = None,
                        model_type: str = "lora",
                        base_model: Optional[str] = None,
                        dataset_version: Optional[int] = None,
                        dataset_dir: Optional[str] = None,
                        run_id: Optional[int] = None,
                        params: Optional[dict] = None,
                        outcome: Optional[str] = None,
                        rating: Optional[int] = None,
                        note: str = "",
                        source: str = "manual",
                        trained_at: Optional[str] = None,
                        hash_model: bool = True,
                        full_hash: bool = False) -> dict:
    """Insert an experiment record. Returns the record dict with a
    ``warnings`` list (dangling dataset version, missing model file...).
    Never destructive."""
    warnings: list[str] = []
    if model_type not in MODEL_TYPES:
        warnings.append(f"unknown model_type '{model_type}' "
                        f"(expected one of {', '.join(MODEL_TYPES)}) - "
                        "stored as-is")
    mpath = str(Path(model_path).expanduser()) if model_path else None
    mhash = None
    if mpath:
        if Path(mpath).is_file():
            if hash_model:
                mhash = quick_hash(mpath, full=full_hash)
        else:
            warnings.append(f"model file not found yet: {mpath} "
                            "(record kept; update with `exp set` later)")
    if dataset_version is not None:
        row = conn.execute("SELECT dir FROM datasets WHERE version = ?",
                           (int(dataset_version),)).fetchone()
        if not row:
            warnings.append(f"dataset v{int(dataset_version):03d} is not in "
                            "the version registry; link recorded anyway")
        elif not dataset_dir:
            dataset_dir = row[0]
    ts = now_iso()
    notes = _stamp_note(note) if note.strip() else ""
    cur = conn.execute(
        "INSERT INTO experiments(name, model_path, model_hash, model_type, "
        "base_model, dataset_version, dataset_dir, run_id, params_json, "
        "outcome, rating, notes, source, trained_at, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (name, mpath, mhash, model_type, base_model,
         int(dataset_version) if dataset_version is not None else None,
         dataset_dir, int(run_id) if run_id is not None else None,
         json.dumps(params or {}, ensure_ascii=False),
         outcome, int(rating) if rating is not None else None,
         notes, source, trained_at, ts, ts))
    conn.commit()
    rec = get_experiment(conn, int(cur.lastrowid)) or {}
    rec["warnings"] = warnings
    sidecar = write_experiment_sidecar(rec)
    if sidecar:
        rec["sidecar"] = str(sidecar)
    return rec


_UPDATABLE = ("name", "model_path", "model_hash", "model_type", "base_model",
              "dataset_version", "dataset_dir", "run_id", "outcome",
              "rating", "trained_at")


def update_experiment(conn: sqlite3.Connection, exp_id: int,
                      params: Optional[dict] = None,
                      hash_model: bool = True,
                      full_hash: bool = False,
                      **fields) -> Optional[dict]:
    """Update selected fields (merge for params). Returns the fresh record
    or None when the experiment does not exist."""
    rec = get_experiment(conn, exp_id)
    if not rec:
        return None
    sets, vals = [], []
    unknown = [k for k in fields if k not in _UPDATABLE]
    if unknown:
        raise ValueError(f"not updatable: {', '.join(unknown)}")
    for k, v in fields.items():
        if k == "model_path" and v:
            v = str(Path(v).expanduser())
        sets.append(f"{k} = ?")
        vals.append(v)
    if fields.get("model_path"):
        p = Path(str(fields["model_path"])).expanduser()
        if p.is_file() and hash_model:
            sets.append("model_hash = ?")
            vals.append(quick_hash(p, full=full_hash))
    if params:
        merged = dict(rec.get("params") or {})
        merged.update(params)
        sets.append("params_json = ?")
        vals.append(json.dumps(merged, ensure_ascii=False))
    sets.append("updated_at = ?")
    vals.append(now_iso())
    vals.append(int(rec["exp_id"]))
    conn.execute(f"UPDATE experiments SET {', '.join(sets)} "
                 "WHERE exp_id = ?", vals)
    conn.commit()
    fresh = get_experiment(conn, int(rec["exp_id"]))
    if fresh:
        write_experiment_sidecar(fresh)
    return fresh


def append_note(conn: sqlite3.Connection, exp_id: int,
                text: str) -> Optional[dict]:
    rec = get_experiment(conn, exp_id)
    if not rec or not text.strip():
        return rec
    notes = (rec.get("notes") or "").rstrip()
    notes = (notes + "\n" if notes else "") + _stamp_note(text)
    conn.execute("UPDATE experiments SET notes = ?, updated_at = ? "
                 "WHERE exp_id = ?", (notes, now_iso(), int(rec["exp_id"])))
    conn.commit()
    fresh = get_experiment(conn, int(rec["exp_id"]))
    if fresh:
        write_experiment_sidecar(fresh)
    return fresh


# ---------------------------------------------------------------------------
# Promotion from an internal training run
# ---------------------------------------------------------------------------

def experiment_from_run(conn: sqlite3.Connection, run_id: int, *,
                        lora_dir: Optional[Path] = None,
                        note: str = "") -> tuple[Optional[dict], list[str]]:
    """Promote a ``runs`` row into an experiment (idempotent per run)."""
    row = conn.execute(
        "SELECT run_id, name, dataset_version, preset, status, config_json, "
        "output_dir, finished_at, last_loss, total_epochs "
        "FROM runs WHERE run_id = ?", (int(run_id),)).fetchone()
    if not row:
        return None, [f"run #{run_id} not found"]
    (rid, rname, dsv, preset, status, config_json,
     output_dir, finished_at, last_loss, total_epochs) = row
    existing = conn.execute(
        "SELECT exp_id FROM experiments WHERE run_id = ?", (rid,)).fetchone()
    if existing:
        return (get_experiment(conn, int(existing[0])),
                [f"run #{rid} is already experiment #{existing[0]}"])
    try:
        params = json.loads(config_json or "{}")
    except Exception:
        params = {}
    params.setdefault("preset", preset)
    if last_loss is not None:
        params["final_loss"] = last_loss
    if total_epochs:
        params.setdefault("epochs", total_epochs)
    model_path = None
    if lora_dir:
        cands = sorted(Path(lora_dir).glob(f"{rname}*.safetensors"))
        if cands:
            model_path = str(cands[-1])
    rec = register_experiment(
        conn, name=rname, model_path=model_path,
        dataset_version=dsv, run_id=rid, params=params,
        source="run", trained_at=finished_at,
        note=note or f"promoted from training run #{rid} ({status})")
    return rec, rec.get("warnings", [])


# ---------------------------------------------------------------------------
# Human-readable layer: sidecars + export templates
# ---------------------------------------------------------------------------

def write_experiment_sidecar(rec: dict) -> Optional[Path]:
    """``<model>.experiment.json`` beside the model file. Never raises."""
    mpath = rec.get("model_path")
    if not mpath:
        return None
    try:
        p = Path(mpath)
        if not p.parent.is_dir():
            return None
        payload = {k: v for k, v in rec.items()
                   if k not in ("warnings", "sidecar")}
        out = p.parent / (p.stem + SIDECAR_SUFFIX)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        return out
    except Exception as exc:                        # pragma: no cover
        log.warning("experiment sidecar not written: %s", exc)
        return None


def write_experiment_template(export_dir: Path | str, *,
                              name: str,
                              dataset_version: Optional[int] = None,
                              dataset_dir: Optional[str] = None,
                              params: Optional[dict] = None,
                              base_model: Optional[str] = None,
                              model_type: str = "lora") -> Optional[Path]:
    """Drop a prefilled ``experiment_template.json`` into a dataset export.

    The template travels with the export (e.g. into a Colab zip); after
    training, ``exp register --from-file <template> --model <file>``
    closes the dataset->model lineage. Never raises.
    """
    try:
        d = Path(export_dir)
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "_help": ("After training, register this experiment with: "
                      "pixel-curator exp register --from-file "
                      f"{TEMPLATE_NAME} --model /path/to/model.safetensors"),
            "name": name,
            "model_type": model_type,
            "base_model": base_model,
            "dataset_version": dataset_version,
            "dataset_dir": str(dataset_dir) if dataset_dir else str(d),
            "params": params or {},
            "created_at": now_iso(),
        }
        out = d / TEMPLATE_NAME
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        return out
    except Exception as exc:                        # pragma: no cover
        log.warning("experiment template not written for %s: %s",
                    export_dir, exc)
        return None


def load_experiment_file(path: Path | str) -> dict:
    """Read a template or a previously written sidecar into register()
    keyword arguments. Raises ValueError on unusable files."""
    p = Path(path).expanduser()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read experiment file {p}: {exc}")
    if not isinstance(data, dict):
        raise ValueError(f"{p} does not contain an experiment object")
    out = {
        "name": data.get("name") or p.parent.name,
        "model_type": data.get("model_type") or "lora",
        "base_model": data.get("base_model"),
        "dataset_version": data.get("dataset_version"),
        "dataset_dir": data.get("dataset_dir"),
        "params": data.get("params") or {},
        "model_path": data.get("model_path"),
    }
    if data.get("trained_at"):
        out["trained_at"] = data["trained_at"]
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _ds_tag(rec: dict) -> str:
    v = rec.get("dataset_version")
    return f"v{int(v):03d}" if v is not None else "-"


def render_experiments(records: list[dict],
                       extra_runs: Optional[list[dict]] = None) -> str:
    if not records and not extra_runs:
        return ("No experiments registered yet.\n"
                "Register one: pixel-curator exp register --name X "
                "--model /path/model.safetensors --dataset N")
    lines = ["", f"{'id':>4}  {'name':<30} {'dataset':<8} {'type':<10} "
                 f"{'outcome':<10} {'model':<6} created"]
    lines.append("-" * 84)
    for r in records:
        mp = r.get("model_path")
        model = "ok" if (mp and Path(mp).is_file()) else \
            ("miss" if mp else "-")
        out = r.get("outcome") or "-"
        if r.get("rating"):
            out += f" *{r['rating']}"
        lines.append(
            f"  #{r['exp_id']:<3} {r['name'][:30]:<30} {_ds_tag(r):<8} "
            f"{(r.get('model_type') or '-'):<10} {out:<10} {model:<6} "
            f"{(r.get('created_at') or '')[:10]}")
    if extra_runs:
        lines.append("")
        lines.append(f"  + {len(extra_runs)} internal training run(s) for "
                     "this dataset without an experiment record:")
        for r in extra_runs:
            lines.append(f"      run #{r['run_id']} {r['name']} "
                         f"[{r['status']}]  -> pixel-curator exp from-run "
                         f"{r['run_id']}")
    return "\n".join(lines)


def render_experiment_detail(conn: sqlite3.Connection, rec: dict) -> str:
    lines = ["", f"EXPERIMENT #{rec['exp_id']}: {rec['name']}",
             "=" * 60]
    mp = rec.get("model_path")
    state = ""
    if mp:
        state = "  [on disk]" if Path(mp).is_file() else "  [MISSING]"
    lines.append(f"model     : {mp or '(not recorded)'}{state}")
    if rec.get("model_hash"):
        lines.append(f"hash      : {rec['model_hash']}")
    lines.append(f"type      : {rec.get('model_type') or '-'}"
                 + (f" | base: {rec['base_model']}"
                    if rec.get("base_model") else ""))
    out = rec.get("outcome") or "-"
    if rec.get("rating"):
        out += f"  rating {rec['rating']}/5"
    lines.append(f"outcome   : {out}")
    lines.append(f"source    : {rec.get('source')} | created "
                 f"{rec.get('created_at')}"
                 + (f" | trained {rec['trained_at']}"
                    if rec.get("trained_at") else ""))
    if rec.get("run_id"):
        row = conn.execute(
            "SELECT name, status, last_loss FROM runs WHERE run_id = ?",
            (rec["run_id"],)).fetchone()
        if row:
            lines.append(f"run       : #{rec['run_id']} {row[0]} "
                         f"[{row[1]}] final loss {row[2]}")
    params = rec.get("params") or {}
    if params:
        lines.append("params    :")
        for k in sorted(params):
            v = params[k]
            if isinstance(v, dict):
                v = json.dumps(v, ensure_ascii=False)
            lines.append(f"  {k} = {v}")

    # dataset lineage (Phase 2 registry)
    dsv = rec.get("dataset_version")
    lines.append("-" * 60)
    if dsv is not None:
        try:
            from . import dataset_versioning as dv
            info = dv.get_version(conn, int(dsv))
            if info:
                lines.append(f"dataset   : v{int(dsv):03d} "
                             f"'{info.get('name')}' [{info.get('kind')}] "
                             f"{info.get('image_count')} images")
                chain = dv.lineage_chain(conn, int(dsv))
                if len(chain) > 1:
                    lines.append("lineage   : " + " <- ".join(
                        f"v{c['version']:03d}" for c in chain))
            else:
                lines.append(f"dataset   : v{int(dsv):03d} "
                             "(not in the version registry)")
        except Exception:
            lines.append(f"dataset   : v{int(dsv):03d}")
    elif rec.get("dataset_dir"):
        lines.append(f"dataset   : {rec['dataset_dir']} (unversioned)")
    else:
        lines.append("dataset   : (not recorded)")

    # eval averages when the evals table knows this model by name/stem
    keys = {rec["name"]}
    if mp:
        keys.add(Path(mp).stem)
    marks = ",".join("?" for _ in keys)
    try:
        rows = conn.execute(
            f"SELECT category, COUNT(*), AVG(likeness) FROM evals "
            f"WHERE lora IN ({marks}) AND likeness IS NOT NULL "
            "GROUP BY category", tuple(keys)).fetchall()
        if rows:
            lines.append("evals     : " + "  ".join(
                f"{c or 'all'}: {avg:.3f} (n={n})" for c, n, avg in rows))
    except Exception:
        pass

    notes = (rec.get("notes") or "").strip()
    lines.append("-" * 60)
    lines.append("notes:")
    lines.append(notes if notes
                 else "  (none - add with: pixel-curator exp note "
                      f"{rec['exp_id']} \"...\")")
    return "\n".join(lines)


def unregistered_runs(conn: sqlite3.Connection,
                      dataset_version: Optional[int] = None) -> list[dict]:
    """Internal training runs with no experiment record (optionally for
    one dataset version) - surfaced in `exp list` as promotion hints."""
    sql = ("SELECT r.run_id, r.name, r.status, r.dataset_version FROM runs r "
           "LEFT JOIN experiments e ON e.run_id = r.run_id "
           "WHERE e.exp_id IS NULL")
    params: list = []
    if dataset_version is not None:
        sql += " AND r.dataset_version = ?"
        params.append(int(dataset_version))
    return [{"run_id": r[0], "name": r[1], "status": r[2],
             "dataset_version": r[3]}
            for r in conn.execute(sql, params)]


__all__ = [
    "register_experiment", "update_experiment", "append_note",
    "get_experiment", "list_experiments", "experiment_from_run",
    "quick_hash", "write_experiment_sidecar", "write_experiment_template",
    "load_experiment_file", "render_experiments",
    "render_experiment_detail", "unregistered_runs",
    "SIDECAR_SUFFIX", "TEMPLATE_NAME", "MODEL_TYPES",
]
