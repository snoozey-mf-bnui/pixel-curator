"""Fast identity re-scoring against a rebuilt anchor.

When the identity anchor changes (e.g. after enriching ``anchor_refs/``), every
already-scanned frame's identity score is stale. Re-running ``curate --smart``
would re-detect faces on the whole archive (hours). But smart curation already
stored each face's ArcFace embedding in ``detections.embed`` - so we can simply
re-dot those embeddings against the new anchor and re-apply the identity
threshold. No face detection, no image reads: minutes, not hours.

Only frames that already have a face embedding and sit in ``selected`` or
``rejected_identity`` are touched. Frames cut for no-face / small-face are left
alone (the anchor change does not affect them). Non-destructive to source media.

CLI:  pixel-curator rescore [--identity-threshold 0.35]
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from . import manifest
from .config import Project, SmartCurateConfig


def _decide(sim: float, threshold: float, subject_ok: bool, relax: float) -> str:
    """Selected vs rejected_identity for one face, with sidecar rescue."""
    effective = threshold - (relax if subject_ok else 0.0)
    return "selected" if sim >= effective else "rejected_identity"


def rescore_identity_generator(
    prj: Project,
    output_base: Path,
    identity_threshold: float = 0.35,
    metadata_dir: Path | None = None,
    subject_names: tuple = (),
    sidecar_relax: float = 0.10,
) -> Generator[str, None, None]:
    """Re-apply the manifest anchor to stored embeddings; re-threshold. Fast."""
    output_base = Path(output_base).expanduser()
    conn = manifest.connect(output_base)

    anchor_list = manifest.meta_get(conn, "identity_anchor")
    if not anchor_list:
        yield "FATAL: no identity anchor in the manifest. Run `anchor` first."
        return
    try:
        import numpy as np
    except Exception:
        yield "FATAL: numpy unavailable (install pixel-curator[ai])."
        return
    anchor = np.asarray(anchor_list, dtype=np.float32)
    info = manifest.meta_get(conn, "identity_anchor_info") or {}
    yield (f"Re-scoring against anchor of {info.get('count', '?')} refs "
           f"(dim {anchor.shape[0]}), threshold {identity_threshold}.")

    sidecar_idx = None
    if metadata_dir and subject_names:
        from .sidecar import build_sidecar_index
        sidecar_idx = build_sidecar_index(metadata_dir)
        for w in sidecar_idx.warnings:
            yield f"Sidecar warning: {w}"
        yield (f"Sidecar rescue on: {sidecar_idx.parsed} tags | "
               f"subjects {list(subject_names)} | relax -{sidecar_relax}")
    from .sidecar import subject_present

    rows = conn.execute(
        "SELECT d.frame_id, d.embed, f.status, s.path "
        "FROM detections d JOIN frames f ON f.frame_id = d.frame_id "
        "LEFT JOIN sources s ON s.source_id = f.source_id "
        "WHERE d.embed IS NOT NULL "
        "AND f.status IN ('selected','rejected_identity')"
    ).fetchall()

    if not rows:
        yield "Nothing to rescore (no stored face embeddings). Run curate --smart first."
        return

    rescored = newly_sel = newly_rej = rescued = 0
    for frame_id, embed, status, src_path in rows:
        emb = np.frombuffer(embed, dtype=np.float32)
        if emb.shape[0] != anchor.shape[0]:
            continue
        sim = float(np.dot(anchor, emb))
        subj_ok = bool(sidecar_idx) and bool(src_path) and subject_present(
            sidecar_idx.lookup(src_path), subject_names)
        new_status = _decide(sim, identity_threshold, subj_ok, sidecar_relax)
        conn.execute("UPDATE detections SET identity_sim=? WHERE frame_id=?",
                     (sim, frame_id))
        if new_status != status:
            conn.execute("UPDATE frames SET status=? WHERE frame_id=?",
                         (new_status, frame_id))
            if new_status == "selected":
                newly_sel += 1
                if subj_ok and sim < identity_threshold:
                    rescued += 1
            else:
                newly_rej += 1
        rescored += 1
        if rescored % 5000 == 0:
            conn.commit()
            yield f"  rescored {rescored} | +{newly_sel} selected | -{newly_rej} rejected"
    conn.commit()

    yield (f"\nRESCORE DONE.\n"
           f"Faces rescored: {rescored}\n"
           f"Newly selected: {newly_sel}"
           + (f" (of which {rescued} via subject face tag)" if rescued else "") + "\n"
           f"Newly rejected: {newly_rej}\n"
           "Next: pixel-curator plan  (then rebuild datasets)")


def config_from(args, prj: Project) -> SmartCurateConfig:
    """Build a SmartCurateConfig view for sidecar values (CLI override → project)."""
    meta_dir = getattr(args, "metadata_dir", None) or prj.metadata_dir or None
    subjects = tuple(getattr(args, "subject_names", None) or prj.subject_names or ())
    return SmartCurateConfig(
        output_base=prj.output_path,
        metadata_dir=Path(meta_dir).expanduser() if meta_dir else None,
        subject_names=subjects,
        sidecar_identity_relax=getattr(args, "sidecar_relax", 0.10),
        identity_threshold=getattr(args, "identity_threshold", 0.35),
    )
