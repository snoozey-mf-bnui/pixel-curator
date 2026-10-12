"""Advanced Curation: duplicate & near-duplicate detection (Phase 1).

Two detection layers over frames already scored by curation:

* perceptual hashing (dHash, already stored per frame) - CPU-safe,
  banded LSH bucketing so large libraries stay O(n) in practice
* optional CLIP embedding similarity - reuses the embeddings the
  diversity-clustering step stored in `clip_embeds`; requires numpy and
  degrades to a clear professional warning when unavailable

Groups carry stable ids (dup_00042), a representative chosen by the
aesthetic proxy, per-frame confidence scores, method and reason codes.
Strictly non-destructive: results are written to additive `dup_*`
manifest columns and surfaced for review - curation status is NEVER
changed and no files are touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Optional

from ..insights import aesthetic_proxy
from .basic import hamming

DEFAULT_MAX_HAMMING = 6          # <= 6/64 bits -> near-duplicate
EXACT_HAMMING = 0
CLIP_SIM_THRESHOLD = 0.97        # cosine similarity for embedding groups
_BANDS = 4                       # 16-bit LSH bands over the 64-bit dhash


@dataclass
class DuplicateScanConfig:
    dataset_version: int = 0     # 0 -> all curated/scored frames
    method: str = "phash"        # phash | clip | both
    max_hamming: int = DEFAULT_MAX_HAMMING
    clip_threshold: float = CLIP_SIM_THRESHOLD
    statuses: tuple = ("selected", "packaged", "scored")
    write: bool = True           # False -> report only


@dataclass
class DuplicateGroup:
    group_id: str
    method: str
    frames: list[dict] = field(default_factory=list)   # {frame_id, score,..}

    def to_json(self) -> dict:
        return {"group_id": self.group_id, "method": self.method,
                "size": len(self.frames), "frames": self.frames}


# ---------------------------------------------------------------------------
# Frame scoping
# ---------------------------------------------------------------------------

def _scoped_frames(conn, cfg: DuplicateScanConfig) -> list[dict]:
    marks = ",".join("?" for _ in cfg.statuses)
    if cfg.dataset_version:
        rows = conn.execute(
            f"SELECT f.frame_id, f.path, f.dhash, f.sharpness, "
            f"f.brightness, d.framing FROM frames f "
            f"JOIN dataset_frames df ON df.frame_id = f.frame_id "
            f"AND df.version = ? "
            f"LEFT JOIN detections d ON d.frame_id = f.frame_id "
            f"WHERE f.dhash IS NOT NULL", (cfg.dataset_version,)).fetchall()
    else:
        rows = conn.execute(
            f"SELECT f.frame_id, f.path, f.dhash, f.sharpness, "
            f"f.brightness, d.framing FROM frames f "
            f"LEFT JOIN detections d ON d.frame_id = f.frame_id "
            f"WHERE f.status IN ({marks}) AND f.dhash IS NOT NULL",
            cfg.statuses).fetchall()
    return [{"frame_id": r[0], "path": r[1], "dhash": r[2],
             "sharpness": r[3], "brightness": r[4], "framing": r[5]}
            for r in rows]


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ---------------------------------------------------------------------------
# Perceptual-hash grouping (banded LSH + verified hamming)
# ---------------------------------------------------------------------------

def phash_groups(frames: list[dict],
                 max_hamming: int = DEFAULT_MAX_HAMMING
                 ) -> tuple[list[list[dict]], dict[str, dict]]:
    """Group frames whose dHash distance <= max_hamming.  Returns
    (groups, per-frame edge info {frame_id: {distance, exact}})."""
    items = []
    for f in frames:
        try:
            items.append((f, int(f["dhash"], 16)))
        except (TypeError, ValueError):
            continue

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, (_f, bits) in enumerate(items):
        for band in range(_BANDS):
            key = (band, (bits >> (band * 16)) & 0xFFFF)
            buckets.setdefault(key, []).append(idx)

    uf = _UnionFind()
    edge: dict[str, dict] = {}
    compared: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                pair = (a, b) if a < b else (b, a)
                if pair in compared:
                    continue
                compared.add(pair)
                fa, ba = items[a]
                fb, bb = items[b]
                dist = hamming(ba, bb)
                if dist <= max_hamming:
                    uf.union(fa["frame_id"], fb["frame_id"])
                    for fid in (fa["frame_id"], fb["frame_id"]):
                        best = edge.get(fid)
                        if best is None or dist < best["distance"]:
                            edge[fid] = {"distance": dist,
                                         "exact": dist == EXACT_HAMMING}

    clusters: dict[str, list[dict]] = {}
    for f, _bits in items:
        fid = f["frame_id"]
        if fid in uf.parent:
            clusters.setdefault(uf.find(fid), []).append(f)
    groups = [g for g in clusters.values() if len(g) > 1]
    groups.sort(key=lambda g: sorted(f["frame_id"] for f in g)[0])
    return groups, edge


# ---------------------------------------------------------------------------
# Optional CLIP-embedding grouping (numpy + stored embeddings required)
# ---------------------------------------------------------------------------

def clip_available(conn) -> tuple[bool, str]:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False, ("numpy not installed - CLIP embedding similarity "
                       "disabled (perceptual hashing still active)")
    n = conn.execute("SELECT COUNT(*) FROM clip_embeds").fetchone()[0]
    if n == 0:
        return False, ("no stored CLIP embeddings - run the diversity "
                       "clustering step ([cluster] extras) first")
    return True, f"{n} stored embeddings"


def clip_groups(conn, frames: list[dict],
                threshold: float = CLIP_SIM_THRESHOLD
                ) -> tuple[list[list[dict]], dict[str, dict]]:
    """Cosine-similarity grouping over stored CLIP embeddings.  Pure
    numpy; chunked matmul keeps memory bounded.  Frames without an
    embedding are skipped silently."""
    import numpy as np
    by_id = {f["frame_id"]: f for f in frames}
    ids, vecs = [], []
    marks = ",".join("?" for _ in by_id)
    for fid, blob in conn.execute(
            f"SELECT frame_id, embed FROM clip_embeds "
            f"WHERE frame_id IN ({marks})", list(by_id)):
        v = np.frombuffer(blob, dtype=np.float32)
        if v.size:
            ids.append(fid)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
    if len(ids) < 2:
        return [], {}
    mat = np.stack(vecs)
    uf = _UnionFind()
    edge: dict[str, dict] = {}
    chunk = 512
    for start in range(0, len(ids), chunk):
        sims = mat[start:start + chunk] @ mat.T
        for li, row in enumerate(sims):
            i = start + li
            for j in np.nonzero(row >= threshold)[0]:
                if j <= i:
                    continue
                sim = float(row[j])
                uf.union(ids[i], ids[int(j)])
                for fid in (ids[i], ids[int(j)]):
                    best = edge.get(fid)
                    if best is None or sim > best["similarity"]:
                        edge[fid] = {"similarity": sim}
    clusters: dict[str, list[dict]] = {}
    for fid in ids:
        if fid in uf.parent:
            clusters.setdefault(uf.find(fid), []).append(by_id[fid])
    groups = [g for g in clusters.values() if len(g) > 1]
    groups.sort(key=lambda g: sorted(f["frame_id"] for f in g)[0])
    return groups, edge


# ---------------------------------------------------------------------------
# Scan: group -> representative -> manifest (non-destructive)
# ---------------------------------------------------------------------------

def _annotate(groups: list[list[dict]], edge: dict, method: str,
              start_index: int = 1) -> list[DuplicateGroup]:
    out: list[DuplicateGroup] = []
    for n, g in enumerate(groups, start=start_index):
        gid = f"dup_{n:05d}"
        rep = max(g, key=lambda f: (aesthetic_proxy(f), f["frame_id"]))
        rows = []
        for f in g:
            info = edge.get(f["frame_id"], {})
            if method == "phash":
                dist = info.get("distance", DEFAULT_MAX_HAMMING)
                score = round(1.0 - dist / 64.0, 3)
                reasons = (["phash_identical"] if info.get("exact")
                           else ["phash_distance_low"])
            else:
                score = round(info.get("similarity", CLIP_SIM_THRESHOLD), 3)
                reasons = ["embedding_similarity_high"]
            rows.append({
                "frame_id": f["frame_id"], "path": f["path"],
                "duplicate_group_id": gid,
                "duplicate_score": score,
                "duplicate_method": method,
                "duplicate_role": ("representative"
                                   if f["frame_id"] == rep["frame_id"]
                                   else "candidate"),
                "duplicate_reason_codes": reasons,
                "aesthetic_score": aesthetic_proxy(f),
            })
        rows.sort(key=lambda r: (r["duplicate_role"] != "representative",
                                 r["frame_id"]))
        out.append(DuplicateGroup(group_id=gid, method=method, frames=rows))
    return out


def scan_duplicates(conn, cfg: DuplicateScanConfig) -> dict:
    """Run the configured duplicate scan; write additive dup_* fields
    (when cfg.write) and return a JSON-safe summary.  Never changes
    frame status, never deletes anything."""
    frames = _scoped_frames(conn, cfg)
    warnings: list[str] = []
    groups: list[DuplicateGroup] = []

    if cfg.method in ("phash", "both"):
        raw, edge = phash_groups(frames, cfg.max_hamming)
        groups += _annotate(raw, edge, "phash")

    if cfg.method in ("clip", "both"):
        ok, reason = clip_available(conn)
        if not ok:
            warnings.append(reason)
        else:
            grouped = {f["frame_id"] for g in groups for f in g.frames}
            pool = [f for f in frames if f["frame_id"] not in grouped]
            raw, edge = clip_groups(conn, pool, cfg.clip_threshold)
            groups += _annotate(raw, edge, "clip_embedding",
                                start_index=len(groups) + 1)

    if cfg.write:
        # clear previous scan results for the scoped frames, then write
        ids = [(f["frame_id"],) for f in frames]
        conn.executemany(
            "UPDATE frames SET dup_group_id=NULL, dup_score=NULL, "
            "dup_method=NULL, dup_role=NULL, dup_reasons=NULL "
            "WHERE frame_id=?", ids)
        rows = [(r["duplicate_group_id"], r["duplicate_score"],
                 r["duplicate_method"], r["duplicate_role"],
                 ",".join(r["duplicate_reason_codes"]), r["frame_id"])
                for g in groups for r in g.frames]
        conn.executemany(
            "UPDATE frames SET dup_group_id=?, dup_score=?, dup_method=?, "
            "dup_role=?, dup_reasons=? WHERE frame_id=?", rows)
        conn.commit()

    candidates = sum(1 for g in groups for r in g.frames
                     if r["duplicate_role"] == "candidate")
    return {
        "frames_scanned": len(frames),
        "method": cfg.method,
        "max_hamming": cfg.max_hamming,
        "groups": [g.to_json() for g in groups],
        "group_count": len(groups),
        "exact_groups": sum(
            1 for g in groups
            if any("phash_identical" in r["duplicate_reason_codes"]
                   for r in g.frames)),
        "review_candidates": candidates,
        "warnings": warnings,
        "non_destructive": True,
    }


def scan_generator(conn, cfg: DuplicateScanConfig
                   ) -> Generator[str, None, None]:
    """CLI-friendly streaming wrapper."""
    yield (f"DUPLICATE SCAN | method {cfg.method} | "
           f"{'dataset v%03d' % cfg.dataset_version if cfg.dataset_version else 'all curated frames'} "
           f"| max hamming {cfg.max_hamming}")
    summary = scan_duplicates(conn, cfg)
    for w in summary["warnings"]:
        yield f"  NOTE: {w}"
    yield (f"  frames scanned: {summary['frames_scanned']}\n"
           f"  duplicate groups: {summary['group_count']} "
           f"(exact/near: {summary['exact_groups']})\n"
           f"  review candidates: {summary['review_candidates']}")
    for g in summary["groups"][:20]:
        rep = next(r for r in g["frames"]
                   if r["duplicate_role"] == "representative")
        yield (f"  {g['group_id']} [{g['method']}] x{g['size']} "
               f"rep={rep['frame_id']} "
               f"(aesthetic {rep['aesthetic_score']:.2f})")
    if summary["group_count"] > 20:
        yield f"  ... {summary['group_count'] - 20} more groups (use --json)"
    yield ("  Non-destructive: groups are flagged for review only - no "
           "status changes, no file operations.")
