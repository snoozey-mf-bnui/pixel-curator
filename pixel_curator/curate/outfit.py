"""Outfit / accessory separation layer (Priority 3).

Tags each frame with a compact *outfit signature* - the dominant clothing
colour(s) of the body region plus the garment tags WD14 already found. This
lets you later cluster the dataset by outfit to build clean "outfit LoRAs" or
redressing workflows without identity leakage, and to see when one outfit is
over-represented (the same bug that bakes a birthday dress into the base).

Colour is read from the BODY region (below the face bbox), not the face, so skin
tone never contaminates the outfit signal. Pure numpy; the named-colour mapping
and signature are fully unit-tested. Garment extraction reuses WD14 tags rather
than running another model.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Optional

import numpy as np

# coarse named palette (RGB anchors) - clothing-relevant colours
_PALETTE = {
    "black": (20, 20, 20), "white": (235, 235, 235), "gray": (128, 128, 128),
    "red": (200, 35, 35), "pink": (235, 150, 175), "orange": (230, 140, 40),
    "yellow": (225, 210, 70), "green": (60, 160, 75), "blue": (55, 95, 200),
    "navy": (30, 40, 95), "purple": (125, 60, 170), "brown": (120, 80, 50),
    "beige": (220, 200, 170),
}
_NAMES = list(_PALETTE)
_ANCHORS = np.array([_PALETTE[n] for n in _NAMES], dtype=np.float32)

# WD14 tags that denote garments/accessories (normalized, underscores)
_GARMENTS = {
    "dress", "shirt", "t-shirt", "blouse", "skirt", "jacket", "coat", "sweater",
    "jeans", "pants", "shorts", "bikini", "swimsuit", "one-piece_swimsuit",
    "suit", "hoodie", "cardigan", "tank_top", "crop_top", "leggings", "gown",
    "romper", "jumpsuit", "top", "vest", "blazer", "kimono", "robe", "overalls",
    "bra", "underwear", "lingerie", "panties", "stockings", "pantyhose",
    "scarf", "hat", "gloves", "necklace", "earrings", "glasses", "sunglasses",
    "boots", "heels", "sneakers",
}


def nearest_color(rgb) -> str:
    d = _ANCHORS - np.asarray(rgb, dtype=np.float32)[None, :]
    return _NAMES[int(np.argmin((d * d).sum(axis=1)))]


def body_region(bbox_px, w, h):
    """Torso/body box below the face. Falls back to lower-centre when no face."""
    if bbox_px is None:
        return (int(w * 0.2), int(h * 0.4), int(w * 0.8), h)
    fx0, fy0, fx1, fy1 = bbox_px
    fw = max(1.0, fx1 - fx0)
    fh = max(1.0, fy1 - fy0)
    cx = (fx0 + fx1) / 2.0
    x0 = max(0, int(cx - 1.5 * fw))
    x1 = min(w, int(cx + 1.5 * fw))
    y0 = min(h - 1, int(fy1))
    y1 = min(h, int(fy1 + 3.0 * fh))
    if y1 <= y0:
        y1 = min(h, y0 + int(fh) + 1)
    return (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))


def dominant_colors(rgb_img, region=None, k: int = 2) -> list[str]:
    """Top-k named clothing colours in the region by pixel share."""
    img = np.asarray(rgb_img)
    if region is not None:
        x0, y0, x1, y1 = region
        img = img[y0:y1, x0:x1]
    if img.size == 0:
        return []
    flat = img.reshape(-1, img.shape[-1])[:, :3].astype(np.float32)
    if flat.shape[0] > 4000:                          # subsample for speed
        idx = np.linspace(0, flat.shape[0] - 1, 4000).astype(int)
        flat = flat[idx]
    # vectorized nearest-anchor
    d = ((flat[:, None, :] - _ANCHORS[None, :, :]) ** 2).sum(axis=2)
    names = [_NAMES[i] for i in d.argmin(axis=1)]
    return [name for name, _ in Counter(names).most_common(k)]


def extract_garments(tags) -> list[str]:
    """Garment/accessory tags from a WD14 tag list. Matches the whole tag *or* a
    garment keyword appearing as a token within it (so 'red dress' -> 'dress',
    'high heels' -> 'heels')."""
    out = []
    for t in tags or []:
        n = str(t).strip().lower().replace(" ", "_")
        matched = None
        if n in _GARMENTS:
            matched = n
        else:
            toks = set(n.split("_"))
            for g in _GARMENTS:
                if "_" not in g and g in toks:
                    matched = g
                    break
        if matched and matched not in out:
            out.append(matched)
    return out


def outfit_signature(colors: list[str], garments: list[str]) -> str:
    """Compact key like 'black_dress' / 'blue_top' / 'red' / 'unknown'."""
    c = colors[0] if colors else ""
    g = garments[0] if garments else ""
    if c and g:
        return f"{c}_{g}"
    return c or g or "unknown"


# ---------------------------------------------------------------------------
# pass
# ---------------------------------------------------------------------------

def _tags_from_json(tags_json) -> list[str]:
    try:
        data = json.loads(tags_json) if tags_json else []
        if data and isinstance(data[0], (list, tuple)):
            return [str(x[0]) for x in data]
        return [str(x) for x in data]
    except Exception:
        return []


def _parse_bbox(raw, w, h):
    if raw is None:
        return None
    try:
        b = json.loads(raw) if isinstance(raw, str) else list(raw)
        if len(b) == 4:
            return [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
    except Exception:
        pass
    return None


def _default_load(path):
    # Canonical orientation path (orient.exif_normalized_rgb) - shared with the
    # rest of curation so outfit analysis sees upright pixels.
    try:
        from ..orient import exif_normalized_rgb
        rgb = exif_normalized_rgb(path)
        return np.asarray(rgb) if rgb is not None else None
    except Exception:
        return None


def run_outfit_pass(conn, *, load_image=None, statuses=("selected", "packaged"),
                    limit: int = 0, store: bool = True) -> dict:
    """Compute an outfit signature per frame and store it. Returns a summary +
    the outfit distribution (which outfits dominate)."""
    from ..util import now_iso
    load_image = load_image or _default_load
    qs = ",".join("?" for _ in statuses)
    rows = conn.execute(
        f"SELECT f.frame_id, f.path, d.bbox, c.tags_json "
        f"FROM frames f LEFT JOIN detections d ON d.frame_id = f.frame_id "
        f"LEFT JOIN captions c ON c.frame_id = f.frame_id "
        f"WHERE f.status IN ({qs})", statuses).fetchall()
    if limit:
        rows = rows[:limit]
    now = now_iso()
    summary = {"scanned": 0, "tagged": 0, "unreadable": 0}
    signatures = Counter()
    for fid, path, bbox_raw, tags_json in rows:
        summary["scanned"] += 1
        img = load_image(path)
        if img is None or np.asarray(img).size == 0:
            summary["unreadable"] += 1
            continue
        img = np.asarray(img)
        h, w = img.shape[:2]
        bbox_px = _parse_bbox(bbox_raw, w, h)
        colors = dominant_colors(img, body_region(bbox_px, w, h))
        garments = extract_garments(_tags_from_json(tags_json))
        sig = outfit_signature(colors, garments)
        signatures[sig] += 1
        if store:
            conn.execute(
                "INSERT OR REPLACE INTO frame_outfit(frame_id, dominant_colors, "
                "garments, signature, calculated_at) VALUES (?,?,?,?,?)",
                (fid, ",".join(colors), ",".join(garments), sig, now))
        summary["tagged"] += 1
    if store:
        conn.commit()
    top = signatures.most_common(10)
    dominant_frac = (top[0][1] / summary["tagged"]) if summary["tagged"] and top else 0.0
    return {"summary": summary, "distinct_outfits": len(signatures),
            "top_outfits": top, "dominant_frac": round(dominant_frac, 3)}


def render_outfit(res: dict) -> str:
    s = res["summary"]
    lines = ["", "Outfit / accessory separation", "=" * 48,
             f"scanned : {s['scanned']}   tagged : {s['tagged']}   "
             f"unreadable : {s['unreadable']}",
             f"distinct outfits : {res['distinct_outfits']}"]
    for sig, n in res["top_outfits"]:
        lines.append(f"   {n:>5}  {sig}")
    if res["dominant_frac"] > 0.4:
        lines.append(f"\n!  one outfit is {res['dominant_frac']:.0%} of the set - "
                     "it may bake into the base. Diversify or cap it for a "
                     "flexible model (see diverse-select).")
    return "\n".join(lines)
