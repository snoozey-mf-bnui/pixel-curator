"""Subject-centered smart-crop to SDXL bucket resolutions (Phase 4 remainder).

Instead of blind center-crop, the crop window is centered on the subject's
face bbox (recorded by smart curation, schema v5). Frames without a bbox
fall back to center crop. Pattern reference: reForge autocrop.py focal-point
weighting, simplified to face-POI + clamp (our detector is far stronger than
its YuNet/Haar, so entropy/edge blending adds little).

Pure math helpers are stdlib-only and unit-tested; the actual image write
uses Pillow (already required by every AI extra).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from .util import HAVE_PIL

if HAVE_PIL:
    from PIL import Image

# Standard SDXL 1024-base buckets (w, h), 64-px aligned.
SDXL_BUCKETS: list[tuple[int, int]] = [
    (1024, 1024),
    (1152, 896), (896, 1152),
    (1216, 832), (832, 1216),
    (1344, 768), (768, 1344),
    (1536, 640), (640, 1536),
]


def scaled_buckets(base: int = 1024) -> list[tuple[int, int]]:
    """Buckets scaled for other bases (e.g. 768 for lighter VRAM)."""
    if base == 1024:
        return SDXL_BUCKETS
    f = base / 1024.0
    return [(max(64, int(w * f) // 64 * 64), max(64, int(h * f) // 64 * 64))
            for w, h in SDXL_BUCKETS]


def choose_bucket(w: int, h: int, base: int = 1024) -> tuple[int, int]:
    """Nearest bucket by log-aspect-ratio distance."""
    ar = math.log(w / h)
    return min(scaled_buckets(base), key=lambda b: abs(math.log(b[0] / b[1]) - ar))


def crop_box_for_poi(
    src_w: int, src_h: int, bucket_w: int, bucket_h: int,
    poi_x: float = 0.5, poi_y: float = 0.5,
) -> tuple[int, int, int, int, int, int]:
    """Returns (resize_w, resize_h, left, top, right, bottom).
    Scale-to-cover, then a bucket-sized window centered on the POI
    (normalized 0-1 in source coords), clamped inside the image."""
    scale = max(bucket_w / src_w, bucket_h / src_h)
    rw, rh = max(bucket_w, round(src_w * scale)), max(bucket_h, round(src_h * scale))
    cx, cy = poi_x * rw, poi_y * rh
    left = int(round(cx - bucket_w / 2))
    top = int(round(cy - bucket_h / 2))
    left = max(0, min(left, rw - bucket_w))
    top = max(0, min(top, rh - bucket_h))
    return rw, rh, left, top, left + bucket_w, top + bucket_h


def poi_from_bbox(bbox_json: Optional[str]) -> tuple[float, float]:
    """Face bbox -> POI. Biased slightly above bbox center (face -> head/torso
    composition). Missing/invalid bbox -> image center."""
    if not bbox_json:
        return 0.5, 0.5
    try:
        x0, y0, x1, y1 = json.loads(bbox_json)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        # keep a touch of headroom: nudge crop center slightly below the face
        cy = min(1.0, cy + (y1 - y0) * 0.35)
        return max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
    except Exception:
        return 0.5, 0.5


def smart_crop_image(
    src: Path, dest: Path, bbox_json: Optional[str],
    base: int = 1024, quality: int = 92,
) -> tuple[int, int]:
    """Crop src to its nearest SDXL bucket, subject-centered. Returns bucket."""
    if not HAVE_PIL:
        raise RuntimeError("smart_crop requires Pillow (pip install Pillow)")
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        bw, bh = choose_bucket(w, h, base)
        px, py = poi_from_bbox(bbox_json)
        rw, rh, l, t, r, b = crop_box_for_poi(w, h, bw, bh, px, py)
        im = im.resize((rw, rh), Image.LANCZOS).crop((l, t, r, b))
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, "JPEG", quality=quality)
    return bw, bh
