"""Behavioral tests for the curation scorers using synthetic images.

No models, no network: images are generated with PIL so scorer *ordering
invariants* are verified (sharp beats blurry, bright beats dark, identical
images hash identically, near-copies stay near).
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageFilter

from pixel_curator.curate.basic import dhash_bits, hamming, laplacian_variance, score_frame
from pixel_curator.curate.duplicates import phash_groups

pytest.importorskip("PIL")


# --------------------------------------------------------------------------
# fixtures: deterministic synthetic images
# --------------------------------------------------------------------------

def _checkerboard(size: int = 96, cell: int = 8) -> Image.Image:
    im = Image.new("L", (size, size))
    px = im.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = 255 if ((x // cell) + (y // cell)) % 2 == 0 else 0
    return im


def _flat(size: int = 96, value: int = 128) -> Image.Image:
    return Image.new("L", (size, size), value)


def _gradient(size: int = 96) -> Image.Image:
    im = Image.new("L", (size, size))
    px = im.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = int(255 * x / (size - 1))
    return im


def _save(im: Image.Image, path) -> str:
    im.save(path)
    return str(path)


# --------------------------------------------------------------------------
# laplacian sharpness
# --------------------------------------------------------------------------

def test_sharp_beats_blurry(tmp_path):
    crisp = _save(_checkerboard(), tmp_path / "crisp.png")
    blurry = _save(_checkerboard().filter(ImageFilter.GaussianBlur(3)),
                   tmp_path / "blurry.png")
    r1 = score_frame(crisp)
    r2 = score_frame(blurry)
    assert r1 is not None and r2 is not None
    assert r1[1] > r2[1], f"crisp {r1[1]} should beat blurry {r2[1]}"


def test_flat_image_has_zero_sharpness(tmp_path):
    r = score_frame(_save(_flat(), tmp_path / "flat.png"))
    assert r is not None
    assert r[1] == pytest.approx(0.0)


def test_laplacian_variance_matches_manual():
    gray = [0, 0, 0,
            0, 255, 0,
            0, 0, 0]
    v = laplacian_variance(gray, 3, 3)
    # single interior point: lap = -4*255 = -1020, variance of one value = 0
    assert v == pytest.approx(0.0)


# --------------------------------------------------------------------------
# brightness
# --------------------------------------------------------------------------

def test_brightness_ordering(tmp_path):
    bright = score_frame(_save(_flat(value=240), tmp_path / "b.png"))
    dark = score_frame(_save(_flat(value=20), tmp_path / "d.png"))
    assert bright[2] > 200 and dark[2] < 60
    assert bright[2] > dark[2]


# --------------------------------------------------------------------------
# dhash / hamming
# --------------------------------------------------------------------------

def test_identical_images_hash_identically(tmp_path):
    a = score_frame(_save(_gradient(), tmp_path / "a.png"))
    b = score_frame(_save(_gradient(), tmp_path / "b.png"))
    assert a[0] == b[0]
    assert hamming(int(a[0], 16), int(b[0], 16)) == 0


def test_different_images_hash_differently(tmp_path):
    a = score_frame(_save(_gradient(), tmp_path / "a.png"))
    b = score_frame(_save(_checkerboard(cell=4), tmp_path / "b.png"))
    dist = hamming(int(a[0], 16), int(b[0], 16))
    assert dist > 8, f"unrelated images should differ (got {dist})"


def test_dhash_bits_deterministic():
    gray9x8 = [(x * 7 + y * 3) % 256 for y in range(8) for x in range(9)]
    assert dhash_bits(gray9x8) == dhash_bits(list(gray9x8))


# --------------------------------------------------------------------------
# duplicate grouping
# --------------------------------------------------------------------------

def _frame(fid: int, dhash: str, status: str = "selected") -> dict:
    return {"frame_id": fid, "dhash": dhash, "status": status,
            "sharpness": 100.0 - fid, "path": f"/f/{fid}.jpg"}


def test_phash_groups_clusters_near_duplicates():
    base = 0b10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010
    near = base ^ 0b11                     # 2-bit difference
    far = base ^ 0xFFFF_FFFF_0000_0000     # 32-bit difference
    frames = [_frame(1, f"{base:016x}"), _frame(2, f"{near:016x}"),
              _frame(3, f"{far:016x}")]
    groups, _edges = phash_groups(frames, max_hamming=8)
    assert len(groups) == 1
    member_ids = sorted(f["frame_id"] for f in groups[0])
    assert member_ids == [1, 2]
