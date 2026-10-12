"""Dual captioning - optional vision-language (Florence-2) rich captions.

WD14 (caption/wd14.py) gives precise *structured tags* (garments, pose, setting
as booru tokens). This adds an optional *natural-language scene description* from
a VLM (Florence-2 by default), which captures relationships and context tags
miss - "leaning against a brick wall in afternoon light" vs just `wall, outdoors`.
Richer captions = more controllable variations at generation time (the lesson:
caption what you want to be able to change).

Modular by design:
  * compose_dual()         pure merge of WD14 caption + VLM sentence (testable)
  * run_vlm_pass()         stores vlm_text per frame; optional compose into the
                           trained caption_text
  * make_florence_captioner()  lazy/optional Florence-2 backend (heavy; best on
                           the A100). Returns (fn, reason); never hard-fails.

The backend is injectable, so the composition logic is fully unit-tested without
torch/transformers installed.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

# boilerplate VLMs love to prepend; stripped so captions stay clean
_VLM_PREFIXES = (
    "the image shows", "this image shows", "the image is", "this is",
    "a photo of", "an image of", "the photo shows", "image of", "photo of",
    "this picture shows", "the picture shows",
)


def clean_vlm_text(text: str) -> str:
    """Lowercase, strip trailing punctuation and the common VLM lead-ins."""
    s = " ".join(str(text or "").strip().split())
    low = s.lower()
    for p in _VLM_PREFIXES:
        if low.startswith(p):
            s = s[len(p):].lstrip(" ,:")
            low = s.lower()
    return s.rstrip(". ").strip()


def compose_dual(wd14_caption: str, vlm_text: str, *, max_words: int = 45) -> str:
    """Merge the WD14 caption with the VLM scene description. The WD14 caption
    (with its score prefix + trigger) leads; the cleaned VLM sentence is appended
    as descriptive context, truncated. Tokens already present aren't re-added."""
    base = " ".join(str(wd14_caption or "").strip().split())
    extra = clean_vlm_text(vlm_text)
    if not extra:
        return base
    if not base:
        return " ".join(extra.split()[:max_words])
    base_tokens = set(re.findall(r"[a-z0-9]+", base.lower()))
    # keep only extra words that add something (skip if >70% already present)
    ewords = extra.split()
    novel = [w for w in ewords if re.sub(r"[^a-z0-9]", "", w.lower()) not in base_tokens]
    if ewords and len(novel) / len(ewords) < 0.30:
        return base                                  # VLM adds nothing new
    return base + ", " + " ".join(ewords[:max_words])


# ---------------------------------------------------------------------------
# Florence-2 backend (lazy, optional)
# ---------------------------------------------------------------------------

def make_florence_captioner(prj=None, model_id: str = "microsoft/Florence-2-base",
                            task: str = "<DETAILED_CAPTION>"):
    """Return (caption_fn, reason). caption_fn(pil_rgb)->str when transformers +
    torch are available, else (None, reason). Heavy - intended for the A100."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except Exception as exc:
        return None, (f"VLM backend unavailable ({exc}); install torch + "
                      "transformers (best run on the A100)")
    state = {"model": None, "proc": None}

    def caption(pil_rgb) -> Optional[str]:
        try:
            if state["model"] is None:
                dev = "cuda" if torch.cuda.is_available() else (
                    "mps" if getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available() else "cpu")
                state["proc"] = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True)
                state["model"] = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True,
                    torch_dtype=torch.float16 if dev != "cpu" else torch.float32
                ).to(dev).eval()
                state["dev"] = dev
            proc, model, dev = state["proc"], state["model"], state["dev"]
            from PIL import Image
            img = pil_rgb if isinstance(pil_rgb, Image.Image) else Image.fromarray(
                pil_rgb)
            inputs = proc(text=task, images=img.convert("RGB"), return_tensors="pt")
            inputs = {k: (v.to(dev).half() if v.dtype == torch.float32 and dev != "cpu"
                          else v.to(dev)) for k, v in inputs.items()}
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=128, num_beams=3)
            text = proc.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = proc.post_process_generation(
                text, task=task, image_size=(img.width, img.height))
            return clean_vlm_text(parsed.get(task, "") if isinstance(parsed, dict) else str(parsed))
        except Exception:
            return None

    return caption, model_id


# ---------------------------------------------------------------------------
# pass
# ---------------------------------------------------------------------------

def _default_load(path):
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        return None


def run_vlm_pass(conn, caption_fn: Callable, *, model_id: str = "vlm",
                 load_image=None, statuses=("selected", "packaged"),
                 limit: int = 0, compose: bool = False, store: bool = True) -> dict:
    """Run the VLM over captioned frames, store vlm_text (and optionally compose
    it into caption_text). Skips frames already VLM-captioned (resumable)."""
    from ..util import now_iso
    load_image = load_image or _default_load
    qs = ",".join("?" for _ in statuses)
    rows = conn.execute(
        f"SELECT f.frame_id, f.path, c.caption_text, c.vlm_text "
        f"FROM frames f JOIN captions c ON c.frame_id = f.frame_id "
        f"WHERE f.status IN ({qs})", statuses).fetchall()
    if limit:
        rows = rows[:limit]
    now = now_iso()
    summary = {"scanned": 0, "captioned": 0, "composed": 0, "skipped": 0,
               "unreadable": 0}
    for fid, path, wd14, existing_vlm in rows:
        summary["scanned"] += 1
        if existing_vlm:
            summary["skipped"] += 1
            continue
        img = load_image(path)
        if img is None:
            summary["unreadable"] += 1
            continue
        vlm_text = caption_fn(img)
        if not vlm_text:
            summary["unreadable"] += 1
            continue
        vlm_text = clean_vlm_text(vlm_text)
        if store:
            conn.execute(
                "UPDATE captions SET vlm_text=?, vlm_model=?, updated_at=? "
                "WHERE frame_id=?", (vlm_text, model_id, now, fid))
            if compose:
                merged = compose_dual(wd14 or "", vlm_text)
                conn.execute(
                    "UPDATE captions SET caption_text=? WHERE frame_id=?",
                    (merged, fid))
                summary["composed"] += 1
        summary["captioned"] += 1
    if store:
        conn.commit()
    return {"summary": summary, "model": model_id}


def render_vlm(res: dict) -> str:
    s = res["summary"]
    return ("\nDual captioning (VLM)\n" + "=" * 48 +
            f"\nmodel       : {res['model']}"
            f"\nscanned     : {s['scanned']}"
            f"\ncaptioned   : {s['captioned']}   composed into caption_text: {s['composed']}"
            f"\nskipped     : {s['skipped']} (already had vlm_text)"
            f"\nunreadable  : {s['unreadable']}")
