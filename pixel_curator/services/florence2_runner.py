"""Florence-2 external runner - executes inside the dedicated VLM environment.

Invoked as a subprocess by ``florence_captioning`` (never imported by the main
service). Heavy dependencies (torch / transformers / PIL) are imported ONLY
inside functions, so ``import pixel_curator.services.florence2_runner`` is cheap
and model-free - tests can import it without any ML stack present.

Usage (payload/result files):
    <vlm-python> -m pixel_curator.services.florence2_runner \
        --payload-json /tmp/payload.json --result-json /tmp/result.json

Manual one-shot (outside tests):
    <vlm-python> -m pixel_curator.services.florence2_runner \
        --image /path/img.jpg --model /path/Florence-2-base-ft \
        --device mps --task "<MORE_DETAILED_CAPTION>" --json

MPS rules honored: no device_map="auto", no FlashAttention, float32 default,
manual ``.to(device)``, optional CPU fallback. All failures are returned as
structured JSON, never raw tracebacks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

VALID_TASKS = ("<MORE_DETAILED_CAPTION>", "<DETAILED_CAPTION>", "<CAPTION>")


def _select_device(requested: str):
    import torch
    if requested in ("mps", "cuda", "cpu"):
        if requested == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested
    # auto
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _dtype(name: str, device: str):
    import torch
    if name == "auto":
        return torch.float32 if device in ("mps", "cpu") else torch.float16
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}.get(name, torch.float32)


def _load_model(model_path, device, dtype, local_files_only, trust_remote_code):
    """Load Florence-2 from a local path. MPS-safe (manual .to, no device_map)."""
    from transformers import AutoModelForCausalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=trust_remote_code,
        local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=trust_remote_code,
        local_files_only=local_files_only, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return processor, model


def _generate(processor, model, image, task, extra, device, dtype,
              max_new_tokens, num_beams):
    import torch
    prompt = task + (extra or "")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}
    if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "to"):
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            do_sample=False)
    text = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = {}
    try:
        parsed = processor.post_process_generation(
            text, task=task, image_size=(image.width, image.height))
    except Exception:
        parsed = {}
    clean = parsed.get(task) if isinstance(parsed, dict) else None
    if isinstance(clean, str):
        out_text = clean
    else:
        out_text = (text.replace("</s>", "").replace("<s>", "")
                    .replace(task, "").strip())
    return out_text, parsed


def caption_image(payload: dict) -> dict:
    """Core captioning. Returns a structured dict (ok/text/parsed/...)."""
    res = {"ok": False, "text": "", "parsed": {}, "warnings": [], "errors": [],
           "device": None, "dtype": payload.get("dtype"),
           "model_path": payload.get("model_path", "")}
    t0 = time.time()
    task = payload.get("task_prompt", "<MORE_DETAILED_CAPTION>")
    if task not in VALID_TASKS:
        task = "<MORE_DETAILED_CAPTION>"
    try:
        import torch  # noqa: F401
        from PIL import Image
    except Exception as exc:
        res["errors"].append(f"VLM deps unavailable: {exc}")
        return res
    model_path = payload.get("model_path", "")
    if not model_path:
        res["errors"].append("no model_path provided")
        return res
    try:
        image = Image.open(payload["image_path"]).convert("RGB")
    except Exception as exc:
        res["errors"].append(f"cannot open image: {exc}")
        return res

    device = _select_device(payload.get("device", "auto"))
    dtype = _dtype(payload.get("dtype", "float32"), device)
    res["device"] = device
    try:
        processor, model = _load_model(
            model_path, device, dtype,
            payload.get("local_files_only", True),
            payload.get("trust_remote_code", True))
    except Exception as exc:
        res["errors"].append(f"model load failed: {exc}")
        return res
    try:
        text, parsed = _generate(
            processor, model, image, task, payload.get("extra_prompt", ""),
            device, dtype, int(payload.get("max_new_tokens", 256)),
            int(payload.get("num_beams", 3)))
        res.update(ok=True, text=text, parsed=parsed if isinstance(parsed, dict)
                   else {})
    except Exception as exc:
        # optional CPU fallback if MPS/CUDA generation failed
        if device != "cpu" and payload.get("cpu_fallback", True):
            res["warnings"].append(f"{device} generation failed ({exc}); "
                                   "retrying on cpu")
            try:
                import torch
                model = model.to("cpu")
                text, parsed = _generate(
                    processor, model, image, task,
                    payload.get("extra_prompt", ""), "cpu", torch.float32,
                    int(payload.get("max_new_tokens", 256)),
                    int(payload.get("num_beams", 3)))
                res.update(ok=True, text=text, device="cpu",
                           parsed=parsed if isinstance(parsed, dict) else {})
            except Exception as exc2:
                res["errors"].append(f"cpu fallback failed: {exc2}")
        else:
            res["errors"].append(f"generation failed: {exc}")
    res["elapsed_s"] = round(time.time() - t0, 2)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Florence-2 external runner")
    ap.add_argument("--payload-json", default=None)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--task", default="<MORE_DETAILED_CAPTION>")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-beams", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.payload_json:
        try:
            payload = json.loads(open(args.payload_json, encoding="utf-8").read())
        except Exception as exc:
            payload = {}
            result = {"ok": False, "errors": [f"bad payload: {exc}"]}
            if args.result_json:
                open(args.result_json, "w", encoding="utf-8").write(
                    json.dumps(result))
            print(json.dumps(result))
            return 1
    else:
        if not args.image or not args.model:
            print(json.dumps({"ok": False,
                              "errors": ["need --image and --model"]}))
            return 2
        payload = {"image_path": args.image, "model_path": args.model,
                   "device": args.device, "dtype": args.dtype,
                   "task_prompt": args.task, "max_new_tokens": args.max_new_tokens,
                   "num_beams": args.num_beams, "local_files_only": True,
                   "trust_remote_code": True, "cpu_fallback": True}

    result = caption_image(payload)
    if args.result_json:
        try:
            open(args.result_json, "w", encoding="utf-8").write(
                json.dumps(result))
        except Exception as exc:
            print(json.dumps({"ok": False,
                              "errors": [f"cannot write result: {exc}"]}))
            return 1
    if args.json or not args.result_json:
        print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
