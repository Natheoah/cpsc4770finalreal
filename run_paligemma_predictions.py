#!/usr/bin/env python3
"""
paligemma_hle.py
----------------
Prompt PaliGemma 2 3B on questions from the Humanity's Last Exam (HLE)
dataset and save responses as a JSON file matching the format:

  {
    "<question_id>": {
      "model": "google/paligemma2-3b-pt-448",
      "response": "<model output>",
      "usage": {
        "prompt_tokens": N,
        "completion_tokens": N,
        "total_tokens": N
      }
    },
    ...
  }

cais/hle is a public dataset – no token needed for it.
google/paligemma2-* is a gated model – a HF token is required.

Usage
-----
  python paligemma_hle.py --sample 20 --output results.json --hf-token hf_xxx

Full options
------------
  --sample N              Questions to sample (default: all)
  --seed INT              Random seed for sampling (default: 42)
  --device {cuda,mps,cpu} Auto-detected if omitted
  --max-new-tokens INT    Max generation length (default: 256)
  --model-id STR          Any PaliGemma 2 HF model ID
  --hf-token STR          HuggingFace token (overrides HF_TOKEN env var)
  --output PATH           Save JSON results to this file (default: results.json)
  --split STR             HLE dataset split: test | validation (default: test)
  --filter-images         Only run questions that have an image
  --filter-text           Only run questions without an image

Dependencies
------------
  pip install transformers torch pillow datasets huggingface_hub accelerate
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path


# ── Lazy import helper ────────────────────────────────────────────────────────

def _require(module: str, pip_name: str | None = None):
    import importlib
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError:
        pkg = pip_name or module
        sys.exit(
            f"[ERROR] Missing package '{pkg}'.\n"
            f"        Install it with:  pip install {pkg}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Prompt PaliGemma 2 on Humanity's Last Exam (HLE) and save responses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sample", type=int, default=None, metavar="N",
                   help="Number of questions to sample (default: all).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible sampling.")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                   help="Compute device (auto-detected if omitted).")
    p.add_argument("--max-new-tokens", type=int, default=256, dest="max_new_tokens",
                   help="Max new tokens to generate per answer.")
    p.add_argument("--model-id", default="google/paligemma2-3b-pt-448", dest="model_id",
                   help="HuggingFace PaliGemma 2 model ID.")
    p.add_argument("--hf-token", default=None, dest="hf_token",
                   help="HuggingFace access token (overrides HF_TOKEN env var).")
    p.add_argument("--output", default="results.json",
                   help="Path to save responses as JSON.")
    p.add_argument("--split", default="test",
                   help="HLE dataset split to use.")
    modality = p.add_mutually_exclusive_group()
    modality.add_argument("--filter-images", action="store_true", dest="filter_images",
                          help="Only run questions that include an image.")
    modality.add_argument("--filter-text", action="store_true", dest="filter_text",
                          help="Only run questions without an image.")
    return p.parse_args(argv)


# ── Device resolution ─────────────────────────────────────────────────────────

def resolve_device(requested: str | None) -> str:
    torch = _require("torch")
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── HF authentication ─────────────────────────────────────────────────────────

def resolve_token(cli_token: str | None) -> str:
    tok = cli_token or os.environ.get("HF_TOKEN", "").strip() or None
    if not tok:
        sys.exit(
            "[ERROR] A Hugging Face token is required to download PaliGemma (gated model).\n"
            "        Provide one via --hf-token hf_xxx  or  export HF_TOKEN=hf_xxx\n"
            "        Get a token at: https://huggingface.co/settings/tokens\n"
            "        Then accept the PaliGemma licence at: "
            "https://huggingface.co/google/paligemma2-3b-pt-448"
        )
    return tok


def hf_login(token: str):
    huggingface_hub = _require("huggingface_hub")
    try:
        huggingface_hub.login(token=token, add_to_git_credential=False)
        print("[INFO] Logged in to Hugging Face.")
    except Exception as exc:
        sys.exit(f"[ERROR] HF login failed: {exc}")


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_hle(split: str):
    datasets = _require("datasets")
    print(f"[INFO] Loading HLE dataset (split='{split}') …")
    try:
        ds = datasets.load_dataset("cais/hle", split=split)
    except Exception as exc:
        sys.exit(
            f"[ERROR] Failed to load HLE dataset: {exc}\n"
            "        Check your internet connection and try again."
        )
    print(f"[INFO] Dataset loaded: {len(ds)} questions.\n")
    return ds


def filter_dataset(ds, filter_images: bool, filter_text: bool):
    PIL_Image = _require("PIL.Image", "pillow")

    def has_image(row):
        return isinstance(row["image"], PIL_Image.Image)

    if filter_images:
        ds = ds.filter(has_image)
        print(f"[INFO] After --filter-images: {len(ds)} questions remain.")
    elif filter_text:
        ds = ds.filter(lambda row: not has_image(row))
        print(f"[INFO] After --filter-text: {len(ds)} questions remain.")
    return ds


def sample_dataset(ds, n: int | None, seed: int):
    total = len(ds)
    if n is None:
        return list(range(total))
    if n < 1:
        sys.exit(f"[ERROR] --sample must be at least 1 (got {n}).")
    if n > total:
        print(f"[WARNING] --sample {n} > available {total}; using all.")
        return list(range(total))
    rng = random.Random(seed)
    return rng.sample(range(total), n)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, device: str, token: str):
    torch = _require("torch")
    transformers = _require("transformers")

    print(f"[INFO] Loading '{model_id}' on '{device}' …")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    processor = transformers.AutoProcessor.from_pretrained(
        model_id,
        token=token,
    )
    model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        token=token,
    )
    model.eval()
    print("[INFO] Model ready.\n")
    return processor, model


# ── Inference ─────────────────────────────────────────────────────────────────

def run_one(row: dict, processor, model, device: str, max_new_tokens: int) -> dict:
    """
    Run a single HLE row through PaliGemma 2.
    Returns a dict with 'response' and 'usage'.
    """
    torch = _require("torch")
    PIL_Image = _require("PIL.Image", "pillow")
 
    question = row["question"]
    image = row.get("image")  # PIL Image | None
 
    if image is not None and not isinstance(image, PIL_Image.Image):
        if isinstance(image, str):
            import base64, io
            try:
                # Try base64 decode first
                image = PIL_Image.open(io.BytesIO(base64.b64decode(image)))
            except Exception:
                try:
                    # Try as a file path
                    image = PIL_Image.open(image)
                except Exception:
                    print(f"  [WARNING] Could not decode image string; using blank placeholder.")
                    image = None
        else:
            print(f"  [WARNING] Unexpected image type ({type(image).__name__}); using blank placeholder.")
            image = None
 
    # PaliGemma always requires an image — use a blank placeholder for text-only questions
    if image is not None:
        image = image.convert("RGB")
    else:
        image = PIL_Image.new("RGB", (448, 448), color=(0, 0, 0))
 
    inputs = processor(text=question, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
 
    prompt_tokens = inputs["input_ids"].shape[1]
 
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
 
    completion_tokens = output_ids.shape[1] - prompt_tokens
    response = processor.decode(
        output_ids[0][prompt_tokens:], skip_special_tokens=True
    ).strip()
 
    return {
        "response": response,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
 


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)

    for pkg in ("torch", "transformers", "datasets", "PIL", "huggingface_hub"):
        _require(pkg, {"PIL": "pillow"}.get(pkg, pkg))

    token = resolve_token(args.hf_token)
    hf_login(token)

    ds = load_hle(args.split)
    ds = filter_dataset(ds, args.filter_images, args.filter_text)
    indices = sample_dataset(ds, args.sample, args.seed)

    n = len(indices)
    print(f"[INFO] Running {n} question(s) | seed={args.seed} | "
          f"max_new_tokens={args.max_new_tokens}\n")

    device = resolve_device(args.device)
    processor, model = load_model(args.model_id, device, token)

    results = {}

    for rank, idx in enumerate(indices, start=1):
        row = ds[idx]
        question_id = row.get("id", str(idx))
        has_img = row.get("image") is not None

        print(f"── [{rank}/{n}]  id={question_id}  ({'image' if has_img else 'text-only'}) ──")
        print(f"   Q: {row['question'][:200]}{'…' if len(row['question']) > 200 else ''}")

        out = run_one(row, processor, model, device, args.max_new_tokens)

        print(f"   A: {out['response']}")
        print()

        results[question_id] = {
            "model": args.model_id,
            "response": out["response"],
            "usage": out["usage"],
        }

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=4, ensure_ascii=False)
    print(f"[INFO] Results saved → {out_path}  ({n} entries)")

    return results


if __name__ == "__main__":
    main()
