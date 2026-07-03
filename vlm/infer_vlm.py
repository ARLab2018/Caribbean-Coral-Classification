"""
VLM Inference: Caribbean Coral Species Classifier
==================================================
Runs a fine-tuned Vision-Language Model to identify coral species in
bounding-box crop images.

This script can be used in two ways:
  1. Stand-alone: give it a folder of crops and it prints species predictions.
  2. As a drop-in classifier inside the existing infer.py pipeline:
     replace the EfficientNet classify_colony() call with classify_with_vlm().

Usage
─────
  # Classify a single crop:
  python vlm/infer_vlm.py --image /path/to/crop.jpg

  # Classify a folder of crops:
  python vlm/infer_vlm.py --image /path/to/crops/

  # Use a different checkpoint:
  python vlm/infer_vlm.py --image crops/ \\
      --checkpoint ~/Independent_study/vlm_checkpoints/best

  # Save results to CSV:
  python vlm/infer_vlm.py --image crops/ --output results.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor

# transformers 5.0 renamed AutoModelForVision2Seq -> AutoModelForImageTextToText
try:
    from transformers import AutoModelForImageTextToText as _VLMAutoModel
except ImportError:
    from transformers import AutoModelForVision2Seq as _VLMAutoModel  # type: ignore[no-redef]

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL  = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CHECKPOINT  = str(
    Path.home() / "Independent_study" / "vlm_checkpoints" / "best"
)

USER_PROMPT = (
    "What coral species is in this underwater photo? "
    "Answer with only the scientific species name."
)

# Known species (used to clean / validate model output)
KNOWN_SPECIES = {
    "Acropora tenuifolia",
    "Agaricia agaricites",
    "Colpophyllia natans",
    "Lobophyllia spp",
    "Madracis auretenra",
    "Madracis mirabilis",
    "Meandrina meandrites",
    "Millepora spp",
    "Montastraea cavernosa",
    "Orbicella annularis",
    "Orbicella faveolata",
    "Orbicella franksi",
    "Porites astreoides",
    "Porites porites",
    "Pseudodiploria strigosa",
    "Siderastrea siderea",
    "Stephanocoenia intersepta",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run VLM coral species classifier on crop images"
    )
    p.add_argument("--image",        required=True,
                   help="Path to a single crop image or a directory of crops")
    p.add_argument("--checkpoint",   default=DEFAULT_CHECKPOINT,
                   help="Path to the fine-tuned VLM checkpoint directory")
    p.add_argument("--base_model",   default=DEFAULT_BASE_MODEL,
                   help="Base model ID (needed when loading a PEFT adapter)")
    p.add_argument("--output",       default=None,
                   help="Optional path to save results CSV")
    p.add_argument("--max_new_tokens", type=int, default=20,
                   help="Max tokens to generate for the species name (default 20)")
    p.add_argument("--device",       default="auto",
                   help="cuda / cpu / auto")
    p.add_argument("--batch_size",   type=int, default=1,
                   help="Batch size for inference (default 1)")
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_processor(checkpoint_dir: str, base_model_id: str, device: str):
    """
    Load processor and model from a checkpoint directory.
    Automatically detects whether the checkpoint is a full model save
    or a PEFT adapter (presence of adapter_config.json).
    """
    ckpt_path = Path(checkpoint_dir)
    is_peft   = (ckpt_path / "adapter_config.json").exists()

    print(f"Loading processor from: {ckpt_path}")
    processor = AutoProcessor.from_pretrained(
        str(ckpt_path), trust_remote_code=True
    )

    if is_peft:
        print(f"Detected PEFT adapter. Loading base model: {base_model_id}")
        base = _VLMAutoModel.from_pretrained(
            base_model_id,
            dtype=torch.bfloat16,
            device_map=device if device != "auto" else "auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(ckpt_path))
        model = model.merge_and_unload()   # fuse LoRA weights for faster inference
    else:
        print(f"Loading full model from: {ckpt_path}")
        model = _VLMAutoModel.from_pretrained(
            str(ckpt_path),
            dtype=torch.bfloat16,
            device_map=device if device != "auto" else "auto",
            trust_remote_code=True,
        )

    model.eval()
    print("Model loaded.")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────

def clean_prediction(raw: str) -> str:
    """
    Strip whitespace and any repeated phrases from the model output.
    Return the first sentence / species name.
    """
    text = raw.strip()
    # Take only the first line / sentence
    text = text.split("\n")[0].split(".")[0].strip()
    return text


def classify_with_vlm(
    model,
    processor,
    image: Image.Image,
    max_new_tokens: int = 20,
) -> tuple[str, str]:
    """
    Classify a single PIL image with the VLM.

    Returns
    -------
    predicted_species : str   — cleaned species name
    raw_output        : str   — raw model text output
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ],
        }
    ]

    # Apply chat template
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    )
    # Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy decoding for reproducibility
            temperature=None,
            top_p=None,
        )

    # Decode only the newly generated tokens
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_text  = processor.batch_decode(generated, skip_special_tokens=True)[0]
    species   = clean_prediction(raw_text)
    return species, raw_text


def classify_batch(
    model,
    processor,
    images: list[Image.Image],
    max_new_tokens: int = 20,
) -> list[tuple[str, str]]:
    """Classify a batch of PIL images."""
    messages_list = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ]
        for _ in images
    ]

    prompts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]

    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_texts  = processor.batch_decode(generated, skip_special_tokens=True)
    return [(clean_prediction(t), t) for t in raw_texts]


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_images(input_path: Path) -> list[Path]:
    """Collect all image files from a file or directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in exts)


def main():
    args = parse_args()

    # ── Load model ─────────────────────────────────────────────────────────
    model, processor = load_model_and_processor(
        args.checkpoint, args.base_model, args.device
    )

    # ── Collect images ─────────────────────────────────────────────────────
    img_paths = collect_images(Path(args.image))
    if not img_paths:
        print(f"No images found at: {args.image}")
        return
    print(f"\nFound {len(img_paths)} image(s)\n")

    # ── Run inference ──────────────────────────────────────────────────────
    results = []
    t_start = time.time()

    # Process in batches
    batch_size = args.batch_size
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i : i + batch_size]
        batch_imgs  = []
        for p in batch_paths:
            try:
                batch_imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"  [!] Could not open {p.name}: {e}")
                batch_imgs.append(Image.new("RGB", (300, 300), (128, 128, 128)))

        preds = classify_batch(model, processor, batch_imgs, args.max_new_tokens)

        for path, (species, raw) in zip(batch_paths, preds):
            in_vocab = species in KNOWN_SPECIES
            flag     = "✓" if in_vocab else "?"
            print(f"  [{flag}] {path.name:<50}  ->  {species}")
            if not in_vocab:
                print(f"       (raw output: {repr(raw)})")

            results.append({
                "image":     str(path),
                "predicted": species,
                "in_vocab":  in_vocab,
                "raw":       raw,
            })

    elapsed = time.time() - t_start
    print(f"\nProcessed {len(img_paths)} images in {elapsed:.1f}s "
          f"({elapsed / len(img_paths):.2f}s per image)")

    # ── Save CSV ───────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "predicted",
                                                    "in_vocab", "raw"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to: {args.output}")

    # ── Summary ────────────────────────────────────────────────────────────
    in_vocab_n = sum(r["in_vocab"] for r in results)
    print(f"\nIn-vocabulary predictions : {in_vocab_n}/{len(results)}")

    from collections import Counter
    counts = Counter(r["predicted"] for r in results if r["in_vocab"])
    if counts:
        print("Species breakdown:")
        for sp, n in counts.most_common():
            print(f"  {sp:<35} {n}")


if __name__ == "__main__":
    main()
