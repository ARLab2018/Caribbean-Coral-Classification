"""
VLM Evaluation: Caribbean Coral Species Classifier
===================================================
Evaluates a fine-tuned VLM checkpoint against test.jsonl ground truth.
Loads the model in 4-bit QLoRA (same as training) to stay within 8 GB VRAM.

Usage
─────
  python vlm/eval_vlm.py

  # Custom checkpoint or data path:
  python vlm/eval_vlm.py \
      --checkpoint ~/Independent_study/vlm_checkpoints/best \
      --test_jsonl ~/Independent_study/vlm_crops/test.jsonl \
      --output     ~/Independent_study/vlm_checkpoints/test_eval.json

  # Limit to first N samples (quick smoke-test):
  python vlm/eval_vlm.py --limit 200
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import AutoModelForImageTextToText as _VLMAutoModel
except ImportError:
    from transformers import AutoModelForVision2Seq as _VLMAutoModel  # type: ignore

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CHECKPOINT = str(Path.home() / "Independent_study" / "vlm_checkpoints" / "best")
DEFAULT_TEST_JSONL = str(Path.home() / "Independent_study" / "vlm_crops" / "test.jsonl")
DEFAULT_OUTPUT     = str(Path.home() / "Independent_study" / "vlm_checkpoints" / "test_eval.json")

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
    p = argparse.ArgumentParser(description="Evaluate fine-tuned VLM on coral test set")
    p.add_argument("--checkpoint",   default=DEFAULT_CHECKPOINT)
    p.add_argument("--base_model",   default=DEFAULT_BASE_MODEL)
    p.add_argument("--test_jsonl",   default=DEFAULT_TEST_JSONL)
    p.add_argument("--output",       default=DEFAULT_OUTPUT)
    p.add_argument("--limit",        type=int, default=None,
                   help="Evaluate only first N samples (for quick testing)")
    p.add_argument("--batch_size",   type=int, default=4,
                   help="Inference batch size (default 4; reduce if OOM)")
    p.add_argument("--max_new_tokens", type=int, default=20)
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_processor(checkpoint_dir: str, base_model_id: str):
    ckpt_path = Path(checkpoint_dir)
    is_peft   = (ckpt_path / "adapter_config.json").exists()

    print(f"Loading processor from: {ckpt_path}")
    processor = AutoProcessor.from_pretrained(str(ckpt_path), trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if is_peft:
        print(f"PEFT adapter detected — loading base model: {base_model_id}")
        base = _VLMAutoModel.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map={"": 0} if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(ckpt_path))
    else:
        print(f"Loading full model from: {ckpt_path}")
        model = _VLMAutoModel.from_pretrained(
            str(ckpt_path),
            quantization_config=bnb_config,
            device_map={"": 0} if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    model.eval()
    print("Model ready.\n")
    return model, processor


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_samples(jsonl_path: str, limit=None):
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            messages = rec["messages"]

            img_path  = None
            user_text = None
            for item in messages[0]["content"]:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        img_path = item.get("image")
                    elif item.get("type") == "text":
                        user_text = item.get("text")

            gt = messages[-1]["content"] if messages[-1]["role"] == "assistant" else None

            samples.append({
                "img_path":  img_path,
                "user_text": user_text,
                "gt":        gt,
            })
            if limit and len(samples) >= limit:
                break

    return samples


# ── Inference ─────────────────────────────────────────────────────────────────

USER_PROMPT = (
    "What coral species is in this underwater photo? "
    "Answer with only the scientific species name."
)


def clean_prediction(raw: str) -> str:
    text = raw.strip().split("\n")[0].split(".")[0].strip()
    return text


def run_batch(model, processor, batch_samples, max_new_tokens):
    images  = []
    prompts = []

    for s in batch_samples:
        img_path = s["img_path"]
        if img_path and Path(img_path).exists():
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.new("RGB", (300, 300), (128, 128, 128))
        images.append(img)

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": USER_PROMPT},
        ]}]
        prompts.append(
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    inputs = processor(
        text=prompts, images=images,
        return_tensors="pt", padding=True,
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
    return [clean_prediction(t) for t in raw_texts]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results):
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])

    # Per-class stats
    per_class = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        gt = r["gt"]
        per_class[gt]["total"]   += 1
        per_class[gt]["correct"] += int(r["correct"])

    per_class_acc = {
        sp: {
            "correct": v["correct"],
            "total":   v["total"],
            "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
        }
        for sp, v in sorted(per_class.items())
    }

    # Out-of-vocabulary predictions
    oov = [r for r in results if r["predicted"] not in KNOWN_SPECIES]

    # Confusion: gt -> predicted -> count
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        confusion[r["gt"]][r["predicted"]] += 1

    return {
        "overall_accuracy": correct / total if total else 0.0,
        "correct":          correct,
        "total":            total,
        "oov_count":        len(oov),
        "per_class":        per_class_acc,
        "confusion":        {gt: dict(preds) for gt, preds in confusion.items()},
    }


def print_report(metrics):
    print("\n" + "=" * 62)
    print("EVALUATION RESULTS")
    print("=" * 62)
    print(f"  Overall accuracy : {metrics['overall_accuracy']*100:.2f}%  "
          f"({metrics['correct']}/{metrics['total']})")
    print(f"  OOV predictions  : {metrics['oov_count']}")
    print()
    print(f"  {'Species':<35} {'Acc':>6}  {'Correct':>7}  {'Total':>6}")
    print(f"  {'-'*35} {'-'*6}  {'-'*7}  {'-'*6}")
    for sp, v in sorted(metrics["per_class"].items(), key=lambda x: -x[1]["accuracy"]):
        print(f"  {sp:<35} {v['accuracy']*100:5.1f}%  {v['correct']:>7}  {v['total']:>6}")
    print("=" * 62)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    model, processor = load_model_and_processor(args.checkpoint, args.base_model)

    print(f"Loading test samples from: {args.test_jsonl}")
    samples = load_test_samples(args.test_jsonl, limit=args.limit)
    print(f"  {len(samples)} samples to evaluate\n")

    results  = []
    t_start  = time.time()
    n_done   = 0

    for i in range(0, len(samples), args.batch_size):
        batch = samples[i : i + args.batch_size]
        preds = run_batch(model, processor, batch, args.max_new_tokens)

        for s, pred in zip(batch, preds):
            correct = (pred.strip().lower() == (s["gt"] or "").strip().lower())
            results.append({
                "img_path":  s["img_path"],
                "gt":        s["gt"],
                "predicted": pred,
                "correct":   correct,
            })

        n_done += len(batch)
        elapsed = time.time() - t_start
        sps     = elapsed / n_done
        eta     = sps * (len(samples) - n_done)
        print(f"  [{n_done:>5}/{len(samples)}]  {elapsed/60:.1f} min elapsed  "
              f"ETA {eta/60:.1f} min  ({sps:.2f}s/sample)", flush=True)

    total_elapsed = time.time() - t_start

    metrics = compute_metrics(results)
    metrics["elapsed_seconds"]  = round(total_elapsed, 1)
    metrics["seconds_per_sample"] = round(total_elapsed / len(results), 3)

    print_report(metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "predictions": results}, f, indent=2)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
