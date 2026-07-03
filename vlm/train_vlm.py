"""
VLM Fine-Tuning: Caribbean Coral Species Classification
========================================================
Supervised fine-tuning of a Vision-Language Model using LoRA/QLoRA.
The vision encoder is frozen; only the language model attention layers
receive LoRA adapters.

Supports:
  - SmolVLM-Instruct (HuggingFaceTB/SmolVLM-Instruct)  [default, ~2B params]
  - Qwen2.5-VL-7B-Instruct
  - meta-llama/Llama-3.2-11B-Vision-Instruct

The dataset must be in the conversational JSONL format produced by
build_vlm_dataset.py — each line is a dict with a "messages" key.

Label masking
─────────────
The data collator sets labels=-100 for all tokens that belong to the user
turn (image tokens + prompt text).  Only the assistant's species-name reply
contributes to the cross-entropy loss.

Usage
─────
  # SmolVLM (8-12 GB VRAM, bfloat16):
  python vlm/train_vlm.py

  # Qwen2.5-VL-7B with 4-bit QLoRA (needs ~16 GB):
  python vlm/train_vlm.py --model_id Qwen/Qwen2.5-VL-7B-Instruct --use_4bit

  # Llama-3.2-11B with 4-bit QLoRA (needs ~24 GB):
  python vlm/train_vlm.py \\
      --model_id meta-llama/Llama-3.2-11B-Vision-Instruct --use_4bit

  # Resume from checkpoint:
  python vlm/train_vlm.py --resume_from vlm_checkpoints/checkpoint-500
"""

import argparse
import json
import os
from pathlib import Path

# ── Version guard — check before importing to give a clear error message ───────
def _check_versions():
    import importlib.metadata as _meta
    from packaging.version import Version as V

    required = {
        # transformers 4.39+ has AutoModelForVision2Seq;
        # transformers 5.0+ renamed it to AutoModelForImageTextToText.
        # Both are handled below — any version >= 4.39 is fine.
        "transformers": "4.39.0",
        "peft":         "0.11.0",
        "trl":          "0.8.6",
        "accelerate":   "0.30.0",
    }
    missing = []
    for pkg, min_ver in required.items():
        try:
            installed = _meta.version(pkg)
            if V(installed) < V(min_ver):
                missing.append(f"  {pkg}=={installed}  (need >= {min_ver})")
        except _meta.PackageNotFoundError:
            missing.append(f"  {pkg}  NOT INSTALLED  (need >= {min_ver})")
    if missing:
        raise RuntimeError(
            "One or more packages are too old. Run:\n"
            "  pip install -U transformers>=4.51.0 trl>=0.8.6 "
            "peft>=0.11.0 accelerate>=0.30.0\n\n"
            "Offending packages:\n" + "\n".join(missing)
        )

    # Qwen2.5-VL's AutoProcessor always loads a Qwen2VLVideoProcessor
    # sub-component that requires torchvision — even for image-only workloads.
    try:
        import torchvision  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "torchvision is required by Qwen2.5-VL's processor. Install it with:\n"
            "  pip install torchvision\n"
            "If you have a custom/nightly PyTorch build, use:\n"
            "  pip install torchvision --index-url https://download.pytorch.org/whl/cu124"
        )

_check_versions()
# ── End version guard ─────────────────────────────────────────────────────────

import torch
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments

# transformers 5.0 renamed AutoModelForVision2Seq -> AutoModelForImageTextToText
try:
    from transformers import AutoModelForImageTextToText as _VLMAutoModel
except ImportError:
    from transformers import AutoModelForVision2Seq as _VLMAutoModel  # type: ignore[no-redef]

# SFTTrainer (trl 1.x) always pre-tokenises the dataset by calling
# apply_chat_template without actual images, which raises StopIteration on
# multimodal data.  We use the base transformers Trainer instead — it calls
# the collator directly with no dataset pre-processing.

# ── Defaults ──────────────────────────────────────────────────────────────────

# Default model choices (all confirmed supported in transformers 5.x):
#   Qwen/Qwen2.5-VL-3B-Instruct  -- 3B, bfloat16, ~8 GB VRAM  [default]
#   Qwen/Qwen2.5-VL-7B-Instruct  -- 7B, needs --use_4bit for <24 GB VRAM
#   meta-llama/Llama-3.2-11B-Vision-Instruct -- 11B, needs --use_4bit
#
# SmolVLM-Instruct and SmolVLM2-2.2B-Instruct are NOT compatible with
# transformers >= 5.0 due to missing image_processor_type in their
# preprocessor_config.json and unregistered model types.
DEFAULT_MODEL_ID   = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_DATA_DIR   = str(Path.home() / "Independent_study" / "vlm_crops")
DEFAULT_OUTPUT_DIR = str(Path.home() / "Independent_study" / "vlm_checkpoints")

# LoRA
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
# Common attention projection names across VLM families
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training
LEARNING_RATE       = 2e-4
NUM_EPOCHS          = 3
BATCH_SIZE          = 1        # per device (OOM with 4 on RTX 4060 8GB + QLoRA)
GRAD_ACCUM_STEPS    = 16       # effective batch = 1 * 16 = 16
MAX_SEQ_LEN         = 1024     # max token length per example
WARMUP_RATIO        = 0.05
WEIGHT_DECAY        = 0.01
EVAL_STRATEGY       = "epoch"
SAVE_STRATEGY       = "epoch"
LOGGING_STEPS       = 25
# dataloader_num_workers=0: use the main process for data loading.
# With device_map='auto' (CPU/GPU offload), forked worker processes
# die immediately due to CUDA multiprocessing incompatibility, producing
# 0 batches.  Single-process loading avoids this entirely.
DATALOADER_WORKERS  = 0
SEED                = 42


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune a VLM on Caribbean coral crops using LoRA/QLoRA"
    )
    p.add_argument("--model_id",   default=DEFAULT_MODEL_ID,
                   help="HuggingFace model ID (default: SmolVLM-Instruct)")
    p.add_argument("--data_dir",   default=DEFAULT_DATA_DIR,
                   help="Directory containing train.jsonl / val.jsonl / test.jsonl")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                   help="Where to save checkpoints")
    p.add_argument("--use_4bit",   action=argparse.BooleanOptionalAction, default=True,
                   help="Load model in 4-bit NF4 (QLoRA) — saves VRAM (default: on)")
    p.add_argument("--use_8bit",   action="store_true",
                   help="Load model in 8-bit — moderate VRAM saving")
    p.add_argument("--epochs",     type=int, default=NUM_EPOCHS)
    p.add_argument("--lr",         type=float, default=LEARNING_RATE)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM_STEPS)
    p.add_argument("--lora_r",     type=int, default=LORA_R)
    p.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--max_seq_len",type=int, default=MAX_SEQ_LEN)
    p.add_argument("--resume_from",default=None,
                   help="Path to a checkpoint directory to resume from")
    p.add_argument("--no_freeze_vision", action="store_true",
                   help="Do NOT freeze the vision encoder (uses more memory)")
    return p.parse_args()


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_hf_dataset(jsonl_path: str) -> Dataset:
    """
    Convert a JSONL file to a HuggingFace Dataset.

    Stores messages as a JSON string to avoid PyArrow struct-schema unification.
    When Dataset.from_list() sees mixed list/non-list content (image items vs
    text items), PyArrow injects phantom keys (image: None on text items,
    text: None on image items).  Qwen2.5-VL's Jinja2 chat template checks
    `'image' in content` (key existence, not value), so every item is then
    treated as an image, causing StopIteration in processor() which the
    DataLoader silently interprets as "exhausted" -> 0 batches.
    Storing as JSON string sidesteps Arrow's schema normalisation entirely.
    """
    records = load_jsonl(jsonl_path)
    return Dataset.from_list(
        [{"messages_json": json.dumps(r["messages"])} for r in records]
    )


# ── Collator with label masking ───────────────────────────────────────────────

class CoralVLMCollator:
    """
    Processes a batch of conversation records for SFT.

    For each example the conversation looks like:
        user:      [<image>] What coral species …?
        assistant: Madracis mirabilis

    We:
      1. Apply the processor's chat template to get input_ids + attention_mask.
      2. Load the image from the path embedded in the user message.
      3. Set labels=-100 for all positions that are NOT in the assistant turn,
         so cross-entropy is only computed on the species-name tokens.
    """

    def __init__(self, processor, max_length: int = MAX_SEQ_LEN):
        self.processor  = processor
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        # Deserialise messages from the JSON-string column (avoids PyArrow
        # struct-schema unification which injects phantom keys and breaks the
        # Qwen2.5-VL chat template's `'image' in content` check).
        parsed_batch = [json.loads(example["messages_json"]) for example in batch]

        texts  = []
        images = []

        for messages in parsed_batch:
            # User turn: extract image path from content list
            img_path = None
            for item in messages[0]["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    img_path = item.get("image")

            if img_path and Path(img_path).exists():
                img = Image.open(img_path).convert("RGB")
            else:
                img = Image.new("RGB", (300, 300), color=(128, 128, 128))

            images.append(img)

            formatted = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(formatted)

        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = inputs["input_ids"].clone()

        # image_token_id is the placeholder token the processor inserts for
        # each image patch.  We count how many appear in the full sequence and
        # use that to correct the prompt-only tokenizer length.
        img_token_id = getattr(self.processor, "image_token_id", None)
        if img_token_id is None:
            # Qwen2.5-VL stores it as a string attribute on the tokenizer
            img_token_id = getattr(
                self.processor.tokenizer, "image_token_id", None
            )

        for i, messages in enumerate(parsed_batch):
            # Build prompt-only text (no assistant turn) and tokenize with the
            # plain tokenizer — this gives text token count without image patches.
            # Then look up how many image-patch tokens are in the full sequence
            # and add them to get the true mask boundary.
            #
            # Why not re-run the full processor?  Running processor() twice per
            # sample doubles peak memory in the collator and causes OOM on 8GB
            # VRAM before the first backward pass.
            prompt_text = self.processor.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            # Strip the image placeholder string before text-tokenizing so we
            # don't count it as text tokens.
            img_placeholder = getattr(
                self.processor, "image_token", "<|image_pad|>"
            )
            prompt_text_no_img = prompt_text.replace(img_placeholder, "")
            prompt_text_ids = self.processor.tokenizer(
                prompt_text_no_img,
                add_special_tokens=False,
            )["input_ids"]
            n_text_tokens = len(prompt_text_ids)

            # Count image-patch tokens in the full (already-encoded) sequence
            if img_token_id is not None:
                n_img_tokens = int(
                    (inputs["input_ids"][i] == img_token_id).sum().item()
                )
            else:
                n_img_tokens = 0

            mask_until = n_text_tokens + n_img_tokens
            labels[i, :mask_until] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        inputs["labels"] = labels
        return inputs


# ── Exact-match accuracy callback ─────────────────────────────────────────────

def compute_exact_match(eval_pred):
    """
    Custom metric: fraction of predictions that exactly match the ground truth
    species name (case-insensitive, stripped).
    """
    logits, labels = eval_pred
    # logits: (batch, seq_len, vocab_size) — take argmax
    pred_ids  = logits.argmax(axis=-1)

    # Only evaluate on non-masked positions
    mask      = labels != -100
    correct   = ((pred_ids == labels) & mask).sum()
    total     = mask.sum()
    return {"exact_match": float(correct) / float(total) if total > 0 else 0.0}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_processor(args):
    """Load the VLM and its processor with optional quantization."""
    print(f"Loading model: {args.model_id}")

    # Quantization config
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("  Quantization: 4-bit NF4 (QLoRA)")
    elif args.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        print("  Quantization: 8-bit")
    else:
        print("  Quantization: none (bfloat16)")

    # transformers 5.x deprecated torch_dtype in favour of dtype
    dtype_kwarg = (
        {} if (args.use_4bit or args.use_8bit)
        else {"dtype": torch.bfloat16}
    )

    model = _VLMAutoModel.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        **dtype_kwarg,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    # Ensure a valid pad token exists.
    # SmolVLM2 sets pad_token_id=128002 which can be out of vocab range;
    # reset it to eos_token_id to avoid unexpected behaviour.
    tok = processor.tokenizer
    vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else len(tok)
    if tok.pad_token is None or (
        tok.pad_token_id is not None and tok.pad_token_id >= vocab_size
    ):
        tok.pad_token    = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        model.config.pad_token_id = tok.eos_token_id
        print(f"  pad_token set to eos_token (id={tok.eos_token_id})")

    return model, processor


def apply_lora(model, args):
    """Wrap the model with LoRA adapters targeting attention layers."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Required for gradient checkpointing + PEFT: ensures frozen backbone
    # layers still propagate gradients to earlier LoRA modules.
    model.enable_input_require_grads()
    return model


def freeze_vision_encoder(model):
    """
    Freeze the vision encoder (image feature extractor) entirely.
    Only the language model's LoRA-adapted attention layers will be trained.

    Tries a list of known attribute paths first, then falls back to scanning
    named_children() for any module whose name suggests it is a vision component.
    Gracefully handles meta-device parameters (from device_map='auto').
    """
    def _try_freeze(module, label):
        """Freeze module; return True on success, False if all params are on meta."""
        real_params = [p for p in module.parameters()
                       if p.device.type != "meta"]
        if not real_params and any(True for _ in module.parameters()):
            # All parameters are on meta device — can't set requires_grad.
            print(f"  Warning: {label} has all parameters on meta device; "
                  "skipping freeze (they won't receive gradients anyway).")
            return True   # treat as frozen — meta params have no gradients
        try:
            module.requires_grad_(False)
            n = sum(p.numel() for p in real_params)
            print(f"  Frozen vision encoder ({label}): {n / 1e6:.1f}M parameters")
            return True
        except Exception as e:
            print(f"  Warning: could not freeze {label}: {e}")
            return False

    # ── 1. Try known attribute paths ──────────────────────────────────────────
    candidates = [
        "visual",              # Qwen2.5-VL, Qwen2-VL  (top-level attribute)
        "model.visual",        # wrapped variant
        "model.vision_model",  # idefics3 / SmolVLM
        "model.vision_tower",  # LLaVA
        "vision_tower",        # InternVL
        "model.vision_encoder",
    ]
    for attr_path in candidates:
        obj = model
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and obj is not model:
            if _try_freeze(obj, attr_path):
                return

    # ── 2. Fallback: scan top-level named_children ────────────────────────────
    vision_keywords = {"visual", "vision", "image_encoder", "patch_embed",
                       "vision_tower", "vision_model", "encoder"}
    for name, child in model.named_children():
        if any(kw in name.lower() for kw in vision_keywords):
            if _try_freeze(child, f"<auto-detected: {name}>"):
                return

    print("  Warning: could not identify vision encoder — "
          "no parameters frozen. Pass --no_freeze_vision to suppress this.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"VLM Fine-Tuning: Caribbean Coral Classification")
    print(f"{'='*60}")
    print(f"  Model       : {args.model_id}")
    print(f"  Data dir    : {data_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  LR          : {args.lr}")
    print(f"  Batch size  : {args.batch_size} x {args.grad_accum} grad accum")
    print(f"  LoRA r/alpha: {args.lora_r}/{args.lora_alpha}")
    print(f"  4-bit QLoRA : {args.use_4bit}")
    print()

    # ── 1. Load model & processor ──────────────────────────────────────────
    model, processor = load_model_and_processor(args)

    # ── 2. Freeze vision encoder ───────────────────────────────────────────
    if not args.no_freeze_vision:
        print("Freezing vision encoder ...")
        freeze_vision_encoder(model)
    else:
        print("Vision encoder NOT frozen (--no_freeze_vision)")

    # ── 3. Apply LoRA ──────────────────────────────────────────────────────
    print("\nApplying LoRA ...")
    model = apply_lora(model, args)

    # ── 4. Load datasets ───────────────────────────────────────────────────
    train_jsonl = data_dir / "train.jsonl"
    val_jsonl   = data_dir / "val.jsonl"

    if not train_jsonl.exists():
        raise FileNotFoundError(
            f"train.jsonl not found at {train_jsonl}. "
            "Run build_vlm_dataset.py first."
        )

    print(f"\nLoading datasets from {data_dir} ...")
    train_dataset = build_hf_dataset(str(train_jsonl))
    eval_dataset  = build_hf_dataset(str(val_jsonl)) if val_jsonl.exists() else None

    print(f"  Train examples : {len(train_dataset):,}")
    if eval_dataset:
        print(f"  Val   examples : {len(eval_dataset):,}")

    # ── 5. Data collator ───────────────────────────────────────────────────
    collator = CoralVLMCollator(processor, max_length=args.max_seq_len)

    # ── 6. Training config ─────────────────────────────────────────────────
    # Use base TrainingArguments + Trainer, NOT SFTTrainer.
    # SFTTrainer 1.x pre-tokenises the dataset by calling apply_chat_template
    # without images, which raises StopIteration on multimodal data.
    num_update_steps = (
        len(train_dataset) // (args.batch_size * args.grad_accum)
    ) * args.epochs
    warmup_steps = max(1, int(num_update_steps * WARMUP_RATIO))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,   # warmup_ratio deprecated in transformers 5.2
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        eval_strategy=EVAL_STRATEGY if eval_dataset else "no",
        save_strategy=SAVE_STRATEGY,
        logging_steps=LOGGING_STEPS,
        # load_best_model_at_end requires eval_loss to be present;
        # disable to avoid KeyError if eval produces no metric on first call.
        load_best_model_at_end=False,
        report_to="none",           # set to "wandb" if you use W&B
        seed=SEED,
        dataloader_num_workers=DATALOADER_WORKERS,
        dataloader_pin_memory=False,
        remove_unused_columns=False,   # keep "messages_json" column for collator
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    print(f"  Warmup steps  : {warmup_steps} / {num_update_steps} total")

    # ── 7. Trainer ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        # compute_metrics omitted: generative VLMs produce very large logit
        # tensors during eval prediction steps which can OOM.  eval_loss is
        # sufficient to track improvement and select the best checkpoint.
    )

    # ── 8. Train ───────────────────────────────────────────────────────────
    print("\nStarting training ...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    # ── 9. Save best model & processor ────────────────────────────────────
    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    print(f"\nBest model + processor saved to: {best_dir}")

    # ── 10. Final evaluation on val set ────────────────────────────────────
    if eval_dataset:
        print("\nRunning final evaluation on validation set ...")
        metrics = trainer.evaluate()
        val_loss = metrics.get("eval_loss")
        exact_match = metrics.get("eval_exact_match")
        print(f"  val_loss      : {val_loss:.4f}" if val_loss is not None else "  val_loss      : N/A")
        print(f"  exact_match   : {exact_match:.4f}" if exact_match is not None else "  exact_match   : N/A")

        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Results saved to: {output_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
