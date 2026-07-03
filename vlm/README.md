# VLM Phase 1 — Caribbean Coral Classification

This directory contains the **Vision-Language Model (VLM)** upgrade of the original
two-stage pipeline (MobileSAM + EfficientNet-B3).  A single VLM is trained to
directly answer *"What coral species is in this image?"* from a natural-background
crop, eliminating the separate classify step.

---

## File overview

| File | Purpose |
|---|---|
| `build_vlm_dataset.py` | Extract crops from CoralNet images → write JSONL for SFT |
| `train_vlm.py` | LoRA/QLoRA supervised fine-tuning with `SFTTrainer` |
| `infer_vlm.py` | Run the fine-tuned VLM on crop images |
| `requirements_vlm.txt` | Python dependencies |

---

## Quick-start

### 0. Create and activate a conda environment

```bash
conda create -n vlm python=3.11 -y
conda activate vlm
pip install -r vlm/requirements_vlm.txt
```

> **Flash Attention 2** (optional, Ampere+ GPU only — speeds up training ~2×):
> ```bash
> pip install flash-attn --no-build-isolation
> ```

---

### 1. Build the dataset

```bash
# Dry run first — see expected crop and JSONL counts:
python vlm/build_vlm_dataset.py --dry_run

# Full extraction (reads CoralNet_Data/, writes vlm_crops/):
python vlm/build_vlm_dataset.py
```

**If you already have `patches_option_f/` on disk**, you can point directly at the
existing `split_manifest.csv` to skip re-extraction of crops that already exist and
just regenerate the JSONL:

```bash
python vlm/build_vlm_dataset.py \
    --manifest ~/Independent_study/patches_option_f/split_manifest.csv
```

Key CLI options:

| Flag | Default | Description |
|---|---|---|
| `--patch_size` | `300` | Square crop side in pixels |
| `--oversample_threshold` | `500` | Species below this → oversampled |
| `--oversample_factor` | `4` | Repeat factor for thin species |
| `--dry_run` | — | Print stats, write nothing |

---

### 2. Fine-tune

> [!WARNING]
> **SmolVLM-Instruct** (v1) and **SmolVLM2-2.2B-Instruct** are **not compatible**
> with `transformers >= 5.0`. Their `preprocessor_config.json` lacks the
> `image_processor_type` field that transformers 5.x requires for auto-detection.
> Use one of the models below instead.

#### Qwen2.5-VL-3B-Instruct (default, ~8 GB VRAM, bfloat16)

```bash
python vlm/train_vlm.py
```

#### Qwen2.5-VL-7B with 4-bit QLoRA (~16 GB VRAM)

```bash
python vlm/train_vlm.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --use_4bit
```

#### Llama-3.2-11B with 4-bit QLoRA (~24 GB VRAM)

```bash
python vlm/train_vlm.py \
    --model_id meta-llama/Llama-3.2-11B-Vision-Instruct \
    --use_4bit
```

#### Resume from checkpoint

```bash
python vlm/train_vlm.py \
    --resume_from ~/Independent_study/vlm_checkpoints/checkpoint-500
```

Key CLI options:

| Flag | Default | Description |
|---|---|---|
| `--model_id` | SmolVLM-Instruct | HuggingFace model ID |
| `--use_4bit` | — | Enable 4-bit NF4 QLoRA |
| `--epochs` | `3` | Number of training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--no_freeze_vision` | — | Train vision encoder too (more VRAM) |

Checkpoints are saved to `~/Independent_study/vlm_checkpoints/`.
The best model (by val loss) is saved to `vlm_checkpoints/best/`.

---

### 3. Inference

```bash
# Single crop:
python vlm/infer_vlm.py \
    --image ~/Independent_study/vlm_crops/test/Madracis_mirabilis/img_r300_c400.jpg

# Whole folder:
python vlm/infer_vlm.py \
    --image ~/Independent_study/vlm_crops/test/ \
    --output predictions.csv
```

#### Plug into the existing SAM pipeline

The existing `infer.py` uses EfficientNet for classification. To swap in the VLM,
replace the `classify_colony()` call with:

```python
from vlm.infer_vlm import classify_with_vlm, load_model_and_processor
from PIL import Image

# Load once at startup:
vlm_model, vlm_processor = load_model_and_processor(
    checkpoint_dir="vlm_checkpoints/best",
    base_model_id="HuggingFaceTB/SmolVLM-Instruct",
    device="auto",
)

# Inside the per-colony loop (replaces classify_colony()):
crop_pil = Image.fromarray(crop_rgb)
species, raw = classify_with_vlm(vlm_model, vlm_processor, crop_pil)
```

---

## Architecture summary

```
[CoralNet CSVs]
      │
      ▼
build_vlm_dataset.py
  • bbox crop (300×300, natural background)
  • image-level 70/15/15 split
  • JSONL: user asks "What species?" → assistant answers species name
  • thin species oversampled 4× in JSONL (no disk duplication)
      │
      ▼
train_vlm.py
  • Load VLM (SmolVLM / Qwen2.5-VL / Llama-3.2-Vision)
  • Freeze vision encoder (no gradient)
  • Apply LoRA to q/k/v/o_proj (LM attention only)
  • SFTTrainer with custom collator — labels=-100 on user tokens
  • 3 epochs, cosine LR, bfloat16 or 4-bit QLoRA
      │
      ▼
vlm_checkpoints/best/
      │
      ▼
infer_vlm.py
  • Load checkpoint (full or PEFT adapter)
  • Greedy decode → species name string
  • Optional: chain after MobileSAM for full detection pipeline
```

---

## Future phases (not yet implemented)

- **Phase 2 — Alignment**: DPO / GRPO to penalise hallucinations and improve
  reasoning quality.
- **Phase 3 — Deployment**: Serve via vLLM (PagedAttention) or compile to a
  TensorRT engine running on NVIDIA Triton Inference Server.
