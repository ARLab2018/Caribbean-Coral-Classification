# Caribbean Coral Reef Species Classification

End-to-end deep learning pipeline for detecting and classifying **17 Caribbean coral species** from underwater photoquadrat images.

**Phase 1** combined instance-level coral colony segmentation (**SAM 2**, **MobileSAM**, **CoralSCOP**) with a fine-tuned **EfficientNet-B3** classifier.

**Phase 2 (current)** transitions the pipeline to a unified **Vision-Language Model** architecture using **Qwen2.5-VL-3B** fine-tuned with LoRA — replacing the two-stage segmenter + classifier with a single conversational VLM.

Independent study project — full report in [`docs/report.pdf`](docs/report.pdf).

---

## Results — Phase 1 (Legacy CV Pipeline)

All completed experiments were evaluated on an image-level held-out test split across 17 coral species.

| # | Segmenter | Classifier | Training patches | Accuracy | Macro-F1 |
|---|-----------|------------|------------------|----------|----------|
| 1 | MobileSAM | EfficientNet-B3 | 49,850 | 70.1% | 0.606 |
| 2 | CoralSCOP | EfficientNet-B3 | See report | See report | See report |
| 3 | **SAM 2** | **EfficientNet-B3** | **43,672** | **72.6%** | **0.624** |
| 4 | SAM 2 | DINOv2-S | 43,672 | 46.8% | 0.376 |

### Per-species highlights (best model: SAM 2 + EfficientNet-B3)

| Species | F1 | Test n |
|---|---:|---:|
| *Madracis mirabilis* | 0.882 | 1,479 |
| *Orbicella annularis* | 0.794 | 1,385 |
| *Siderastrea siderea* | 0.774 | 408 |
| *Agaricia agaricites* | 0.768 | 1,371 |
| *Porites porites* | 0.431 | 111 |
| *Orbicella franksi* | 0.200 | 90 |

---

## Phase 2 — VLM Transition

### Motivation

The legacy pipeline required two separate models (segmenter + classifier) and was sensitive to segmentation failures. The VLM approach:
- Uses **natural-background crops** (no segmentation needed) — preserving reef context that transformers rely on
- Produces **interpretable species names** as free text output rather than class-index logits
- Is a single end-to-end model fine-tuned with LoRA on only **0.196% of parameters**

### Architecture

| Component | Detail |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Fine-tuning | LoRA (r=16, α=32) on `q/k/v/o_proj` layers |
| Vision encoder | Frozen (668.7M params) |
| Trainable params | 7.37M / 3.76B (0.196%) |
| Training data | 42,355 conversational examples (17 species, 4× oversampling for rare classes) |
| Loss | Cross-entropy on assistant tokens only (label masking) |

### Dataset

- **Source**: 8 CoralNet survey datasets, 66,219 annotations
- **Split**: 70 / 15 / 15 (image-level, no data leakage)
- **Crop size**: 300 × 300 px with natural reef background preserved
- **Format**: Conversational JSONL — user asks "What coral species is in this photo?", assistant answers with the scientific species name

---

## Repository layout

```text
caribbean-coral-classification/
│
├── README.md
├── .gitignore
│
├── docs/
│   └── report.pdf
│
├── src/                                  # Phase 1 — Legacy CV pipeline
│   ├── data/
│   │   └── build_manifest.py             # Build split_manifest.csv from CoralNet CSVs
│   ├── patches/
│   │   ├── extract_mobilesam.py          # MobileSAM masked crops
│   │   ├── extract_coralscop.py          # CoralSCOP masked crops
│   │   └── extract_sam2.py               # SAM 2 point-prompt crops
│   ├── train.py                          # Train EfficientNet-B3 / DINOv2-S
│   └── inference/
│       ├── infer_mobilesam_effnet.py
│       ├── infer_coralscop_effnet.py
│       ├── infer_sam2_effnet.py          # Best Phase 1 model
│       └── infer_sam2_dinov2.py
│
├── vlm/                                  # Phase 2 — VLM pipeline (new)
│   ├── README.md                         # VLM-specific docs & usage
│   ├── requirements_vlm.txt              # Python dependencies
│   ├── build_vlm_dataset.py              # CoralNet CSVs → conversational JSONL crops
│   ├── train_vlm.py                      # LoRA SFT training (Qwen2.5-VL-3B)
│   ├── eval_vlm.py                       # Test-set evaluation & confusion matrix
│   └── infer_vlm.py                      # Single/batch inference
│
├── patch_extractor_v2.py                 # Phase 1 patch extraction utility
├── infer.py                              # Phase 1 two-stage inference script
│
├── data/                                 # NOT committed — download separately
├── models/                               # NOT committed — download separately
└── outputs/                              # NOT committed — generated locally
```

---

## Quick start — Phase 2 (VLM)

```bash
# 1. Create environment
conda create -n vlm python=3.11 -y && conda activate vlm
pip install -r vlm/requirements_vlm.txt

# 2. Extract dataset crops (first time only; ~30–60 min)
python vlm/build_vlm_dataset.py
# Preview expected counts without writing:
python vlm/build_vlm_dataset.py --dry_run

# 3. Train (Qwen2.5-VL-3B, bfloat16, ~8–12 GB VRAM)
python vlm/train_vlm.py

# With 4-bit QLoRA for 8 GB VRAM:
python vlm/train_vlm.py --use_4bit

# 4. Evaluate on test set
python vlm/eval_vlm.py

# 5. Run inference on a single image
python vlm/infer_vlm.py --image /path/to/crop.jpg
```

See [`vlm/README.md`](vlm/README.md) for full argument reference and model compatibility table.

---

## Quick start — Phase 1 (Legacy)

```bash
pip install -r requirements.txt

# Build annotation manifest
python src/data/build_manifest.py

# Run best pipeline: SAM 2 + EfficientNet-B3
python src/patches/extract_sam2.py
python src/train.py
python src/inference/infer_sam2_effnet.py
```

---

## External files required

This repository contains **code and documentation only**. Large files must be downloaded separately:

```text
data/
└── CoralNet_Data/          # CoralNet annotation CSVs + survey images

models/
├── mobile_sam.pt           # MobileSAM weights
├── sam2.1_hiera_base_plus.pt  # SAM 2 weights
└── vlm_checkpoints/        # Fine-tuned VLM adapter (after training)
```

---

## Future phases

| Phase | Technology |
|---|---|
| **Alignment** | DPO / GRPO to penalise hallucinations and improve rare-species recall |
| **Serving** | vLLM with PagedAttention, or TensorRT-LLM + NVIDIA Triton Inference Server |
