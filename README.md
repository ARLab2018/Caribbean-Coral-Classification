# Caribbean Coral Reef Species Classification

End-to-end deep learning pipeline for detecting and classifying **17 Caribbean coral species** from underwater photoquadrat images. Combines **SAM 2** (Segment Anything Model 2) for instance-level coral colony segmentation with a fine-tuned **EfficientNet-B3** classifier.

Independent study project — full report in [`docs/report.pdf`](docs/report.pdf).

---

## Results

Four pipelines were trained and evaluated on the same image-level held-out test set (9,879 patches across 17 species).

| # | Segmenter | Classifier | Patches | Accuracy | Macro-F1 |
|---|-----------|------------|---------|----------|----------|
| 1 | MobileSAM | EfficientNet-B3 | 49,850 | 70.1% | 0.606 |
| 2 | CoralSCOP | EfficientNet-B3 | — | failed* | failed* |
| 3 | **SAM 2** | **EfficientNet-B3** | **43,672** | **72.6%** | **0.624** ← best |
| 4 | SAM 2 | DINOv2-S | 43,672 | 46.8% | 0.376 |

\* CoralSCOP frozen-encoder fine-tuning did not converge to a usable classifier; details in the report.

Switching the segmenter from MobileSAM to SAM 2 improved macro-F1 by **+0.018** despite training on roughly 6,000 fewer patches — cleaner masks more than offset the smaller dataset. DINOv2-S underperformed because zeroed-background masked crops are out-of-distribution for its self-supervised pretraining.

### Per-species highlights

| Species | F1 | Test n |
|---|---|---|
| *Madracis mirabilis* | 0.882 | 1,479 |
| *Orbicella annularis* | 0.794 | 1,385 |
| *Siderastrea siderea* | 0.774 | 408 |
| *Agaricia agaricites* | 0.768 | 1,371 |
| ... | ... | ... |
| *Porites porites* | 0.431 | 111 |
| *Orbicella franksi* | 0.200 | 90 |

Full per-species table in the [report](docs/report.pdf).

---

## Repository layout

```
src/
├── data/build_manifest.py          # Build split_manifest.csv from CoralNet
├── patches/
│   ├── extract_mobilesam.py        # Pipeline 1: MobileSAM masked crops
│   ├── extract_coralscop.py        # Pipeline 2: CoralSCOP masked crops
│   └── extract_sam2.py             # Pipelines 3 & 4: SAM2 point-prompt crops
├── train.py                        # Trains EfficientNet-B3 or DINOv2-S
└── inference/
    ├── infer_mobilesam_effnet.py
    ├── infer_coralscop_effnet.py
    ├── infer_sam2_effnet.py        # Best model
    └── infer_sam2_dinov2.py
tools/
└── label_tool.py                   # Tkinter GUI for manual labelling
docs/
├── report.pdf                      # Full independent study report
└── report.tex                      # LaTeX source
examples/
├── sample_input.jpg
└── sample_output_overlay.jpg
```

---

## External assets

The repository contains code only. Trained model weights, source images, and patches are too large to host on GitHub.

| Asset | Size | Location |
|---|---|---|
| Trained model checkpoints (4 pipelines) | ~600 MB | [Google Drive](TODO: paste link here) |
| Source images from 8 CoralNet sources | ~12 GB | [Google Drive](TODO: paste link here) |
| Generated training patches (SAM 2) | ~5 GB | [Google Drive](TODO: paste link here) |
| SAM 2 base weights | 308 MB | [Meta AI](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt) |
| MobileSAM weights | 38 MB | [MobileSAM repo](https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt) |
| CoralSCOP weights | 350 MB | [CoralSCOP project](https://coralscop.hkustvgd.com) |

---

## Installation

Tested on Ubuntu 22.04 with Python 3.10, CUDA 12.x, and an NVIDIA RTX 4060 (8 GB).

```bash
# 1. Clone this repo
git clone https://github.com/YOUR-USERNAME/caribbean-coral-classification.git
cd caribbean-coral-classification

# 2. Create environment
conda create -n coral python=3.10
conda activate coral

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install SAM 2 (required for pipelines 3 & 4)
git clone https://github.com/facebookresearch/sam2.git external/sam2
cd external/sam2 && pip install -e . && cd ../..

# 5. Download segmentation weights
mkdir -p weights
wget -P weights https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget -P weights https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# 6. Download trained classifier weights from Google Drive
# (link above) → place in checkpoints/
```

---

## Quickstart — inference on your own image

```bash
python src/inference/infer_sam2_effnet.py \
    --input path/to/reef_photo.jpg \
    --output ./inference_output \
    --sam2_weights weights/sam2.1_hiera_base_plus.pt \
    --classifier checkpoints/sam2_effnet_best.pt
```

Outputs per image:
- `<name>_overlay.jpg` — semi-transparent species colour overlay
- `<name>_species_map.jpg` — solid colour species map
- `<name>_label_map.png` — integer label map (0 = background, 1–17 = species)
- `<name>_results.json` — per-colony detections with confidence and bbox
- `summary.csv` — collated across all processed images

---

## Reproducing the experiments

### 1. Download and prepare CoralNet data

```bash
# Install CoralNet-Toolbox separately and use it to download these 8 sources:
#   - Curaçao Coral Reef Assessment 2023
#   - Curaçao Reef Assessment 2025
#   - STINAPA GCRMN
#   - MarineGEO RLS
#   - LH CBC 2019-2022
#   - Little Cayman MarineGEO
#   - FSU_STINAPA
#   - St. John 2016

# Build the unified manifest
python src/data/build_manifest.py
```

### 2. Generate training patches

```bash
# Pipeline 3 (recommended): SAM 2 point-prompt extraction (~8-10 hours on RTX 4060)
python src/patches/extract_sam2.py

# Or compare with the baselines:
python src/patches/extract_mobilesam.py
python src/patches/extract_coralscop.py
```

### 3. Train

```bash
# Best pipeline
python src/train.py --dataset sam2 --backbone efficientnet_b3

# DINOv2 ablation
python src/train.py --dataset sam2 --backbone dinov2_s
```

Training takes ~6 hours on an RTX 4060 with early stopping (patience 12).

---

## Pipeline overview

```
Image
  ↓
SAM 2 automatic mask generator    ← instance segmentation
  ↓
Per-colony masks + bboxes
  ↓
Masked crop (zero background)     ← Option A style crop
  ↓
EfficientNet-B3 classifier        ← species identification
  ↓
Species label + confidence per colony
  ↓
Not-coral rejection (conf < 0.45) ← filter substrate false positives
  ↓
Output: species map + JSON
```

For training, the same SAM 2 model is used with **point prompts** (one prompt per CoralNet annotation point) rather than automatic mask generation. This guarantees each patch has a verified species label.

---

## Limitations

The model is competent on common species but struggles on rare ones. Known failure modes:

- **Low-support species** (*Orbicella franksi*, *Meandrina meandrites*, *Porites porites*) — F1 below 0.45, partly from low annotation counts, partly from confusion with congeners.
- **Colony fragmentation** — large encrusting colonies sometimes split into multiple sub-masks.
- **False positives on substrate** — the not-coral confidence threshold catches most, not all.
- **DINOv2 backbone fails on Option A crops** — distribution shift from zeroed background; would likely need natural-background crops to work.

See the report's Discussion section for detailed analysis and proposed mitigations.

---

## Citation

If you use this code or pipeline, please cite:

```bibtex
@misc{mitanshu2026coral,
  author       = {Mitanshu},
  title        = {Automated Species-Level Classification of Caribbean Coral Reefs:
                  A SAM 2 Segmentation and EfficientNet-B3 Pipeline},
  year         = {2026},
  howpublished = {Independent study report},
  url          = {https://github.com/YOUR-USERNAME/caribbean-coral-classification}
}
```

Key upstream references (full list in the report):

- Ravi, N., et al. (2024). *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714
- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML 2019
- Zheng, Z., et al. (2024). *CoralSCOP: Segment any Coral Image on this Planet.* CVPR 2024
- Wang, J., et al. (2025). *Multi-dataset-integrated Coral-Lab segmentation.* IJAEOG 143:104819
- Beijbom, O., et al. (2015). *Towards Automated Annotation of Benthic Survey Images.* PLOS ONE

---

## Licence

- **Code** in this repository: MIT (see [LICENSE](LICENSE)).
- **CoralNet annotations** used for training: CC BY-NC 4.0 per CoralNet's terms.
- **Trained model weights**: CC BY-NC 4.0, consistent with the training data licence.

Commercial use of the trained models is not permitted by the underlying data licence. Commercial use of the *code itself* is permitted under MIT.

---

## Acknowledgements

CoralNet data from eight publicly accessible sources. CoralNet-Toolbox by Jordan Pierce (NOAA). SAM 2 weights and code from Meta AI Research. This work was conducted as an independent study; see report for full acknowledgements.
