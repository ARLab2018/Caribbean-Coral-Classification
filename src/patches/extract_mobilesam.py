"""
Full-Scale Patch Extractor: MobileSAM Edition
================================================
Generates complete patch datasets in a single pass over all images:

  SAM point prompt → polygon mask crop (background zeroed)

Both datasets are saved in PyTorch ImageFolder format, ready for training.

Usage from the repository root:
    python src/patches/extract_mobilesam.py

Default local paths:
    data/CoralNet_Data/                 CoralNet image/annotation folders
    models/mobile_sam.pt                MobileSAM weights, externally downloaded
    outputs/split_manifest.csv          image-level split manifest
    outputs/patches_option_a/           MobileSAM masked crops
    outputs/patches_option_f/           MobileSAM bbox crops

All defaults can be overridden with command-line arguments.

"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Project paths ─────────────────────────────────────────────────────────────

def find_project_root(script_path: Path) -> Path:
    """Return the repository root so the script works from any current folder."""
    for parent in [script_path.parent, *script_path.parents]:
        if (parent / "README.md").exists() or (parent / ".git").exists():
            return parent

    # Expected GitHub location: src/patches/<this_file>.py
    if script_path.parent.name == "patches" and script_path.parent.parent.name == "src":
        return script_path.parent.parent.parent

    return script_path.parent


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def repo_path(*parts: str) -> Path:
    """Build an absolute path inside this repository."""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_path(path_value) -> Path:
    """Resolve user paths. Relative paths are interpreted from the repo root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return repo_path(str(path)).resolve()

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_ROOT       = repo_path("data", "CoralNet_Data")
OUT_A           = repo_path("outputs", "patches_option_a")
OUT_F           = repo_path("outputs", "patches_option_f")
SAM_CHECKPOINT  = MODEL_DIR / "mobile_sam.pt"
SAM_MODEL_TYPE  = "vit_t"

# Reuse image-level splits so datasets are directly comparable.
BASELINE_MANIFEST = repo_path("outputs", "split_manifest.csv")

TARGET_SIZE   = 224     # final patch size for EfficientNet
BBOX_PADDING  = 0.15    # padding fraction added to bounding box (Option F)

# Augment species with fewer than this many training patches.
AUG_THRESHOLD = 500

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Resume checkpoint — tracks completed image paths.
CHECKPOINT_FILE = repo_path("outputs", "patch_extractor_mobilesam_checkpoint.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MobileSAM point-prompt coral patches for training."
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT),
                        help="CoralNet data folder containing annotations.csv files")
    parser.add_argument("--output-a", default=str(OUT_A),
                        help="Output folder for Option A masked crops")
    parser.add_argument("--output-f", default=str(OUT_F),
                        help="Output folder for Option F bbox crops")
    parser.add_argument("--sam-checkpoint", default=str(SAM_CHECKPOINT),
                        help="Path to MobileSAM weights")
    parser.add_argument("--sam-model-type", default=SAM_MODEL_TYPE,
                        help="MobileSAM model type, usually vit_t")
    parser.add_argument("--manifest", default=str(BASELINE_MANIFEST),
                        help="Optional split manifest CSV from src/data/build_manifest.py")
    parser.add_argument("--checkpoint-file", default=str(CHECKPOINT_FILE),
                        help="Resume checkpoint JSON path")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE,
                        help="Final saved patch size")
    parser.add_argument("--aug-threshold", type=int, default=AUG_THRESHOLD,
                        help="Augment train species below this count")
    return parser.parse_args()


def configure_from_args(args):
    global DATA_ROOT, OUT_A, OUT_F, SAM_CHECKPOINT, SAM_MODEL_TYPE
    global BASELINE_MANIFEST, CHECKPOINT_FILE, TARGET_SIZE, AUG_THRESHOLD, log

    DATA_ROOT = resolve_path(args.data_root)
    OUT_A = resolve_path(args.output_a)
    OUT_F = resolve_path(args.output_f)
    SAM_CHECKPOINT = resolve_path(args.sam_checkpoint)
    SAM_MODEL_TYPE = args.sam_model_type
    BASELINE_MANIFEST = resolve_path(args.manifest)
    CHECKPOINT_FILE = resolve_path(args.checkpoint_file)
    TARGET_SIZE = args.target_size
    AUG_THRESHOLD = args.aug_threshold

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"CoralNet data folder not found: {DATA_ROOT}\n"
            "Download/place the external CoralNet data under data/CoralNet_Data, "
            "or pass --data-root /path/to/CoralNet_Data."
        )

    log = setup_logging(OUT_A, "extractor")


# ── Label map ─────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "MadMir":"Madracis mirabilis",
    "AgAga":"Agaricia agaricites","BL_Aga":"Agaricia agaricites",
    "AAGA":"Agaricia agaricites","AAGA_BL":"Agaricia agaricites","AGAAGA":"Agaricia agaricites",
    "OrbAnn":"Orbicella annularis","BL_OrbAnn":"Orbicella annularis",
    "OANN":"Orbicella annularis","OANN_BL":"Orbicella annularis","ORBANN":"Orbicella annularis",
    "Lobo":"Lobophyllia spp","LOBO":"Lobophyllia spp",
    "OrbFav":"Orbicella faveolata","BL_OrbFav":"Orbicella faveolata",
    "OFAV":"Orbicella faveolata","OFAV_BL":"Orbicella faveolata","ORBFAV":"Orbicella faveolata",
    "MCav":"Montastraea cavernosa","BL_MCav":"Montastraea cavernosa",
    "MCAV":"Montastraea cavernosa","MCAV_BL":"Montastraea cavernosa",
    "PorAstr":"Porites astreoides","BL_PAst":"Porites astreoides",
    "PAST":"Porites astreoides","PAST_BL":"Porites astreoides",
    "Millepo":"Millepora spp","Mil_spp":"Millepora spp",
    "MILA":"Millepora spp","MILC":"Millepora spp","MILLE":"Millepora spp","BL_Mille":"Millepora spp",
    "SidSid":"Siderastrea siderea","BL_SidSid":"Siderastrea siderea",
    "SSID":"Siderastrea siderea","SSID_BL":"Siderastrea siderea",
    "ColNat":"Colpophyllia natans","BL_CoNat":"Colpophyllia natans",
    "CNAT":"Colpophyllia natans","CNAT_BL":"Colpophyllia natans","CNAt":"Colpophyllia natans",
    "DipStr":"Pseudodiploria strigosa","BL_DipStr":"Pseudodiploria strigosa",
    "PSTRI":"Pseudodiploria strigosa","PSTR_BL":"Pseudodiploria strigosa",
    "MAUR":"Madracis auretenra","MALC":"Madracis auretenra","MLAM":"Madracis auretenra",
    "StInt":"Stephanocoenia intersepta","BL_StInt":"Stephanocoenia intersepta",
    "SINT":"Stephanocoenia intersepta","SINT_BL":"Stephanocoenia intersepta",
    "OrbFrank":"Orbicella franksi","BL_OrbFran":"Orbicella franksi",
    "OFRA":"Orbicella franksi","OFRA_BL":"Orbicella franksi",
    "ATEN":"Acropora tenuifolia","ATEN_BL":"Acropora tenuifolia",
    "PorPor":"Porites porites","BL_PorPor":"Porites porites",
    "PPOR":"Porites porites","PPOR_BL":"Porites porites","POP":"Porites porites",
    "MeanMean":"Meandrina meandrites","BL_Mean":"Meandrina meandrites",
    "MMEA":"Meandrina meandrites","MMEA_BL":"Meandrina meandrites","MM":"Meandrina meandrites",
}

SPECIES_LIST  = sorted(set(LABEL_MAP.values()))
SPECIES_TO_IDX = {s: i for i, s in enumerate(SPECIES_LIST)}

# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(out_dir: Path, name: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(out_dir / "extraction_log.txt", mode="w")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s  %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(ch)
    return log

log = logging.getLogger("extractor")

# ── SAM ───────────────────────────────────────────────────────────────────────

def load_sam(checkpoint: Path, model_type: str):
    from x_segment_anything import sam_model_registry, SamPredictor
    import torch
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM weights not found: {checkpoint}\n"
            f"Download: wget https://github.com/ChaoningZhang/MobileSAM/"
            f"raw/master/weights/mobile_sam.pt -O {checkpoint}"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading SAM {model_type} on {device} ...")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device)
    predictor = SamPredictor(sam)
    log.info("SAM loaded.")
    return predictor


def predict_mask_and_bbox(predictor, image_rgb: np.ndarray,
                           col: int, row: int):
    """
    Run SAM with a single foreground point prompt.
    Returns (mask_bool, (x1,y1,x2,y2)) or (None, None) on failure.
    SAM convention: point_coords are (x, y) = (col, row).
    """
    try:
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[col, row]]),
            point_labels=np.array([1]),
            num_multimask_outputs=3,
        )
        best      = int(np.argmax(scores))
        mask_bool = masks[best].astype(bool)

        rows_m = np.any(mask_bool, axis=1)
        cols_m = np.any(mask_bool, axis=0)
        if not rows_m.any():
            return None, None

        y1 = int(np.argmax(rows_m))
        y2 = int(len(rows_m) - np.argmax(rows_m[::-1]) - 1)
        x1 = int(np.argmax(cols_m))
        x2 = int(len(cols_m) - np.argmax(cols_m[::-1]) - 1)
        return mask_bool, (x1, y1, x2, y2)

    except Exception as e:
        return None, None

# ── Crop functions ─────────────────────────────────────────────────────────────

def crop_option_a(img_rgb: np.ndarray,
                  mask_bool: np.ndarray,
                  bbox: tuple) -> np.ndarray | None:
    """Polygon-masked crop: bbox region with background zeroed out."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    crop      = img_rgb[y1:y2, x1:x2].copy()
    mask_crop = mask_bool[y1:y2, x1:x2]
    crop[~mask_crop] = 0
    return crop


def crop_option_f(img_rgb: np.ndarray,
                  bbox: tuple,
                  padding: float = BBOX_PADDING) -> np.ndarray | None:
    """Bounding box crop with padding — natural background preserved."""
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(w, x2 + pad_x)
    y2p = min(h, y2 + pad_y)
    if x2p <= x1p or y2p <= y1p:
        return None
    return img_rgb[y1p:y2p, x1p:x2p].copy()


def augment_patch(patch: np.ndarray) -> list[np.ndarray]:
    """Flip + rotate augmentations."""
    variants = []
    for flip_code in (0, 1):
        variants.append(cv2.flip(patch, flip_code))
    for angle in (90, 180, 270):
        h, w = patch.shape[:2]
        M    = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        variants.append(cv2.warpAffine(patch, M, (w, h)))
    return variants


def resize_and_save(patch_rgb: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resized = cv2.resize(patch_rgb, (TARGET_SIZE, TARGET_SIZE),
                         interpolation=cv2.INTER_LANCZOS4)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

# ── Data loading ──────────────────────────────────────────────────────────────

def load_annotations_with_splits() -> pd.DataFrame:
    """
    Load all annotations and assign splits using the same image-level
    assignments as the baseline extraction (loaded from split_manifest.csv).
    This ensures all three datasets train/val/test on the same images.
    """
    log.info("Loading annotations ...")
    all_anns = []

    for ann_csv in sorted(DATA_ROOT.rglob("annotations.csv")):
        source_name = ann_csv.parts[len(DATA_ROOT.parts)]
        try:
            df = pd.read_csv(ann_csv, low_memory=False)
        except Exception:
            continue
        if "Label code" not in df.columns:
            continue

        df["species"] = df["Label code"].map(LABEL_MAP)
        df_coral = df[df["species"].notna()].copy()
        if df_coral.empty:
            continue

        img_dir = ann_csv.parent / "images"
        if not img_dir.exists():
            img_dir = ann_csv.parent
        img_index = {}
        for ext in ("*.jpg","*.JPG","*.jpeg","*.JPEG","*.png","*.PNG"):
            for p in img_dir.rglob(ext):
                img_index[p.stem.lower()] = p
                img_index[p.name.lower()] = p

        for _, row in df_coral.iterrows():
            img_name = str(row["Name"]).strip()
            img_path = (img_index.get(img_name.lower()) or
                        img_index.get(Path(img_name).stem.lower()))
            if img_path is None:
                continue
            all_anns.append({
                "source":   source_name,
                "img_name": img_name,
                "img_path": str(img_path),
                "row":      int(row["Row"]),
                "col":      int(row["Column"]),
                "species":  row["species"],
            })

    ann_df = pd.DataFrame(all_anns)
    log.info(f"Total coral annotations: {len(ann_df):,} | "
             f"Unique images: {ann_df['img_path'].nunique():,}")

    # Load split assignments from baseline manifest
    if BASELINE_MANIFEST.exists():
        log.info(f"Loading split assignments from {BASELINE_MANIFEST} ...")
        manifest = pd.read_csv(BASELINE_MANIFEST)
        # Build img_name → split map
        split_map = (manifest.groupby("img_name")["split"]
                     .first().to_dict())
        ann_df["split"] = ann_df["img_name"].map(split_map)
        n_missing = ann_df["split"].isna().sum()
        if n_missing > 0:
            log.warning(f"  {n_missing} annotations have no split assignment "
                        f"(new images not in baseline) — assigning to train")
            ann_df["split"] = ann_df["split"].fillna("train")
    else:
        log.warning("Baseline manifest not found — assigning splits fresh")
        split_map = {}
        for source, grp in ann_df.groupby("source"):
            imgs   = grp["img_path"].unique().tolist()
            random.shuffle(imgs)
            n      = len(imgs)
            n_test = max(1, int(n * 0.15))
            n_val  = max(1, int(n * 0.15))
            for img in imgs[:n_test]:
                split_map[img] = "test"
            for img in imgs[n_test:n_test+n_val]:
                split_map[img] = "val"
            for img in imgs[n_test+n_val:]:
                split_map[img] = "train"
        ann_df["split"] = ann_df["img_path"].map(split_map)

    for split in ("train","val","test"):
        n = (ann_df["split"] == split).sum()
        log.info(f"  {split}: {n:,} annotations")

    return ann_df

# ── Main extraction ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    configure_from_args(args)

    for out_dir in (OUT_A, OUT_F):
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write label map to both output dirs
        with open(out_dir / "label_map.json", "w") as f:
            json.dump(SPECIES_TO_IDX, f, indent=2)

    ann_df   = load_annotations_with_splits()
    predictor = load_sam(SAM_CHECKPOINT, SAM_MODEL_TYPE)

    # Identify thin species for augmentation
    train_counts  = (ann_df[ann_df["split"] == "train"]
                     ["species"].value_counts().to_dict())
    thin_species  = {s for s, c in train_counts.items() if c < AUG_THRESHOLD}
    log.info(f"Species below aug threshold: {sorted(thin_species)}")

    # Counters
    counters_a = defaultdict(lambda: defaultdict(int))
    counters_f = defaultdict(lambda: defaultdict(int))
    aug_a = defaultdict(int)
    aug_f = defaultdict(int)
    sam_fail = edge_fail_a = edge_fail_f = open_fail = 0

    # ── Load resume checkpoint ────────────────────────────────────────────
    completed_images = set()
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            checkpoint_data = json.load(f)
            completed_images = set(checkpoint_data.get("completed", []))
        log.info(f"Resuming: {len(completed_images):,} images already processed")
    else:
        checkpoint_data = {"completed": []}

    all_images = list(ann_df.groupby("img_path"))
    remaining  = [(p, g) for p, g in all_images if p not in completed_images]
    log.info(f"Images to process: {len(remaining):,} "
             f"(skipping {len(completed_images):,} already done)")

    log.info("\nExtracting patches ...")
    SAVE_CHECKPOINT_EVERY = 100   # save progress every N images

    for img_idx, (img_path_str, group) in enumerate(tqdm(
            remaining,
            total=len(remaining),
            desc="Images")):

        try:
            img_bgr = cv2.imread(img_path_str)
            if img_bgr is None:
                raise ValueError("imread returned None")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            open_fail += len(group)
            # Still mark as completed so we don't retry broken files
            completed_images.add(img_path_str)
            continue

        for _, ann in group.iterrows():
            row, col = int(ann["row"]), int(ann["col"])
            species  = ann["species"]
            split    = ann["split"]
            sp_safe  = species.replace(" ","_").replace("/","-")
            stem     = f"{Path(img_path_str).stem}_r{row}_c{col}"

            # Run SAM once per annotation point
            mask_bool, bbox = predict_mask_and_bbox(
                predictor, img_rgb, col, row
            )

            if mask_bool is None:
                sam_fail += 1
                continue

            # ── Option A ──────────────────────────────────────────────────
            patch_a = crop_option_a(img_rgb, mask_bool, bbox)
            if patch_a is None:
                edge_fail_a += 1
            else:
                out_path = OUT_A / split / sp_safe / f"{stem}.jpg"
                resize_and_save(patch_a, out_path)
                counters_a[species][split] += 1

                if split == "train" and species in thin_species:
                    for i, aug in enumerate(augment_patch(patch_a)):
                        resize_and_save(aug, OUT_A / split / sp_safe /
                                        f"{stem}_aug{i}.jpg")
                        aug_a[species] += 1

            # ── Option F ──────────────────────────────────────────────────
            patch_f = crop_option_f(img_rgb, bbox)
            if patch_f is None:
                edge_fail_f += 1
            else:
                out_path = OUT_F / split / sp_safe / f"{stem}.jpg"
                resize_and_save(patch_f, out_path)
                counters_f[species][split] += 1

                if split == "train" and species in thin_species:
                    for i, aug in enumerate(augment_patch(patch_f)):
                        resize_and_save(aug, OUT_F / split / sp_safe /
                                        f"{stem}_aug{i}.jpg")
                        aug_f[species] += 1

        # Mark image as completed and periodically save checkpoint
        completed_images.add(img_path_str)
        if (img_idx + 1) % SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_data["completed"] = list(completed_images)
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(checkpoint_data, f)

    # Save final checkpoint
    checkpoint_data["completed"] = list(completed_images)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint_data, f)
    log.info(f"Checkpoint saved: {CHECKPOINT_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────
    for label, counters, aug_c, out_dir in [
            ("OPTION A", counters_a, aug_a, OUT_A),
            ("OPTION F", counters_f, aug_f, OUT_F)]:

        log.info(f"\n{'='*68}")
        log.info(f"{label}  ({out_dir})")
        log.info(f"{'='*68}")
        log.info(f"{'SPECIES':<35} {'TRAIN':>7} {'VAL':>7} "
                 f"{'TEST':>7} {'AUG':>7}")
        log.info(f"{'='*68}")
        tot_tr = tot_va = tot_te = tot_au = 0
        for sp in sorted(SPECIES_LIST):
            tr = counters[sp]["train"]
            va = counters[sp]["val"]
            te = counters[sp]["test"]
            au = aug_c[sp]
            log.info(f"{sp:<35} {tr:>7} {va:>7} {te:>7} {au:>7}")
            tot_tr+=tr; tot_va+=va; tot_te+=te; tot_au+=au
        log.info(f"{'='*68}")
        log.info(f"{'TOTAL':<35} {tot_tr:>7} {tot_va:>7} "
                 f"{tot_te:>7} {tot_au:>7}")

    log.info(f"\nSAM failed:      {sam_fail:,}")
    log.info(f"Edge fail (A):   {edge_fail_a:,}")
    log.info(f"Edge fail (F):   {edge_fail_f:,}")
    log.info(f"Open failed:     {open_fail:,}")
    log.info(f"\nOption A patches: {OUT_A}")
    log.info(f"Option F patches: {OUT_F}")


if __name__ == "__main__":
    main()
