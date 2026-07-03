"""
Caribbean Coral Species Patch Extractor v2
==========================================
Extracts 300x300px image patches from CoralNet-Toolbox downloaded images,
using species-level annotations from 8 Caribbean sources.

17 target species:
    Madracis mirabilis, Agaricia agaricites, Orbicella annularis,
    Lobophyllia spp, Orbicella faveolata, Montastraea cavernosa,
    Porites astreoides, Millepora spp, Siderastrea siderea,
    Colpophyllia natans, Pseudodiploria strigosa, Madracis auretenra,
    Stephanocoenia intersepta, Orbicella franksi, Acropora tenuifolia,
    Porites porites, Meandrina meandrites

Input layout (CoralNet-Toolbox download structure):
    ~/Independent_study/CoralNet_Data/
        <Source Name>/
            <Source Name>\<ID>/
                annotations.csv
                images/          ← downloaded images land here
                    img001.jpg
                    ...

Output:
    ~/Independent_study/patches/
        train/<species>/patch_xxx.jpg
        val/<species>/patch_xxx.jpg
        test/<species>/patch_xxx.jpg
    ~/Independent_study/patches/extraction_log.txt
    ~/Independent_study/patches/split_manifest.csv

Usage:
    python patch_extractor_v2.py
"""

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_ROOT   = Path.home() / "Independent_study" / "CoralNet_Data"
PATCHES_DIR = Path.home() / "Independent_study" / "patches"

PATCH_SIZE  = 300     # pixels, square crop centred on annotation point
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15    # must sum to 1.0

# Augment species with fewer than this many training patches
AUG_THRESHOLD = 500
AUG_FACTOR    = 3     # how many augmented copies per original for thin classes

SEED = 42

# ── Label map: every CoralNet code → canonical species name ──────────────────
# Covers all 8 sources. Codes NOT in this map are ignored (substrate, algae etc)

LABEL_MAP = {
    # ── Madracis mirabilis ──────────────────────────────────────────────────
    "MadMir":           "Madracis mirabilis",

    # ── Agaricia agaricites ─────────────────────────────────────────────────
    "AgAga":            "Agaricia agaricites",
    "BL_Aga":           "Agaricia agaricites",   # bleached — same morphology
    "AAGA":             "Agaricia agaricites",
    "AAGA_BL":          "Agaricia agaricites",
    "AGAAGA":           "Agaricia agaricites",

    # ── Orbicella annularis ─────────────────────────────────────────────────
    "OrbAnn":           "Orbicella annularis",
    "BL_OrbAnn":        "Orbicella annularis",
    "OANN":             "Orbicella annularis",
    "OANN_BL":          "Orbicella annularis",
    "ORBANN":           "Orbicella annularis",

    # ── Lobophyllia spp ─────────────────────────────────────────────────────
    "Lobo":             "Lobophyllia spp",
    "LOBO":             "Lobophyllia spp",

    # ── Orbicella faveolata ─────────────────────────────────────────────────
    "OrbFav":           "Orbicella faveolata",
    "BL_OrbFav":        "Orbicella faveolata",
    "OFAV":             "Orbicella faveolata",
    "OFAV_BL":          "Orbicella faveolata",
    "ORBFAV":           "Orbicella faveolata",

    # ── Montastraea cavernosa ───────────────────────────────────────────────
    "MCav":             "Montastraea cavernosa",
    "BL_MCav":          "Montastraea cavernosa",
    "MCAV":             "Montastraea cavernosa",
    "MCAV_BL":          "Montastraea cavernosa",

    # ── Porites astreoides ──────────────────────────────────────────────────
    "PorAstr":          "Porites astreoides",
    "BL_PAst":          "Porites astreoides",
    "PAST":             "Porites astreoides",
    "PAST_BL":          "Porites astreoides",

    # ── Millepora spp (all species merged) ──────────────────────────────────
    "Millepo":          "Millepora spp",
    "Mil_spp":          "Millepora spp",
    "MILA":             "Millepora spp",
    "MILC":             "Millepora spp",
    "MILLE":            "Millepora spp",
    "BL_Mille":         "Millepora spp",

    # ── Siderastrea siderea ─────────────────────────────────────────────────
    "SidSid":           "Siderastrea siderea",
    "BL_SidSid":        "Siderastrea siderea",
    "SSID":             "Siderastrea siderea",
    "SSID_BL":          "Siderastrea siderea",

    # ── Colpophyllia natans ─────────────────────────────────────────────────
    "ColNat":           "Colpophyllia natans",
    "BL_CoNat":         "Colpophyllia natans",
    "CNAT":             "Colpophyllia natans",
    "CNAT_BL":          "Colpophyllia natans",
    "CNAt":             "Colpophyllia natans",

    # ── Pseudodiploria strigosa ─────────────────────────────────────────────
    "DipStr":           "Pseudodiploria strigosa",
    "BL_DipStr":        "Pseudodiploria strigosa",
    "PSTRI":            "Pseudodiploria strigosa",
    "PSTR_BL":          "Pseudodiploria strigosa",

    # ── Madracis auretenra ──────────────────────────────────────────────────
    "MAUR":             "Madracis auretenra",
    "MALC":             "Madracis auretenra",
    "MLAM":             "Madracis auretenra",

    # ── Stephanocoenia intersepta ───────────────────────────────────────────
    "StInt":            "Stephanocoenia intersepta",
    "BL_StInt":         "Stephanocoenia intersepta",
    "SINT":             "Stephanocoenia intersepta",
    "SINT_BL":          "Stephanocoenia intersepta",

    # ── Orbicella franksi ───────────────────────────────────────────────────
    "OrbFrank":         "Orbicella franksi",
    "BL_OrbFran":       "Orbicella franksi",
    "OFRA":             "Orbicella franksi",
    "OFRA_BL":          "Orbicella franksi",

    # ── Acropora tenuifolia ─────────────────────────────────────────────────
    "ATEN":             "Acropora tenuifolia",
    "ATEN_BL":          "Acropora tenuifolia",

    # ── Porites porites ─────────────────────────────────────────────────────
    "PorPor":           "Porites porites",
    "BL_PorPor":        "Porites porites",
    "PPOR":             "Porites porites",
    "PPOR_BL":          "Porites porites",
    "POP":              "Porites porites",

    # ── Meandrina meandrites ────────────────────────────────────────────────
    "MeanMean":         "Meandrina meandrites",
    "BL_Mean":          "Meandrina meandrites",
    "MMEA":             "Meandrina meandrites",
    "MMEA_BL":          "Meandrina meandrites",
    "MM":               "Meandrina meandrites",
}

SPECIES_LIST = sorted(set(LABEL_MAP.values()))
SPECIES_TO_IDX = {s: i for i, s in enumerate(SPECIES_LIST)}

# ── Logging ───────────────────────────────────────────────────────────────────

PATCHES_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PATCHES_DIR / "extraction_log.txt",
                            mode="w", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

random.seed(SEED)

def find_annotations_and_images(data_root: Path):
    """
    Walk data_root and find all annotations.csv files.
    Returns list of (source_name, annotations_csv_path, images_dir).
    Images may be in an 'images' subfolder or alongside the CSV.
    """
    sources = []
    for ann_csv in sorted(data_root.rglob("annotations.csv")):
        source_name = ann_csv.parts[len(data_root.parts)]
        # Images folder: try 'images/' sibling first, then parent
        img_dir = ann_csv.parent / "images"
        if not img_dir.exists():
            img_dir = ann_csv.parent
        sources.append((source_name, ann_csv, img_dir))
        log.info(f"  Found source: {source_name}")
        log.info(f"    Annotations: {ann_csv}")
        log.info(f"    Images dir:  {img_dir}")
    return sources


def index_images(img_dir: Path) -> dict[str, Path]:
    """
    Build stem → path index for all images in img_dir (recursive).
    Handles jpg/JPG/jpeg/png extensions.
    """
    index = {}
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        for p in img_dir.rglob(ext):
            index[p.stem.lower()] = p
            index[p.name.lower()] = p  # also index by full filename
    return index


def assign_splits(image_names: list[str]) -> dict[str, str]:
    """Assign train/val/test splits at image level to prevent spatial leakage."""
    names = sorted(image_names)
    random.shuffle(names)
    n = len(names)
    n_test  = max(1, int(n * TEST_RATIO))
    n_val   = max(1, int(n * VAL_RATIO))
    splits  = {}
    for name in names[:n_test]:
        splits[name] = "test"
    for name in names[n_test:n_test + n_val]:
        splits[name] = "val"
    for name in names[n_test + n_val:]:
        splits[name] = "train"
    return splits


def crop_patch(img: Image.Image, row: int, col: int,
               half: int) -> Image.Image | None:
    """
    Crop a square patch of size (2*half) x (2*half) centred on (row, col).
    Returns None if the crop would go out of bounds.
    row = y coordinate, col = x coordinate (CoralNet convention).
    """
    w, h = img.size
    x1, y1 = col - half, row - half
    x2, y2 = col + half, row + half
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None
    return img.crop((x1, y1, x2, y2))


def augment_patch(patch: Image.Image) -> list[Image.Image]:
    """Return augmented variants: flips + 90° rotations."""
    variants = []
    for flip in (Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM):
        variants.append(patch.transpose(flip))
    for angle in (90, 180, 270):
        variants.append(patch.rotate(angle))
    return variants


def save_patch(patch: Image.Image, out_dir: Path,
               stem: str, suffix: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{stem}{suffix}.jpg"
    out_path = out_dir / fname
    patch.convert("RGB").save(out_path, "JPEG", quality=90)
    return out_path

# ── Main extraction ────────────────────────────────────────────────────────────

def main():
    log.info("Caribbean Coral Species Patch Extractor v2")
    log.info(f"Data root:   {DATA_ROOT}")
    log.info(f"Patches dir: {PATCHES_DIR}")
    log.info(f"Target species ({len(SPECIES_LIST)}): {SPECIES_LIST}")

    # ── 1. Discover all sources ────────────────────────────────────────────
    log.info("\nDiscovering sources ...")
    sources = find_annotations_and_images(DATA_ROOT)
    if not sources:
        log.error("No annotations.csv files found. Check DATA_ROOT path.")
        return

    # ── 2. Load and filter all annotations ────────────────────────────────
    log.info("\nLoading annotations ...")
    all_annotations = []   # list of dicts

    for source_name, ann_csv, img_dir in sources:
        try:
            df = pd.read_csv(ann_csv, low_memory=False)
        except Exception as e:
            log.error(f"  Could not read {ann_csv}: {e}")
            continue

        # Normalise column names (CoralNet uses 'Label code')
        if "Label code" not in df.columns:
            log.warning(f"  {source_name}: no 'Label code' column — skipping")
            continue

        # Filter to target species only
        df["species"] = df["Label code"].map(LABEL_MAP)
        df_coral = df[df["species"].notna()].copy()

        if df_coral.empty:
            log.warning(f"  {source_name}: no target species found")
            continue

        img_index = index_images(img_dir)

        kept = skipped_notfound = 0
        for _, row in df_coral.iterrows():
            img_name  = str(row["Name"]).strip()
            img_stem  = Path(img_name).stem.lower()
            img_path  = img_index.get(img_name.lower()) or img_index.get(img_stem)

            if img_path is None:
                skipped_notfound += 1
                continue

            try:
                ann_row = int(row["Row"])
                ann_col = int(row["Column"])
            except (ValueError, KeyError):
                continue

            all_annotations.append({
                "source":    source_name,
                "img_name":  img_name,
                "img_path":  str(img_path),
                "row":       ann_row,
                "col":       ann_col,
                "species":   row["species"],
            })
            kept += 1

        log.info(f"  {source_name}: {kept} coral annotations "
                 f"({skipped_notfound} skipped — image not found)")

    if not all_annotations:
        log.error("No annotations loaded. Check image paths.")
        return

    ann_df = pd.DataFrame(all_annotations)
    log.info(f"\nTotal coral annotations: {len(ann_df):,}")
    log.info(f"Unique images: {ann_df['img_name'].nunique():,}")
    log.info("\nPer-species counts:")
    for species, count in ann_df["species"].value_counts().items():
        log.info(f"  {species:<35} {count:>6}")

    # ── 3. Image-level train/val/test splits ───────────────────────────────
    log.info("\nAssigning image-level splits ...")

    # Split per source to avoid cross-source leakage dominating splits
    split_map = {}
    for source_name, group in ann_df.groupby("source"):
        images_in_source = group["img_name"].unique().tolist()
        src_splits = assign_splits(images_in_source)
        split_map.update(src_splits)

    ann_df["split"] = ann_df["img_name"].map(split_map)

    for split in ("train", "val", "test"):
        n = (ann_df["split"] == split).sum()
        log.info(f"  {split}: {n:,} annotations")

    # ── 4. Identify thin species needing augmentation ──────────────────────
    train_counts = (ann_df[ann_df["split"] == "train"]
                    ["species"].value_counts().to_dict())
    thin_species = {s for s, c in train_counts.items() if c < AUG_THRESHOLD}
    if thin_species:
        log.info(f"\nSpecies below aug threshold ({AUG_THRESHOLD}): "
                 f"{sorted(thin_species)}")

    # ── 5. Extract patches ─────────────────────────────────────────────────
    log.info("\nExtracting patches ...")
    half = PATCH_SIZE // 2

    counters = defaultdict(lambda: defaultdict(int))  # species → split → count
    aug_counters = defaultdict(int)
    skipped_edge = skipped_open = 0

    manifest_rows = []

    # Group annotations by image path so each image is opened exactly once
    # and immediately closed — no unbounded cache, O(1) memory per image
    grouped = ann_df.groupby("img_path")
    total_images = len(grouped)

    for img_idx, (img_path_str, group) in enumerate(
            tqdm(grouped, total=total_images, desc="Images")):

        try:
            img = Image.open(img_path_str)
            img.load()   # force full decode into memory
        except Exception as e:
            skipped_open += len(group)
            continue

        for _, ann in group.iterrows():
            species = ann["species"]
            split   = ann["split"]
            row, col = int(ann["row"]), int(ann["col"])

            patch = crop_patch(img, row, col, half)
            if patch is None:
                skipped_edge += 1
                continue

            # Save patch
            species_safe = species.replace(" ", "_").replace("/", "-")
            out_dir  = PATCHES_DIR / split / species_safe
            stem     = f"{Path(img_path_str).stem}_r{row}_c{col}"
            out_path = save_patch(patch, out_dir, stem)

            counters[species][split] += 1
            manifest_rows.append({
                "patch_path": str(out_path),
                "species":    species,
                "split":      split,
                "source":     ann["source"],
                "img_name":   ann["img_name"],
                "row":        row,
                "col":        col,
            })

            # Augmentation for thin training classes only
            if split == "train" and species in thin_species:
                for i, aug_patch in enumerate(augment_patch(patch)):
                    aug_path = save_patch(aug_patch, out_dir, stem, f"_aug{i}")
                    aug_counters[species] += 1
                    manifest_rows.append({
                        "patch_path": str(aug_path),
                        "species":    species,
                        "split":      "train",
                        "source":     ann["source"],
                        "img_name":   ann["img_name"],
                        "row":        row,
                        "col":        col,
                    })

        img.close()  # release immediately after all patches from this image

    # ── 6. Save manifest ───────────────────────────────────────────────────
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(PATCHES_DIR / "split_manifest.csv", index=False)

    # ── 7. Save label map ─────────────────────────────────────────────────
    import json
    with open(PATCHES_DIR / "label_map.json", "w") as f:
        json.dump(SPECIES_TO_IDX, f, indent=2)

    # ── 8. Summary report ─────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info(f"{'SPECIES':<35} {'TRAIN':>7} {'VAL':>7} {'TEST':>7} {'AUG':>7}")
    log.info("=" * 70)

    total_train = total_val = total_test = total_aug = 0
    for species in sorted(SPECIES_LIST):
        tr = counters[species]["train"]
        va = counters[species]["val"]
        te = counters[species]["test"]
        au = aug_counters[species]
        log.info(f"{species:<35} {tr:>7} {va:>7} {te:>7} {au:>7}")
        total_train += tr
        total_val   += va
        total_test  += te
        total_aug   += au

    log.info("=" * 70)
    log.info(f"{'TOTAL':<35} {total_train:>7} {total_val:>7} "
             f"{total_test:>7} {total_aug:>7}")
    log.info(f"\nSkipped (edge):        {skipped_edge:,}")
    log.info(f"Skipped (open failed): {skipped_open:,}")
    log.info(f"\nPatches saved to: {PATCHES_DIR}")
    log.info(f"Manifest saved to: {PATCHES_DIR / 'split_manifest.csv'}")
    log.info(f"Label map saved to: {PATCHES_DIR / 'label_map.json'}")


if __name__ == "__main__":
    main()
