"""
VLM Dataset Builder for Caribbean Coral Classification
=======================================================
Reads CoralNet annotation CSVs (same sources as patch_extractor_v2.py),
crops natural-background bounding-box patches around each annotation point,
and writes conversational JSONL files for SFT with a VLM.

Output layout
─────────────
vlm_crops/
  train/
    Madracis_mirabilis/
      18PalmsT200002_r512_c768.jpg
      ...
  val/   ...
  test/  ...
  train.jsonl
  val.jsonl
  test.jsonl
  dataset_stats.json

Each JSONL line (conversation format for SFTTrainer):
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "<abs_path_to_crop>"},
          {"type": "text",  "text": "What coral species is in this underwater photo? Answer with only the scientific species name."}
        ]
      },
      {"role": "assistant", "content": "Madracis mirabilis"}
    ]
  }

Class imbalance handling
────────────────────────
Species with fewer than OVERSAMPLE_THRESHOLD training crops get their JSONL
entries repeated OVERSAMPLE_FACTOR times.  This happens purely in the JSONL
file — no duplicate images are written to disk, avoiding storage bloat.

Usage
─────
  python vlm/build_vlm_dataset.py
  python vlm/build_vlm_dataset.py --patch_size 300 --oversample_threshold 500
  python vlm/build_vlm_dataset.py --dry_run          # stats only, nothing written
  python vlm/build_vlm_dataset.py --manifest existing_split_manifest.csv
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Defaults ──────────────────────────────────────────────────────────────────

DATA_ROOT            = Path.home() / "Independent_study" / "CoralNet_Data"
OUTPUT_DIR           = Path.home() / "Independent_study" / "vlm_crops"
DEFAULT_PATCH_SIZE   = 300      # square crop side in pixels
DEFAULT_TRAIN_RATIO  = 0.70
DEFAULT_VAL_RATIO    = 0.15
DEFAULT_SEED         = 42
OVERSAMPLE_THRESHOLD = 500      # species below this → oversampled in JSONL
OVERSAMPLE_FACTOR    = 4        # repeat factor for thin species

USER_PROMPT = (
    "What coral species is in this underwater photo? "
    "Answer with only the scientific species name."
)

# ── Label map (identical to patch_extractor_v2.py) ────────────────────────────

LABEL_MAP = {
    # Madracis mirabilis
    "MadMir":           "Madracis mirabilis",

    # Agaricia agaricites
    "AgAga":            "Agaricia agaricites",
    "BL_Aga":           "Agaricia agaricites",
    "AAGA":             "Agaricia agaricites",
    "AAGA_BL":          "Agaricia agaricites",
    "AGAAGA":           "Agaricia agaricites",

    # Orbicella annularis
    "OrbAnn":           "Orbicella annularis",
    "BL_OrbAnn":        "Orbicella annularis",
    "OANN":             "Orbicella annularis",
    "OANN_BL":          "Orbicella annularis",
    "ORBANN":           "Orbicella annularis",

    # Lobophyllia spp
    "Lobo":             "Lobophyllia spp",
    "LOBO":             "Lobophyllia spp",

    # Orbicella faveolata
    "OrbFav":           "Orbicella faveolata",
    "BL_OrbFav":        "Orbicella faveolata",
    "OFAV":             "Orbicella faveolata",
    "OFAV_BL":          "Orbicella faveolata",
    "ORBFAV":           "Orbicella faveolata",

    # Montastraea cavernosa
    "MCav":             "Montastraea cavernosa",
    "BL_MCav":          "Montastraea cavernosa",
    "MCAV":             "Montastraea cavernosa",
    "MCAV_BL":          "Montastraea cavernosa",

    # Porites astreoides
    "PorAstr":          "Porites astreoides",
    "BL_PAst":          "Porites astreoides",
    "PAST":             "Porites astreoides",
    "PAST_BL":          "Porites astreoides",

    # Millepora spp
    "Millepo":          "Millepora spp",
    "Mil_spp":          "Millepora spp",
    "MILA":             "Millepora spp",
    "MILC":             "Millepora spp",
    "MILLE":            "Millepora spp",
    "BL_Mille":         "Millepora spp",

    # Siderastrea siderea
    "SidSid":           "Siderastrea siderea",
    "BL_SidSid":        "Siderastrea siderea",
    "SSID":             "Siderastrea siderea",
    "SSID_BL":          "Siderastrea siderea",

    # Colpophyllia natans
    "ColNat":           "Colpophyllia natans",
    "BL_CoNat":         "Colpophyllia natans",
    "CNAT":             "Colpophyllia natans",
    "CNAT_BL":          "Colpophyllia natans",
    "CNAt":             "Colpophyllia natans",

    # Pseudodiploria strigosa
    "DipStr":           "Pseudodiploria strigosa",
    "BL_DipStr":        "Pseudodiploria strigosa",
    "PSTRI":            "Pseudodiploria strigosa",
    "PSTR_BL":          "Pseudodiploria strigosa",

    # Madracis auretenra
    "MAUR":             "Madracis auretenra",
    "MALC":             "Madracis auretenra",
    "MLAM":             "Madracis auretenra",

    # Stephanocoenia intersepta
    "StInt":            "Stephanocoenia intersepta",
    "BL_StInt":         "Stephanocoenia intersepta",
    "SINT":             "Stephanocoenia intersepta",
    "SINT_BL":          "Stephanocoenia intersepta",

    # Orbicella franksi
    "OrbFrank":         "Orbicella franksi",
    "BL_OrbFran":       "Orbicella franksi",
    "OFRA":             "Orbicella franksi",
    "OFRA_BL":          "Orbicella franksi",

    # Acropora tenuifolia
    "ATEN":             "Acropora tenuifolia",
    "ATEN_BL":          "Acropora tenuifolia",

    # Porites porites
    "PorPor":           "Porites porites",
    "BL_PorPor":        "Porites porites",
    "PPOR":             "Porites porites",
    "PPOR_BL":          "Porites porites",
    "POP":              "Porites porites",

    # Meandrina meandrites
    "MeanMean":         "Meandrina meandrites",
    "BL_Mean":          "Meandrina meandrites",
    "MMEA":             "Meandrina meandrites",
    "MMEA_BL":          "Meandrina meandrites",
    "MM":               "Meandrina meandrites",
}

SPECIES_LIST = sorted(set(LABEL_MAP.values()))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build VLM crop dataset from CoralNet annotations"
    )
    p.add_argument("--data_root",  default=str(DATA_ROOT),
                   help="Root of CoralNet_Data directory")
    p.add_argument("--output_dir", default=str(OUTPUT_DIR),
                   help="Where to write crops and JSONL files")
    p.add_argument("--manifest",   default=None,
                   help="Optional: path to an existing split_manifest.csv "
                        "to reuse splits (skips re-crawling CSVs)")
    p.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE,
                   help="Side length of the square crop in pixels (default 300)")
    p.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    p.add_argument("--val_ratio",   type=float, default=DEFAULT_VAL_RATIO)
    p.add_argument("--oversample_threshold", type=int,
                   default=OVERSAMPLE_THRESHOLD,
                   help="Species below this count get oversampled in the JSONL")
    p.add_argument("--oversample_factor", type=int, default=OVERSAMPLE_FACTOR,
                   help="Repeat factor for thin species JSONL entries")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--dry_run", action="store_true",
                   help="Print statistics only — do not write any files")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_logging(output_dir: Path, dry_run: bool):
    handlers = [logging.StreamHandler()]
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(output_dir / "build_vlm_dataset.log",
                                mode="w", encoding="utf-8")
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def index_images(img_dir: Path) -> dict:
    """Build lowercase stem/filename -> Path index."""
    index = {}
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        for p in img_dir.rglob(ext):
            index[p.stem.lower()] = p
            index[p.name.lower()] = p
    return index


def assign_splits(
    image_names: list,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict:
    """Image-level train/val/test split to prevent spatial leakage."""
    rng = random.Random(seed)
    names = sorted(image_names)
    rng.shuffle(names)
    n = len(names)
    n_test = max(1, int(n * (1 - train_ratio - val_ratio)))
    n_val  = max(1, int(n * val_ratio))
    splits = {}
    for name in names[:n_test]:
        splits[name] = "test"
    for name in names[n_test : n_test + n_val]:
        splits[name] = "val"
    for name in names[n_test + n_val :]:
        splits[name] = "train"
    return splits


def crop_patch(img: Image.Image, row: int, col: int, half: int):
    """
    Return a square PIL crop of side (2*half) centred on (row=y, col=x).
    Returns None if the crop would go out of bounds.
    """
    w, h = img.size
    x1, y1 = col - half, row - half
    x2, y2 = col + half, row + half
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return None
    return img.crop((x1, y1, x2, y2))


def make_conversation(image_path: str, species_name: str) -> dict:
    """
    Build a single conversational JSONL record.

    Both user and assistant use the list-of-dicts content format so that
    PyArrow sees a uniform schema and does not raise ArrowInvalid when
    building the HuggingFace Dataset (mixed list / string would fail).
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": USER_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": species_name}],
            },
        ]
    }


# ── Annotation loading ────────────────────────────────────────────────────────

def load_annotations_from_csvs(
    data_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    log,
) -> pd.DataFrame:
    """Crawl CoralNet CSV files and return a unified DataFrame with splits."""
    all_rows = []

    for ann_csv in sorted(data_root.rglob("annotations.csv")):
        source_name = ann_csv.parts[len(data_root.parts)]
        try:
            df = pd.read_csv(ann_csv, low_memory=False)
        except Exception as e:
            log.warning(f"  Could not read {ann_csv}: {e}")
            continue

        if "Label code" not in df.columns:
            log.warning(f"  {source_name}: no 'Label code' column -- skipping")
            continue

        df["species"] = df["Label code"].map(LABEL_MAP)
        df_coral = df[df["species"].notna()].copy()
        if df_coral.empty:
            log.info(f"  {source_name}: no target species")
            continue

        # Locate image directory
        img_dir = ann_csv.parent / "images"
        if not img_dir.exists():
            img_dir = ann_csv.parent
        img_index = index_images(img_dir)

        # Image-level splits for this source (avoids cross-source imbalance)
        image_names = df_coral["Name"].unique().tolist()
        split_map   = assign_splits(image_names, train_ratio, val_ratio, seed)

        kept = skipped = 0
        for _, row in df_coral.iterrows():
            img_name = str(row["Name"]).strip()
            img_path = (img_index.get(img_name.lower()) or
                        img_index.get(Path(img_name).stem.lower()))
            if img_path is None:
                skipped += 1
                continue
            try:
                ann_row = int(row["Row"])
                ann_col = int(row["Column"])
            except (ValueError, KeyError):
                skipped += 1
                continue

            all_rows.append({
                "source":   source_name,
                "img_name": img_name,
                "img_path": str(img_path),
                "row":      ann_row,
                "col":      ann_col,
                "species":  row["species"],
                "split":    split_map.get(img_name, "train"),
            })
            kept += 1

        log.info(f"  {source_name}: {kept} annotations ({skipped} skipped)")

    return pd.DataFrame(all_rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    log     = setup_logging(out_dir, args.dry_run)

    random.seed(args.seed)
    half = args.patch_size // 2

    log.info("VLM Dataset Builder")
    log.info(f"  Patch size  : {args.patch_size}px")
    log.info(f"  Output dir  : {out_dir}")
    log.info(f"  Dry run     : {args.dry_run}")

    # ── 1. Load annotations ────────────────────────────────────────────────
    if args.manifest:
        log.info(f"\nLoading existing manifest: {args.manifest}")
        ann_df = pd.read_csv(args.manifest)
        required = {"img_path", "row", "col", "species", "split"}
        if not required.issubset(ann_df.columns):
            log.error(f"Manifest must have columns: {required}")
            return
        log.info(f"  Loaded {len(ann_df):,} rows")
    else:
        log.info(f"\nCrawling CoralNet CSVs in: {args.data_root}")
        ann_df = load_annotations_from_csvs(
            Path(args.data_root),
            args.train_ratio, args.val_ratio, args.seed, log,
        )

    if ann_df.empty:
        log.error("No annotations found. Check DATA_ROOT or manifest path.")
        return

    log.info(f"\nTotal annotations: {len(ann_df):,}")
    log.info(f"Unique images    : {ann_df['img_path'].nunique():,}")
    log.info("\nPer-species counts (all splits):")
    for sp, n in ann_df["species"].value_counts().items():
        log.info(f"  {sp:<35} {n:>6}")

    # ── 2. Identify thin species needing oversampling ──────────────────────
    train_counts = (
        ann_df[ann_df["split"] == "train"]["species"]
        .value_counts().to_dict()
    )
    thin_species = {
        s for s, c in train_counts.items()
        if c < args.oversample_threshold
    }
    if thin_species:
        log.info(
            f"\nThin species (< {args.oversample_threshold} train crops) "
            f"-> will be {args.oversample_factor}x oversampled in JSONL:"
        )
        for s in sorted(thin_species):
            log.info(f"  {s}")

    if args.dry_run:
        log.info("\n-- DRY RUN -- no files written --")
        for split in ("train", "val", "test"):
            sub = ann_df[ann_df["split"] == split]
            jsonl_n = sum(
                args.oversample_factor
                if (row["species"] in thin_species and split == "train")
                else 1
                for _, row in sub.iterrows()
            )
            log.info(
                f"  {split:<6}: {len(sub):>6} crops  ->  {jsonl_n:>6} JSONL lines"
            )
        return

    # ── 3. Extract crops & build JSONL ─────────────────────────────────────
    log.info("\nExtracting crops ...")

    jsonl_writers = {}
    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        jsonl_writers[split] = open(
            out_dir / f"{split}.jsonl", "w", encoding="utf-8"
        )

    counters      = defaultdict(lambda: defaultdict(int))
    skipped_edge  = 0
    skipped_open  = 0
    manifest_rows = []

    grouped      = ann_df.groupby("img_path")
    total_images = len(grouped)

    for img_path_str, group in tqdm(grouped, total=total_images, desc="Images"):
        try:
            img = Image.open(img_path_str)
            img.load()
        except Exception:
            skipped_open += len(group)
            continue

        for _, ann in group.iterrows():
            species = ann["species"]
            split   = ann["split"]
            row_px, col_px = int(ann["row"]), int(ann["col"])

            patch = crop_patch(img, row_px, col_px, half)
            if patch is None:
                skipped_edge += 1
                continue

            species_safe = species.replace(" ", "_").replace("/", "-")
            crop_dir  = out_dir / split / species_safe
            crop_dir.mkdir(parents=True, exist_ok=True)
            stem      = f"{Path(img_path_str).stem}_r{row_px}_c{col_px}"
            crop_path = crop_dir / f"{stem}.jpg"
            patch.convert("RGB").save(crop_path, "JPEG", quality=90)

            counters[species][split] += 1
            manifest_rows.append({
                "crop_path": str(crop_path),
                "species":   species,
                "split":     split,
                "source":    ann.get("source", ""),
                "img_name":  ann.get("img_name", Path(img_path_str).name),
                "row":       row_px,
                "col":       col_px,
            })

            record   = make_conversation(str(crop_path), species)
            n_copies = (
                args.oversample_factor
                if (split == "train" and species in thin_species)
                else 1
            )
            for _ in range(n_copies):
                jsonl_writers[split].write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        img.close()

    for w in jsonl_writers.values():
        w.close()

    # ── 4. Save manifest & stats ───────────────────────────────────────────
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(out_dir / "crop_manifest.csv", index=False)

    stats = {
        "patch_size":           args.patch_size,
        "oversample_threshold": args.oversample_threshold,
        "oversample_factor":    args.oversample_factor,
        "skipped_edge":         skipped_edge,
        "skipped_open":         skipped_open,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        sub_m   = manifest_df[manifest_df["split"] == split]
        jsonl_n = sum(
            args.oversample_factor
            if (split == "train" and row["species"] in thin_species)
            else 1
            for _, row in sub_m.iterrows()
        )
        stats["splits"][split] = {
            "crops": int(len(sub_m)),
            "jsonl_lines": jsonl_n,
        }
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── 5. Summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 72)
    log.info(f"{'SPECIES':<35} {'TRAIN':>7} {'VAL':>7} {'TEST':>7}")
    log.info("=" * 72)
    total_tr = total_va = total_te = 0
    for species in sorted(SPECIES_LIST):
        tr = counters[species]["train"]
        va = counters[species]["val"]
        te = counters[species]["test"]
        log.info(f"{species:<35} {tr:>7} {va:>7} {te:>7}")
        total_tr += tr; total_va += va; total_te += te
    log.info("=" * 72)
    log.info(f"{'TOTAL':<35} {total_tr:>7} {total_va:>7} {total_te:>7}")
    log.info(f"\nSkipped (edge / out-of-bounds): {skipped_edge:,}")
    log.info(f"Skipped (image open failed)   : {skipped_open:,}")
    log.info(f"\nOutput:")
    log.info(f"  Crops    -> {out_dir}/{{train,val,test}}/<species>/")
    log.info(f"  JSONL    -> {out_dir}/{{train,val,test}}.jsonl")
    log.info(f"  Manifest -> {out_dir}/crop_manifest.csv")
    log.info(f"  Stats    -> {out_dir}/dataset_stats.json")


if __name__ == "__main__":
    main()
