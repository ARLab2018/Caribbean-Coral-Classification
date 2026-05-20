"""
Build Split Manifest for CoralSCOP Fine-tuning
==============================================
Generates split_manifest.csv from raw CoralNet annotation CSV files.

This script is GitHub/local-machine friendly:
- It does not depend on a user-specific folder such as ~/Independent_study.
- By default, it looks for data inside the repository.
- You can override paths from the command line.

Expected default input location:
    <repo_root>/data/CoralNet_Data/

Default output location:
    <repo_root>/outputs/split_manifest.csv

Example usage:
    python build_manifest.py

    python build_manifest.py \
        --data-root data/CoralNet_Data \
        --output-csv outputs/split_manifest.csv

Columns in the output CSV:
    img_name, img_path, source, row, col, species, split
"""

import argparse
import random
from pathlib import Path

import pandas as pd

# ── Default paths ─────────────────────────────────────────────────────────────
# Assumption: this file is placed in the GitHub repository root.
# If your data is stored elsewhere, pass --data-root and --output-csv manually.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "CoralNet_Data"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "split_manifest.csv"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15  # remainder

SEED = 42

# ── Label map ─────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "MadMir": "Madracis_mirabilis",
    "AgAga": "Agaricia_agaricites", "BL_Aga": "Agaricia_agaricites",
    "AAGA": "Agaricia_agaricites", "AAGA_BL": "Agaricia_agaricites", "AGAAGA": "Agaricia_agaricites",
    "OrbAnn": "Orbicella_annularis", "BL_OrbAnn": "Orbicella_annularis",
    "OANN": "Orbicella_annularis", "OANN_BL": "Orbicella_annularis", "ORBANN": "Orbicella_annularis",
    "Lobo": "Lobophyllia_spp", "LOBO": "Lobophyllia_spp",
    "OrbFav": "Orbicella_faveolata", "BL_OrbFav": "Orbicella_faveolata",
    "OFAV": "Orbicella_faveolata", "OFAV_BL": "Orbicella_faveolata", "ORBFAV": "Orbicella_faveolata",
    "MCav": "Montastraea_cavernosa", "BL_MCav": "Montastraea_cavernosa",
    "MCAV": "Montastraea_cavernosa", "MCAV_BL": "Montastraea_cavernosa",
    "PorAstr": "Porites_astreoides", "BL_PAst": "Porites_astreoides",
    "PAST": "Porites_astreoides", "PAST_BL": "Porites_astreoides",
    "Millepo": "Millepora_spp", "Mil_spp": "Millepora_spp",
    "MILA": "Millepora_spp", "MILC": "Millepora_spp", "MILLE": "Millepora_spp", "BL_Mille": "Millepora_spp",
    "SidSid": "Siderastrea_siderea", "BL_SidSid": "Siderastrea_siderea",
    "SSID": "Siderastrea_siderea", "SSID_BL": "Siderastrea_siderea",
    "ColNat": "Colpophyllia_natans", "BL_CoNat": "Colpophyllia_natans",
    "CNAT": "Colpophyllia_natans", "CNAT_BL": "Colpophyllia_natans", "CNAt": "Colpophyllia_natans",
    "DipStr": "Pseudodiploria_strigosa", "BL_DipStr": "Pseudodiploria_strigosa",
    "PSTRI": "Pseudodiploria_strigosa", "PSTR_BL": "Pseudodiploria_strigosa",
    "MAUR": "Madracis_auretenra", "MALC": "Madracis_auretenra", "MLAM": "Madracis_auretenra",
    "StInt": "Stephanocoenia_intersepta", "BL_StInt": "Stephanocoenia_intersepta",
    "SINT": "Stephanocoenia_intersepta", "SINT_BL": "Stephanocoenia_intersepta",
    "OrbFrank": "Orbicella_franksi", "BL_OrbFran": "Orbicella_franksi",
    "OFRA": "Orbicella_franksi", "OFRA_BL": "Orbicella_franksi",
    "ATEN": "Acropora_tenuifolia", "ATEN_BL": "Acropora_tenuifolia",
    "PorPor": "Porites_porites", "BL_PorPor": "Porites_porites",
    "PPOR": "Porites_porites", "PPOR_BL": "Porites_porites", "POP": "Porites_porites",
    "MeanMean": "Meandrina_meandrites", "BL_Mean": "Meandrina_meandrites",
    "MMEA": "Meandrina_meandrites", "MMEA_BL": "Meandrina_meandrites", "MM": "Meandrina_meandrites",
}


def parse_args():
    """Read command-line arguments for portable local/GitHub usage."""
    parser = argparse.ArgumentParser(
        description="Build split_manifest.csv from CoralNet annotation CSV files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Folder containing CoralNet annotation folders. "
            "Default: data/CoralNet_Data relative to this script."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=(
            "Path where split_manifest.csv will be saved. "
            "Default: outputs/split_manifest.csv relative to this script."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed used for image-level train/val/test splitting.",
    )
    return parser.parse_args()


def assign_splits(image_names, seed=SEED):
    """Assign train/val/test splits at image level."""
    rng = random.Random(seed)
    names = sorted(image_names)
    rng.shuffle(names)

    n = len(names)
    n_test = max(1, int(n * VAL_RATIO)) if n > 0 else 0  # 15% test
    n_val = max(1, int(n * VAL_RATIO)) if n > 0 else 0   # 15% val

    splits = {}
    for name in names[:n_test]:
        splits[name] = "test"
    for name in names[n_test:n_test + n_val]:
        splits[name] = "val"
    for name in names[n_test + n_val:]:
        splits[name] = "train"
    return splits


def build_image_index(img_dir):
    """Create a lookup from image filename/stem to full image path."""
    img_index = {}
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        for path in img_dir.rglob(ext):
            img_index[path.stem.lower()] = path
            img_index[path.name.lower()] = path
    return img_index


def build_manifest(data_root, seed=SEED):
    """Build the manifest dataframe from all annotations.csv files."""
    all_rows = []

    for ann_csv in sorted(data_root.rglob("annotations.csv")):
        source_name = ann_csv.relative_to(data_root).parts[0]

        try:
            df = pd.read_csv(ann_csv, low_memory=False)
        except Exception as error:
            print(f"  Could not read {ann_csv}: {error}")
            continue

        required_columns = {"Name", "Label code", "Row", "Column"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            print(f"  {source_name}: skipped because columns are missing: {sorted(missing_columns)}")
            continue

        # Map label codes to species names.
        df["species"] = df["Label code"].map(LABEL_MAP)
        df_coral = df[df["species"].notna()].copy()
        if df_coral.empty:
            print(f"  {source_name}: no target species")
            continue

        # Build image index for this source.
        img_dir = ann_csv.parent / "images"
        if not img_dir.exists():
            img_dir = ann_csv.parent
        img_index = build_image_index(img_dir)

        # Assign image-level splits for this source.
        image_names = df_coral["Name"].unique().tolist()
        split_map = assign_splits(image_names, seed=seed)

        kept = skipped = 0
        for _, row in df_coral.iterrows():
            img_name = str(row["Name"]).strip()
            img_path = (
                img_index.get(img_name.lower())
                or img_index.get(Path(img_name).stem.lower())
            )
            if img_path is None:
                skipped += 1
                continue

            try:
                ann_row = int(row["Row"])
                ann_col = int(row["Column"])
            except (ValueError, TypeError):
                skipped += 1
                continue

            all_rows.append({
                "img_name": img_name,
                "img_path": str(img_path),
                "source": source_name,
                "row": ann_row,
                "col": ann_col,
                "species": row["species"],
                "split": split_map.get(img_name, "train"),
            })
            kept += 1

        print(f"  {source_name}: {kept} annotations ({skipped} skipped — image not found)")

    columns = ["img_name", "img_path", "source", "row", "col", "species", "split"]
    return pd.DataFrame(all_rows, columns=columns)


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()

    print("Building split manifest from CoralNet annotations ...")
    print(f"Data root:  {data_root}")
    print(f"Output CSV: {output_csv}")

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_root}\n"
            "Place your CoralNet data at data/CoralNet_Data, or run with:\n"
            "python build_manifest.py --data-root /path/to/CoralNet_Data"
        )

    manifest = build_manifest(data_root=data_root, seed=args.seed)

    print(f"\nTotal: {len(manifest):,} annotations")

    if manifest.empty:
        print("No annotations were found. CSV was not created.")
        return

    print("Splits:")
    for split, grp in manifest.groupby("split"):
        print(f"  {split}: {len(grp):,} annotations ({grp['img_name'].nunique():,} images)")

    print("\nSpecies counts:")
    for sp, n in manifest["species"].value_counts().items():
        print(f"  {sp:<35} {n}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")


if __name__ == "__main__":
    main()
