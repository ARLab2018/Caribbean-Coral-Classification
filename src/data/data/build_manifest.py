"""
Build Split Manifest for CoralSCOP Fine-tuning
================================================
Generates split_manifest.csv from the raw CoralNet annotation CSVs.
Uses the same image-level 70/15/15 split logic as patch_extractor_v2.py
but sources data directly from CoralNet annotations, not from patch folders.

Output:
    ~/Independent_study/patches_option_f/split_manifest.csv

Columns:
    img_name, img_path, source, row, col, species, split
"""

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ── Settings ──────────────────────────────────────────────────────────────────

DATA_ROOT  = Path.home() / "Independent_study" / "CoralNet_Data"
OUTPUT_CSV = Path.home() / "Independent_study" / "patches_option_f" / "split_manifest.csv"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)

SEED = 42
random.seed(SEED)

# ── Label map ─────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "MadMir":"Madracis_mirabilis",
    "AgAga":"Agaricia_agaricites","BL_Aga":"Agaricia_agaricites",
    "AAGA":"Agaricia_agaricites","AAGA_BL":"Agaricia_agaricites","AGAAGA":"Agaricia_agaricites",
    "OrbAnn":"Orbicella_annularis","BL_OrbAnn":"Orbicella_annularis",
    "OANN":"Orbicella_annularis","OANN_BL":"Orbicella_annularis","ORBANN":"Orbicella_annularis",
    "Lobo":"Lobophyllia_spp","LOBO":"Lobophyllia_spp",
    "OrbFav":"Orbicella_faveolata","BL_OrbFav":"Orbicella_faveolata",
    "OFAV":"Orbicella_faveolata","OFAV_BL":"Orbicella_faveolata","ORBFAV":"Orbicella_faveolata",
    "MCav":"Montastraea_cavernosa","BL_MCav":"Montastraea_cavernosa",
    "MCAV":"Montastraea_cavernosa","MCAV_BL":"Montastraea_cavernosa",
    "PorAstr":"Porites_astreoides","BL_PAst":"Porites_astreoides",
    "PAST":"Porites_astreoides","PAST_BL":"Porites_astreoides",
    "Millepo":"Millepora_spp","Mil_spp":"Millepora_spp",
    "MILA":"Millepora_spp","MILC":"Millepora_spp","MILLE":"Millepora_spp","BL_Mille":"Millepora_spp",
    "SidSid":"Siderastrea_siderea","BL_SidSid":"Siderastrea_siderea",
    "SSID":"Siderastrea_siderea","SSID_BL":"Siderastrea_siderea",
    "ColNat":"Colpophyllia_natans","BL_CoNat":"Colpophyllia_natans",
    "CNAT":"Colpophyllia_natans","CNAT_BL":"Colpophyllia_natans","CNAt":"Colpophyllia_natans",
    "DipStr":"Pseudodiploria_strigosa","BL_DipStr":"Pseudodiploria_strigosa",
    "PSTRI":"Pseudodiploria_strigosa","PSTR_BL":"Pseudodiploria_strigosa",
    "MAUR":"Madracis_auretenra","MALC":"Madracis_auretenra","MLAM":"Madracis_auretenra",
    "StInt":"Stephanocoenia_intersepta","BL_StInt":"Stephanocoenia_intersepta",
    "SINT":"Stephanocoenia_intersepta","SINT_BL":"Stephanocoenia_intersepta",
    "OrbFrank":"Orbicella_franksi","BL_OrbFran":"Orbicella_franksi",
    "OFRA":"Orbicella_franksi","OFRA_BL":"Orbicella_franksi",
    "ATEN":"Acropora_tenuifolia","ATEN_BL":"Acropora_tenuifolia",
    "PorPor":"Porites_porites","BL_PorPor":"Porites_porites",
    "PPOR":"Porites_porites","PPOR_BL":"Porites_porites","POP":"Porites_porites",
    "MeanMean":"Meandrina_meandrites","BL_Mean":"Meandrina_meandrites",
    "MMEA":"Meandrina_meandrites","MMEA_BL":"Meandrina_meandrites","MM":"Meandrina_meandrites",
}


def assign_splits(image_names):
    """Assign train/val/test splits at image level."""
    names = sorted(image_names)
    random.shuffle(names)
    n      = len(names)
    n_test = max(1, int(n * VAL_RATIO))   # 15% test
    n_val  = max(1, int(n * VAL_RATIO))   # 15% val
    splits = {}
    for name in names[:n_test]:
        splits[name] = "test"
    for name in names[n_test:n_test + n_val]:
        splits[name] = "val"
    for name in names[n_test + n_val:]:
        splits[name] = "train"
    return splits


def main():
    print("Building split manifest from CoralNet annotations ...")

    all_rows = []

    for ann_csv in sorted(DATA_ROOT.rglob("annotations.csv")):
        source_name = ann_csv.parts[len(DATA_ROOT.parts)]

        try:
            df = pd.read_csv(ann_csv, low_memory=False)
        except Exception as e:
            print(f"  Could not read {ann_csv}: {e}")
            continue

        if "Label code" not in df.columns:
            continue

        # Map label codes to species names
        df["species"] = df["Label code"].map(LABEL_MAP)
        df_coral = df[df["species"].notna()].copy()
        if df_coral.empty:
            print(f"  {source_name}: no target species")
            continue

        # Build image index for this source
        img_dir = ann_csv.parent / "images"
        if not img_dir.exists():
            img_dir = ann_csv.parent
        img_index = {}
        for ext in ("*.jpg","*.JPG","*.jpeg","*.JPEG","*.png","*.PNG"):
            for p in img_dir.rglob(ext):
                img_index[p.stem.lower()] = p
                img_index[p.name.lower()] = p

        # Assign image-level splits for this source
        image_names  = df_coral["Name"].unique().tolist()
        split_map    = assign_splits(image_names)

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
                "img_name": img_name,
                "img_path": str(img_path),
                "source":   source_name,
                "row":      ann_row,
                "col":      ann_col,
                "species":  row["species"],
                "split":    split_map.get(img_name, "train"),
            })
            kept += 1

        print(f"  {source_name}: {kept} annotations "
              f"({skipped} skipped — image not found)")

    manifest = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(manifest):,} annotations")
    print(f"Splits:")
    for split, grp in manifest.groupby("split"):
        print(f"  {split}: {len(grp):,} annotations  "
              f"({grp['img_name'].nunique():,} images)")

    print(f"\nSpecies counts:")
    for sp, n in manifest["species"].value_counts().items():
        print(f"  {sp:<35} {n}")

    # Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
