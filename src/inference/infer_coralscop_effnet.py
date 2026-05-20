"""
Coral Reef Inference — CoralSCOP Segmentation + EfficientNet-B3 Classification
================================================================================
Best of both worlds:
  - CoralSCOP finds coral colonies (coral-specific SAM, ~86% pixel accuracy)
  - EfficientNet-B3 Option A classifies each colony to species level (macro-F1 0.580)

How it differs from infer_option_a.py:
  - infer_option_a.py uses MobileSAM (general-purpose) for segmentation
  - This script uses CoralSCOP (coral-specific) for segmentation
  - Classification is identical — same EfficientNet-B3 Option A checkpoint
  - CoralSCOP produces fewer false positives (rocks, fish, sponge) than MobileSAM
    because it was fine-tuned on 1.3M coral masks

Output per image:
  <name>_species_map.jpg    original photo with colony masks painted per species
  <name>_overlay.jpg        semi-transparent overlay with colony outlines and labels
  <name>_label_map.png      integer label map (0=background, 1-17=species index)
  <name>_results.json       all detections with species, confidence, bbox, mask area
  summary.csv               spreadsheet of all detections across all images

How to run from the repository root:
    conda activate coralnet10
    python src/inference/infer_coralscop_effnet.py --input examples/sample_input.jpg
    python src/inference/infer_coralscop_effnet.py --input /path/to/folder/

Model files are not committed to GitHub. By default, place them under:
    models/vit_b_coralscop.pth
    models/checkpoints_option_a/best_model.pt
If CoralSCOP is not installed as a package, pass --coralscop_repo /path/to/CoralSCOP.

Arguments:
    --input           image file or folder
    --output          output folder (default: outputs/inference/coralscop_effnet)
    --scop_weights    CoralSCOP ViT-B weights (default: models/vit_b_coralscop.pth)
    --clf_checkpoint  EfficientNet-B3 Option A checkpoint (default: models/checkpoints_option_a/best_model.pt)
    --coralscop_repo  optional local CoralSCOP repo path if not installed as a package
    --min_area        min colony size as fraction of image (default: 0.003)
    --conf_threshold  min confidence to show species label (default: 0.40)
    --overlay_alpha   colour overlay transparency 0-1 (default: 0.50)
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image

# ── Project paths ─────────────────────────────────────────────────────────────


def find_project_root(script_path: Path) -> Path:
    """Return the repository root so the script works from any current folder."""
    for parent in [script_path.parent, *script_path.parents]:
        if (parent / "README.md").exists() or (parent / ".git").exists():
            return parent

    # Expected GitHub location: src/inference/<this_file>.py
    if script_path.parent.name == "inference" and script_path.parent.parent.name == "src":
        return script_path.parent.parent.parent

    return script_path.parent


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
MODEL_DIR = PROJECT_ROOT / "models"


def repo_path(*parts: str) -> Path:
    """Build an absolute path inside this repository."""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_path(path_value) -> Path:
    """Resolve user paths. Relative paths are interpreted from the repo root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return repo_path(str(path)).resolve()


SAM_IMG_SIZE = 1024


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CoralSCOP segmentation + EfficientNet-B3 Option A classification"
    )
    p.add_argument("--input",          required=True,
                   help="Image file or folder of images")
    p.add_argument("--output",         default=str(repo_path("outputs", "inference", "coralscop_effnet")),
                   help="Output folder")
    p.add_argument("--scop_weights",   default=str(MODEL_DIR / "vit_b_coralscop.pth"),
                   help="CoralSCOP ViT-B weights path")
    p.add_argument("--clf_checkpoint", default=str(MODEL_DIR / "checkpoints_option_a" / "best_model.pt"),
                   help="EfficientNet-B3 Option A classifier checkpoint")
    p.add_argument("--coralscop_repo", default=str(repo_path("external", "CoralSCOP")),
                   help="Optional local CoralSCOP repo path if segment_anything is not installed")
    p.add_argument("--min_area",       type=float, default=0.003,
                   help="Min colony size as fraction of image (default: 0.003)")
    p.add_argument("--max_area",       type=float, default=0.50,
                   help="Max colony size as fraction of image (default: 0.50)")
    p.add_argument("--conf_threshold", type=float, default=0.40,
                   help="Min confidence to show species label (default: 0.40)")
    p.add_argument("--overlay_alpha",  type=float, default=0.50,
                   help="Colour overlay transparency 0-1 (default: 0.50)")
    p.add_argument("--device",         default="auto")
    return p.parse_args()


# ── Species colours ───────────────────────────────────────────────────────────

SPECIES_COLOURS = {
    "Acropora_tenuifolia":       (255, 100, 100),
    "Agaricia_agaricites":       (100, 200, 255),
    "Colpophyllia_natans":       (255, 180,  50),
    "Lobophyllia_spp":           (150, 255, 150),
    "Madracis_auretenra":        (200, 100, 255),
    "Madracis_mirabilis":        (255,  50, 200),
    "Meandrina_meandrites":      ( 50, 220, 180),
    "Millepora_spp":             (220, 220,  50),
    "Montastraea_cavernosa":     ( 80, 160, 255),
    "Orbicella_annularis":       (255, 140,  40),
    "Orbicella_faveolata":       (180, 255,  80),
    "Orbicella_franksi":         (255,  80,  80),
    "Porites_astreoides":        ( 80, 255, 220),
    "Porites_porites":           (200, 255, 100),
    "Pseudodiploria_strigosa":   (255, 160, 200),
    "Siderastrea_siderea":       (160, 120, 255),
    "Stephanocoenia_intersepta": (255, 200, 100),
    "Unknown":                   (180, 180, 180),
}

SPECIES_LIST = [
    "Acropora_tenuifolia",    "Agaricia_agaricites",    "Colpophyllia_natans",
    "Lobophyllia_spp",        "Madracis_auretenra",     "Madracis_mirabilis",
    "Meandrina_meandrites",   "Millepora_spp",          "Montastraea_cavernosa",
    "Orbicella_annularis",    "Orbicella_faveolata",    "Orbicella_franksi",
    "Porites_astreoides",     "Porites_porites",        "Pseudodiploria_strigosa",
    "Siderastrea_siderea",    "Stephanocoenia_intersepta",
]
SPECIES_TO_IDX = {s: i+1 for i, s in enumerate(SPECIES_LIST)}


# ── Step 1: Load CoralSCOP ────────────────────────────────────────────────────

def load_coralscop(weights_path, device, coralscop_repo=None):
    """
    Load CoralSCOP ViT-B in automatic mask generation mode.
    CoralSCOP was fine-tuned on 1.3M coral masks so it finds coral
    colonies with far fewer false positives than plain SAM.
    """
    weights_path = resolve_path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"CoralSCOP weights not found: {weights_path}\n"
            "Download the weights and place them in models/, or pass --scop_weights."
        )

    if coralscop_repo:
        repo = Path(coralscop_repo).expanduser()
        if not repo.is_absolute():
            repo = repo_path(str(repo))
        if repo.exists() and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

    try:
        from segment_anything import build_sam_vit_b
    except ImportError as exc:
        raise ImportError(
            "Could not import CoralSCOP/segment_anything. Install CoralSCOP, "
            "or pass --coralscop_repo /path/to/CoralSCOP."
        ) from exc

    print(f"Loading CoralSCOP ViT-B ...")
    model = build_sam_vit_b(checkpoint=str(weights_path))
    model.to(device)
    model.eval()
    print("CoralSCOP loaded.")
    return model


def resize_for_sam(img_rgb):
    """Shrink large images so CoralSCOP fits in GPU memory."""
    h, w  = img_rgb.shape[:2]
    scale = max(h, w) / SAM_IMG_SIZE
    if scale <= 1.0:
        return img_rgb, 1.0
    resized = cv2.resize(img_rgb, (int(w/scale), int(h/scale)),
                         interpolation=cv2.INTER_AREA)
    return resized, scale


def find_colonies_coralscop(model, img_rgb, min_area_frac, max_area_frac):
    """
    Use CoralSCOP's automatic mask generator to find all coral colonies.

    Key advantage over MobileSAM:
    CoralSCOP was specifically trained on coral imagery so it:
      - Draws tighter, more accurate boundaries around coral colonies
      - Produces fewer false positives on substrate, algae, and sponge
      - Works better on low-visibility and complex-boundary images
    """
    from segment_anything import SamAutomaticMaskGenerator

    h, w     = img_rgb.shape[:2]
    img_area = h * w

    sam_img, scale = resize_for_sam(img_rgb)
    sh, sw = sam_img.shape[:2]
    if scale > 1.0:
        print(f"  Resized: {w}x{h} -> {sw}x{sh} (scale={scale:.2f})")

    # Scale points_per_side to image resolution
    points = max(8, min(32, min(sw, sh) // 60))
    gen = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=points,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=200,
    )

    raw_masks = gen.generate(sam_img)

    colonies = []
    for m in raw_masks:
        frac = m["area"] / img_area
        if frac < min_area_frac or frac > max_area_frac:
            continue

        x, y, bw, bh = m["bbox"]
        x1 = int(x * scale);        y1 = int(y * scale)
        x2 = int((x+bw) * scale);   y2 = int((y+bh) * scale)

        # Scale mask back to original image resolution
        small_mask = m["segmentation"].astype(np.uint8)
        full_mask  = cv2.resize(small_mask, (w, h),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

        colonies.append({
            "bbox":      (x1, y1, x2, y2),
            "mask":      full_mask,
            "area":      int(np.sum(full_mask)),
            "area_frac": round(frac, 4),
            "stability": round(float(m.get("stability_score", 0)), 3),
            "iou":       round(float(m.get("predicted_iou", 0)), 3),
        })

    # Remove overlapping detections — keep most stable
    before    = len(colonies)
    colonies  = nms(colonies, iou_threshold=0.5)
    print(f"  CoralSCOP: {before} raw -> {len(colonies)} after overlap removal")
    return colonies


def nms(detections, iou_threshold=0.5):
    """Remove duplicate detections keeping the most stable mask."""
    if not detections:
        return detections
    detections = sorted(detections, key=lambda d: -d["stability"])
    kept = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        dup = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k["bbox"]
            ix1=max(x1,kx1); iy1=max(y1,ky1)
            ix2=min(x2,kx2); iy2=min(y2,ky2)
            if ix2<=ix1 or iy2<=iy1: continue
            inter=(ix2-ix1)*(iy2-iy1)
            union=(x2-x1)*(y2-y1)+(kx2-kx1)*(ky2-ky1)-inter+1e-6
            if inter/union > iou_threshold:
                dup=True; break
        if not dup:
            kept.append(det)
    return kept


# ── Step 2: Load EfficientNet-B3 Option A classifier ─────────────────────────

# Option A preprocessing: background zeroed out, then resized to 224x224
# This must match exactly how Option A patches were generated during training
CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize(int(224 * 1.1)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_classifier(checkpoint_path, device):
    """Load the trained EfficientNet-B3 Option A classifier."""
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {checkpoint_path}\n"
            "Download the trained checkpoint and place it in models/, or pass --clf_checkpoint."
        )

    print(f"Loading EfficientNet-B3 Option A from {checkpoint_path} ...")
    ckpt        = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt.get("class_names", [])
    num_classes = len(class_names) if class_names else 17

    model = timm.create_model("efficientnet_b3", pretrained=False,
                               num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"Classifier loaded — {num_classes} species, "
          f"val F1={ckpt.get('val_f1', 0):.3f}")
    return model, class_names


def make_option_a_crop(img_rgb, mask, bbox):
    """
    Produce an Option A style crop:
      1. Extract the bounding box region from the image
      2. Zero out all pixels outside the coral mask (set to black)

    This matches exactly what the Option A model was trained on —
    the classifier sees the colony texture without background distractions.
    """
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    crop      = img_rgb[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]
    crop[~mask_crop] = 0   # zero out background
    return crop


def classify_colony(model, class_names, crop_rgb, device):
    """
    Classify one Option A crop.
    Returns (species_name, confidence, top3_list).
    """
    if crop_rgb is None or crop_rgb.size == 0 or min(crop_rgb.shape[:2]) < 10:
        return "Unknown", 0.0, []

    tensor = CLASSIFIER_TRANSFORM(
        Image.fromarray(crop_rgb)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    best_idx  = int(np.argmax(probs))
    best_conf = float(probs[best_idx])
    best_name = class_names[best_idx] if class_names else str(best_idx)
    top3      = [
        (class_names[i] if class_names else str(i), float(c))
        for i, c in sorted(enumerate(probs), key=lambda x: -x[1])[:3]
    ]
    return best_name, best_conf, top3


# ── Step 3: Build output images ───────────────────────────────────────────────

def build_species_map(img_rgb, detections, conf_threshold, overlay_alpha):
    """
    Paint each detected colony in its species colour.
    Returns species_map, overlay, label_map, painted_species dict.
    """
    h, w          = img_rgb.shape[:2]
    colour_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    label_map     = np.zeros((h, w), dtype=np.uint8)
    painted       = {}

    for det in detections:
        sp  = det["species"] if det["confidence"] >= conf_threshold else "Unknown"
        rgb = SPECIES_COLOURS.get(sp, SPECIES_COLOURS["Unknown"])
        colour_canvas[det["mask"]] = rgb
        label_map[det["mask"]]     = SPECIES_TO_IDX.get(sp, 0)
        if sp != "Unknown":
            painted[sp] = rgb

    # Overlay: blend colour with original photo on coral pixels
    coral_px = colour_canvas.any(axis=2)
    overlay  = img_rgb.copy().astype(float)
    overlay[coral_px] = (
        (1 - overlay_alpha) * img_rgb.astype(float)[coral_px] +
        overlay_alpha * colour_canvas.astype(float)[coral_px]
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Species map: solid colour on coral, photo elsewhere
    species_map = img_rgb.copy()
    species_map[coral_px] = colour_canvas[coral_px]

    return species_map, overlay, label_map, painted


def draw_outlines_and_labels(img_bgr, detections, conf_threshold):
    """Draw colony outlines and species labels on image."""
    out   = img_bgr.copy()
    h, w  = out.shape[:2]
    scale = max(w, h) / 1920
    font  = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        sp   = det["species"] if det["confidence"] >= conf_threshold else "Unknown"
        conf = det["confidence"]
        mask = det["mask"].astype(np.uint8)
        rgb  = SPECIES_COLOURS.get(sp, SPECIES_COLOURS["Unknown"])
        bgr  = (rgb[2], rgb[1], rgb[0])

        # Colony outline from mask contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, bgr, max(1, int(1.5*scale)))

        # Label
        x1, y1, x2, y2 = det["bbox"]
        if conf >= conf_threshold:
            label = f"{sp.replace('_',' ')} {conf:.0%}"
        else:
            label = f"Coral ({conf:.0%})"

        fs = max(0.3, 0.4*scale)
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        lx = max(0, min(x1, w-tw-4))
        ly = max(th+4, y1-4)
        cv2.rectangle(out, (lx, ly-th-2), (lx+tw+4, ly+2), bgr, -1)
        br = 0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]
        tc = (20,20,20) if br > 160 else (255,255,255)
        cv2.putText(out, label, (lx+2, ly), font, fs, tc, 1, cv2.LINE_AA)

    return out


def add_legend(img_bgr, painted, image_height):
    """Add species legend panel on the right."""
    if not painted:
        return img_bgr
    pad  = 12; lh = 28
    ph   = max(image_height, pad*2 + 50 + len(painted)*lh)
    panel = np.full((ph, 310, 3), 30, dtype=np.uint8)
    font  = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, "CoralSCOP + EfficientNet-B3",
                (pad, pad+16), font, 0.45, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(panel, "Detected species:",
                (pad, pad+34), font, 0.45, (230,230,230), 1, cv2.LINE_AA)

    for i, (sp, rgb) in enumerate(sorted(painted.items())):
        y   = pad + 48 + i * lh
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(panel, (pad, y-12), (pad+16, y+4), bgr, -1)
        name = sp.replace("_"," ")
        if len(name) > 28: name = name[:26] + "…"
        cv2.putText(panel, name, (pad+22, y+2),
                    font, 0.42, (210,210,210), 1, cv2.LINE_AA)

    h = img_bgr.shape[0]
    if h < panel.shape[0]:
        img_bgr = np.pad(img_bgr,
                         ((0, panel.shape[0]-h), (0,0), (0,0)),
                         mode="edge")
    return np.concatenate([img_bgr, panel[:img_bgr.shape[0]]], axis=1)


# ── Step 4: Process one image ─────────────────────────────────────────────────

def process_image(img_path, coralscop, classifier, class_names,
                  output_dir, min_area, max_area, conf_threshold,
                  overlay_alpha, device):
    """Run CoralSCOP segmentation + EfficientNet-B3 classification on one image."""
    print(f"\n{'─'*60}")
    print(f"Image: {img_path.name}")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print("  ERROR: could not open image")
        return []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    print(f"  Size: {w}x{h}")

    # Step 1: CoralSCOP finds coral colonies
    t0       = time.time()
    colonies = find_colonies_coralscop(
        coralscop, img_rgb, min_area, max_area)
    det_time = time.time() - t0
    print(f"  Detection: {det_time:.1f}s")

    if not colonies:
        print("  No colonies found — try lowering --min_area")
        return []

    # Step 2: Classify each colony using Option A style (background zeroed)
    detections = []
    for idx, colony in enumerate(colonies):
        x1, y1, x2, y2 = colony["bbox"]

        # Make Option A crop: bbox crop with background zeroed out
        crop = make_option_a_crop(img_rgb, colony["mask"], colony["bbox"])
        species, conf, top3 = classify_colony(
            classifier, class_names, crop, device)

        det = {
            "id":         idx + 1,
            "species":    species,
            "confidence": conf,
            "top3":       top3,
            "bbox":       colony["bbox"],
            "area":       colony["area"],
            "area_frac":  colony["area_frac"],
            "stability":  colony["stability"],
            "mask":       colony["mask"],
        }
        detections.append(det)

        flag = "✓" if conf >= conf_threshold else "~"
        print(f"  [{idx+1:2d}] {flag}  "
              f"{species.replace('_',' '):<33}  "
              f"{conf:.1%}  "
              f"(covers {colony['area_frac']:.1%})")

    clf_time = time.time() - t0 - det_time
    print(f"  Classification: {clf_time:.1f}s")

    # Coverage stats
    total_coral = sum(d["area"] for d in detections
                      if d["confidence"] >= conf_threshold)
    coverage    = total_coral / (h * w) * 100
    print(f"\n  Coral coverage: {coverage:.1f}%")
    sp_counts = Counter(d["species"] for d in detections
                        if d["confidence"] >= conf_threshold)
    for sp, n in sp_counts.most_common():
        sp_area = sum(d["area"] for d in detections
                      if d["species"] == sp and
                      d["confidence"] >= conf_threshold)
        print(f"    {sp.replace('_',' '):<35}  {n} colonies  "
              f"{sp_area/(h*w)*100:.1f}% cover")

    # Step 3: Build output images
    species_map, overlay, label_map, painted = build_species_map(
        img_rgb, detections, conf_threshold, overlay_alpha)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    outlined    = draw_outlines_and_labels(
        overlay_bgr, detections, conf_threshold)

    sp_map_bgr  = cv2.cvtColor(species_map, cv2.COLOR_RGB2BGR)
    sp_out      = add_legend(sp_map_bgr, painted, h)
    ov_out      = add_legend(outlined, painted, h)

    # Save outputs
    stem = img_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_species_map.jpg"),
                sp_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(output_dir / f"{stem}_overlay.jpg"),
                ov_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(output_dir / f"{stem}_label_map.png"), label_map)

    print(f"\n  Saved: {stem}_species_map.jpg  |  _overlay.jpg  |  _label_map.png")

    # Save JSON (exclude numpy mask arrays)
    with open(output_dir / f"{stem}_results.json", "w") as f:
        json.dump({
            "image":          img_path.name,
            "segmentation":   "CoralSCOP ViT-B",
            "classification": "EfficientNet-B3 Option A",
            "size":           {"width": w, "height": h},
            "coral_coverage": round(coverage, 2),
            "n_colonies":     len(detections),
            "detections": [
                {k: v for k, v in d.items() if k != "mask"}
                for d in detections
            ],
        }, f, indent=2)

    return detections


# ── Step 5: Summary CSV ───────────────────────────────────────────────────────

def save_summary(all_dets, output_dir, conf_threshold):
    import csv
    rows = []
    for img_name, dets in all_dets.items():
        for d in dets:
            rows.append({
                "image":           img_name,
                "id":              d["id"],
                "species":         d["species"].replace("_", " "),
                "confidence":      round(d["confidence"], 3),
                "above_threshold": d["confidence"] >= conf_threshold,
                "x1": d["bbox"][0], "y1": d["bbox"][1],
                "x2": d["bbox"][2], "y2": d["bbox"][3],
                "area_pixels":     d["area"],
                "area_fraction":   d["area_frac"],
                "stability":       d["stability"],
            })
    if not rows:
        return
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary saved: {csv_path}")

    above = [r["species"] for r in rows if r["above_threshold"]]
    if above:
        print(f"\nSpecies breakdown (>= {conf_threshold:.0%} confidence):")
        for sp, n in Counter(above).most_common():
            print(f"  {sp:<35}  {n} detection(s)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    print(f"Device         : {device}")
    if device == "cuda":
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Segmentation   : CoralSCOP ViT-B")
    print(f"Classification : EfficientNet-B3 Option A")

    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output         : {output_dir}")

    # Collect images
    input_path = resolve_path(args.input)
    if input_path.is_file():
        img_paths = [input_path]
    elif input_path.is_dir():
        img_paths = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    else:
        print(f"ERROR: path does not exist: {input_path}")
        return

    if not img_paths:
        print("No images found.")
        return
    print(f"Images         : {len(img_paths)}\n")

    # Load models
    coralscop             = load_coralscop(args.scop_weights, device, args.coralscop_repo)
    classifier, class_names = load_classifier(args.clf_checkpoint, device)

    # Process all images
    all_dets = {}
    for img_path in img_paths:
        dets = process_image(
            img_path, coralscop, classifier, class_names,
            output_dir, args.min_area, args.max_area,
            args.conf_threshold, args.overlay_alpha, device
        )
        if dets:
            all_dets[img_path.name] = dets

    save_summary(all_dets, output_dir, args.conf_threshold)

    total     = sum(len(d) for d in all_dets.values())
    confident = sum(1 for dets in all_dets.values()
                    for d in dets if d["confidence"] >= args.conf_threshold)
    print(f"\n{'='*60}")
    print(f"Images processed  : {len(all_dets)}")
    print(f"Total colonies    : {total}")
    print(f"Confidently labelled: {confident} (>= {args.conf_threshold:.0%})")
    print(f"Results saved to  : {output_dir}")


if __name__ == "__main__":
    main()
