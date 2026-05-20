"""
Coral Reef Inference — SAM2 Segmentation + EfficientNet-B3 Option A Classification
=====================================================================================
Upgrades MobileSAM → SAM2 for better coral colony segmentation:
  - SAM2 produces cleaner single-colony masks (no more fragmentation)
  - Better boundary detection on textured surfaces like Orbicella
  - Larger model trained on 11M+ images/videos vs MobileSAM's lightweight design
  - Not_coral rejection filter removes sand/substrate false positives

How it differs from infer_option_a.py:
  - Uses SAM2 (sam2.1_hiera_base_plus) instead of MobileSAM
  - Adds Not_coral confidence threshold to reject non-coral detections
  - Uses SAM2 automatic mask generator with coral-optimised parameters
  - Better handling of large colonies that MobileSAM fragmented

Requirements:
    Install SAM2 as a package, for example: pip install -e /path/to/sam2/
    Put the SAM2 weights at: models/sam2.1_hiera_base_plus.pt
    The default config uses the installed SAM2 package: configs/sam2.1/sam2.1_hiera_b+

Output per image:
    <name>_species_map.jpg    — colour overlay per species
    <name>_overlay.jpg        — semi-transparent with outlines and labels
    <name>_label_map.png      — integer label map (0=background, 1-17=species)
    <name>_results.json       — all detections with species, confidence, bbox
    summary.csv               — spreadsheet across all images

Usage from the repository root:
    conda activate coralnet10
    python src/inference/infer_sam2_effnet.py --input examples/sample_input.jpg
    python src/inference/infer_sam2_effnet.py --input /path/to/folder/
    python src/inference/infer_sam2_effnet.py --input /path/to/folder/ --not_coral_threshold 0.5

Model files are not committed to GitHub. By default, place them under:
    models/sam2.1_hiera_base_plus.pt
    models/checkpoints_sam2/best_model.pt
"""


import sys as _sys
import os as _os
# Remove script directory from sys.path to prevent sam2/ repo folder
# from shadowing the installed sam2 package
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
if _script_dir in _sys.path:
    _sys.path.remove(_script_dir)
_sys.path = [p for p in _sys.path
             if not _os.path.exists(_os.path.join(p, "sam2", "build_sam.py"))]

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


SAM2_WEIGHTS  = MODEL_DIR / "sam2.1_hiera_base_plus.pt"
SAM2_CONFIG   = "configs/sam2.1/sam2.1_hiera_b+"
CLF_CKPT      = MODEL_DIR / "checkpoints_sam2" / "best_model.pt"

# ── Species ───────────────────────────────────────────────────────────────────

SPECIES_LIST = [
    "Acropora_tenuifolia",    "Agaricia_agaricites",    "Colpophyllia_natans",
    "Lobophyllia_spp",        "Madracis_auretenra",     "Madracis_mirabilis",
    "Meandrina_meandrites",   "Millepora_spp",          "Montastraea_cavernosa",
    "Orbicella_annularis",    "Orbicella_faveolata",    "Orbicella_franksi",
    "Porites_astreoides",     "Porites_porites",        "Pseudodiploria_strigosa",
    "Siderastrea_siderea",    "Stephanocoenia_intersepta",
]

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

# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SAM2 segmentation + EfficientNet-B3 Option A classification"
    )
    p.add_argument("--input",                required=True,
                   help="Image file or folder")
    p.add_argument("--output",               default=str(repo_path("outputs", "inference", "sam2_effnet")),
                   help="Output folder")
    p.add_argument("--sam2_weights",         default=str(SAM2_WEIGHTS),
                   help="SAM2 model weights path")
    p.add_argument("--sam2_config",          default=str(SAM2_CONFIG),
                   help="SAM2 model config yaml path")
    p.add_argument("--clf_checkpoint",       default=str(CLF_CKPT),
                   help="EfficientNet-B3 Option A checkpoint")
    p.add_argument("--min_area",             type=float, default=0.003,
                   help="Min colony size as fraction of image (default: 0.003)")
    p.add_argument("--max_area",             type=float, default=0.15,
                   help="Max colony size as fraction of image (default: 0.15)")
    p.add_argument("--conf_threshold",       type=float, default=0.40,
                   help="Min confidence to label a species (default: 0.40)")
    p.add_argument("--not_coral_threshold",  type=float, default=0.45,
                   help="If top prediction is below this treat as not-coral (default: 0.45)")
    p.add_argument("--overlay_alpha",        type=float, default=0.50,
                   help="Colour overlay transparency (default: 0.50)")
    p.add_argument("--device",               default="auto")
    return p.parse_args()


# ── SAM2 Segmentation ─────────────────────────────────────────────────────────

def normalise_sam2_config(config_path):
    """Return the SAM2 config name expected by Hydra/build_sam2."""
    config_str = str(config_path)
    config_file = Path(config_str).expanduser()
    if config_file.suffix == ".yaml" and not config_file.is_absolute():
        repo_config = resolve_path(config_file)
        if repo_config.exists():
            config_file = repo_config

    if config_file.suffix == ".yaml" and config_file.exists():
        import os
        sam2_pkg = Path(__import__("sam2").__file__).resolve().parent
        return os.path.relpath(config_file.resolve(), sam2_pkg).replace(".yaml", "")

    return config_str.replace(".yaml", "")


def load_sam2(weights_path, config_path, device):
    """Load SAM2 automatic mask generator."""
    weights_path = resolve_path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"SAM2 weights not found: {weights_path}\n"
            "Download the weights and place them in models/, or pass --sam2_weights."
        )

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    print(f"Loading SAM2 from {weights_path} ...")
    config_name = normalise_sam2_config(config_path)
    sam2 = build_sam2(config_name, str(weights_path), device=device, apply_postprocessing=False)
    sam2.eval()

    # Coral-optimised automatic mask generator parameters
    # Key differences from MobileSAM defaults:
    #   - Higher points_per_side: denser sampling catches more colonies
    #   - Lower pred_iou_thresh: accept more candidate masks
    #   - Higher stability_score_thresh: filter unstable fragments
    #   - crop_n_layers=1: multi-scale detection for large colonies
    generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.72,
        stability_score_thresh=0.86,
        stability_score_offset=0.75,
        mask_threshold=-0.5,
        box_nms_thresh=0.7,
        crop_n_layers=1,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=0.45,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=300,
    )
    print("SAM2 loaded.")
    return generator


def find_colonies_sam2(generator, img_rgb, min_area_frac, max_area_frac):
    """
    Run SAM2 automatic mask generation on the image.

    SAM2 improvements over MobileSAM for coral detection:
    1. Hiera ViT backbone trained on 11M+ diverse images/videos
    2. Better boundary detection on textured surfaces (brain corals, plating corals)
    3. Multi-scale cropping finds both small recruits and large colonies
    4. More coherent single-colony masks vs MobileSAM's tendency to fragment
    """
    h, w     = img_rgb.shape[:2]
    img_area = h * w

    # SAM2 handles large images natively but we resize for memory safety
    scale = 1.0
    if max(h, w) > 1024:
        scale   = max(h, w) / 1024
        new_w   = int(w / scale)
        new_h   = int(h / scale)
        sam_img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  Resized: {w}x{h} → {new_w}x{new_h}")
    else:
        sam_img = img_rgb

    with torch.inference_mode(), torch.autocast(
        "cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16
    ):
        raw_masks = generator.generate(sam_img)

    print(f"  SAM2: {len(raw_masks)} raw masks")

    colonies = []
    for m in raw_masks:
        area_frac = m["area"] / img_area
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue

        # Skip masks spanning the full image dimension (background/substrate)
        x, y, bw, bh = m["bbox"]
        if bw >= sam_img.shape[1] * 0.90 or bh >= sam_img.shape[0] * 0.90:
            continue

        # Scale bbox and mask back to original resolution
        if scale > 1.0:
            x1 = int(x * scale);         y1 = int(y * scale)
            x2 = int((x + bw) * scale);  y2 = int((y + bh) * scale)
            small_mask = m["segmentation"].astype(np.uint8)
            full_mask  = cv2.resize(small_mask, (w, h),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bw), int(y + bh)
            full_mask = m["segmentation"].astype(bool)

        colonies.append({
            "bbox":      (x1, y1, x2, y2),
            "mask":      full_mask,
            "area":      int(np.sum(full_mask)),
            "area_frac": round(area_frac, 4),
            "stability": round(float(m.get("stability_score", 0)), 3),
            "iou":       round(float(m.get("predicted_iou", 0)), 3),
        })

    # Remove overlapping detections — keep most stable
    colonies = nms(colonies, iou_threshold=0.5)
    print(f"  After NMS: {len(colonies)} colonies")
    return colonies


def nms(detections, iou_threshold=0.5):
    """Remove duplicate detections keeping most stable."""
    if not detections:
        return detections
    detections = sorted(detections, key=lambda d: -d["stability"])
    kept = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        dup = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k["bbox"]
            ix1 = max(x1, kx1); iy1 = max(y1, ky1)
            ix2 = min(x2, kx2); iy2 = min(y2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2-ix1) * (iy2-iy1)
            union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter + 1e-6
            if inter / union > iou_threshold:
                dup = True
                break
        if not dup:
            kept.append(det)
    return kept


# ── EfficientNet-B3 Option A Classifier ───────────────────────────────────────

CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize(int(224 * 1.1)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_classifier(checkpoint_path, device):
    """Load trained EfficientNet-B3 Option A classifier."""
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {checkpoint_path}\n"
            "Download the trained checkpoint and place it in models/, or pass --clf_checkpoint."
        )

    print(f"Loading EfficientNet-B3 Option A from {checkpoint_path} ...")
    ckpt        = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt.get("class_names", SPECIES_LIST)
    num_classes = len(class_names)

    model = timm.create_model("efficientnet_b3", pretrained=False,
                               num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    val_f1 = ckpt.get("val_f1", 0)
    print(f"Classifier loaded — {num_classes} classes, val F1={val_f1:.3f}")
    return model, class_names


def make_option_a_crop(img_rgb, mask, bbox):
    """
    Option A crop: extract bbox region, zero out non-mask pixels.
    This matches the training distribution — classifier trained on
    coral-only pixels against black background.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    crop      = img_rgb[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]
    crop[~mask_crop] = 0
    return crop


def classify_colony(clf, class_names, crop_rgb, device):
    """Classify one Option A crop. Returns (species, confidence, top3)."""
    if crop_rgb is None or min(crop_rgb.shape[:2]) < 10:
        return "Unknown", 0.0, []

    tensor = CLASSIFIER_TRANSFORM(
        Image.fromarray(crop_rgb)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(clf(tensor), dim=1)[0].cpu().numpy()

    best_idx  = int(np.argmax(probs))
    best_conf = float(probs[best_idx])
    best_name = class_names[best_idx] if class_names else str(best_idx)
    top3      = [(class_names[i] if class_names else str(i), float(c))
                 for i, c in sorted(enumerate(probs), key=lambda x: -x[1])[:3]]
    return best_name, best_conf, top3


# ── Output Image Building ─────────────────────────────────────────────────────

def build_outputs(img_rgb, detections, conf_threshold, overlay_alpha):
    """Build species map, overlay, and label map from detections."""
    h, w          = img_rgb.shape[:2]
    colour_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    label_map     = np.zeros((h, w), dtype=np.uint8)

    for i, det in enumerate(detections):
        if det.get("rejected"):
            continue
        sp  = det["species"] if det["confidence"] >= conf_threshold else "Unknown"
        rgb = SPECIES_COLOURS.get(sp, SPECIES_COLOURS["Unknown"])
        colour_canvas[det["mask"]] = rgb
        label_map[det["mask"]]     = SPECIES_LIST.index(sp) + 1 \
                                     if sp in SPECIES_LIST else 0

    coral_px    = colour_canvas.any(axis=2)
    overlay     = img_rgb.copy().astype(float)
    overlay[coral_px] = (
        (1 - overlay_alpha) * img_rgb.astype(float)[coral_px] +
        overlay_alpha * colour_canvas.astype(float)[coral_px]
    )
    overlay     = np.clip(overlay, 0, 255).astype(np.uint8)
    species_map = img_rgb.copy()
    species_map[coral_px] = colour_canvas[coral_px]

    return species_map, overlay, label_map


def draw_labels(img_bgr, detections, conf_threshold):
    """Draw colony outlines and species labels on image."""
    out   = img_bgr.copy()
    h, w  = out.shape[:2]
    scale = max(w, h) / 1920
    font  = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        if det.get("rejected"):
            continue
        sp   = det["species"] if det["confidence"] >= conf_threshold else "Unknown"
        conf = det["confidence"]
        mask = det["mask"].astype(np.uint8)
        rgb  = SPECIES_COLOURS.get(sp, SPECIES_COLOURS["Unknown"])
        bgr  = (rgb[2], rgb[1], rgb[0])

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, bgr, max(1, int(1.5*scale)))

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
        bright = 0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]
        tc = (20,20,20) if bright > 160 else (255,255,255)
        cv2.putText(out, label, (lx+2, ly), font, fs, tc, 1, cv2.LINE_AA)

    return out


def add_legend(img_bgr, detections, conf_threshold):
    """Add species legend panel."""
    painted = {}
    for d in detections:
        if d.get("rejected"):
            continue
        sp = d["species"] if d["confidence"] >= conf_threshold else "Unknown"
        if sp not in painted:
            painted[sp] = SPECIES_COLOURS.get(sp, SPECIES_COLOURS["Unknown"])

    if not painted:
        return img_bgr

    h    = img_bgr.shape[0]
    pad  = 12; lh = 28
    ph   = max(h, pad*2 + 60 + len(painted)*lh)
    panel = np.full((ph, 320, 3), 30, dtype=np.uint8)
    font  = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, "SAM2 + EfficientNet-B3",
                (pad, pad+16), font, 0.45, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(panel, "Unpainted = background/substrate",
                (pad, pad+32), font, 0.38, (140,140,140), 1, cv2.LINE_AA)

    for i, (sp, rgb) in enumerate(sorted(painted.items())):
        y   = pad + 46 + i * lh
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(panel, (pad, y-12), (pad+16, y+4), bgr, -1)
        name = sp.replace("_"," ")
        if len(name) > 27: name = name[:25] + "…"
        cv2.putText(panel, name, (pad+22, y+2),
                    font, 0.42, (210,210,210), 1, cv2.LINE_AA)

    if h < panel.shape[0]:
        img_bgr = np.pad(img_bgr,
                         ((0, panel.shape[0]-h), (0,0), (0,0)), mode="edge")
    return np.concatenate([img_bgr, panel[:img_bgr.shape[0]]], axis=1)


# ── Process One Image ─────────────────────────────────────────────────────────

def process_image(img_path, generator, clf, class_names,
                  output_dir, args, device):
    """Full pipeline: SAM2 segmentation → Option A classification → save outputs."""
    print(f"\n{'─'*60}")
    print(f"Image: {img_path.name}")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print("  ERROR: could not open image")
        return []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    print(f"  Size: {w}×{h}")

    # Step 1: SAM2 finds coral colonies
    t0       = time.time()
    colonies = find_colonies_sam2(
        generator, img_rgb, args.min_area, args.max_area)
    seg_time = time.time() - t0

    if not colonies:
        print(f"  No colonies found ({seg_time:.1f}s) — try lowering --min_area")
        return []

    # Step 2: Classify each colony
    detections = []
    for idx, col in enumerate(colonies):
        crop = make_option_a_crop(img_rgb, col["mask"], col["bbox"])
        species, conf, top3 = classify_colony(clf, class_names, crop, device)

        # Not-coral rejection: if confidence below threshold treat as substrate
        rejected = conf < args.not_coral_threshold

        det = {
            "id":         idx + 1,
            "species":    species,
            "confidence": conf,
            "top3":       top3,
            "bbox":       col["bbox"],
            "area":       col["area"],
            "area_frac":  col["area_frac"],
            "stability":  col["stability"],
            "iou":        col["iou"],
            "rejected":   rejected,
            "mask":       col["mask"],
        }
        detections.append(det)

        x1, y1, x2, y2 = col["bbox"]
        status = "REJECTED (not coral)" if rejected else \
                 ("✓" if conf >= args.conf_threshold else "~")
        print(f"  [{idx+1:2d}] {status}  "
              f"{species.replace('_',' '):<33}  "
              f"{conf:.1%}  area={col['area_frac']:.1%}")

    clf_time = time.time() - t0 - seg_time
    print(f"\n  SAM2: {seg_time:.1f}s  |  Classification: {clf_time:.1f}s")

    # Coverage stats (excluding rejected)
    accepted = [d for d in detections if not d["rejected"]]
    total_coral = sum(d["area"] for d in accepted
                      if d["confidence"] >= args.conf_threshold)
    coverage = total_coral / (h * w) * 100
    n_rejected = len(detections) - len(accepted)
    print(f"  Accepted: {len(accepted)}  Rejected (not-coral): {n_rejected}")
    print(f"  Coral coverage: {coverage:.1f}%")

    sp_counts = Counter(d["species"] for d in accepted
                        if d["confidence"] >= args.conf_threshold)
    for sp, n in sp_counts.most_common():
        area = sum(d["area"] for d in accepted
                   if d["species"]==sp and
                   d["confidence"]>=args.conf_threshold)
        print(f"    {sp.replace('_',' '):<35}  {n} colonies  "
              f"{area/(h*w)*100:.1f}% cover")

    # Build and save output images
    species_map, overlay, label_map = build_outputs(
        img_rgb, detections, args.conf_threshold, args.overlay_alpha)

    overlay_bgr  = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    outlined     = draw_labels(overlay_bgr, detections, args.conf_threshold)
    sp_map_bgr   = cv2.cvtColor(species_map, cv2.COLOR_RGB2BGR)
    sp_out       = add_legend(sp_map_bgr, detections, args.conf_threshold)
    ov_out       = add_legend(outlined, detections, args.conf_threshold)

    stem = img_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_species_map.jpg"),
                sp_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(output_dir / f"{stem}_overlay.jpg"),
                ov_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(output_dir / f"{stem}_label_map.png"), label_map)

    # Save JSON (no mask arrays)
    with open(output_dir / f"{stem}_results.json", "w") as f:
        json.dump({
            "image":              img_path.name,
            "segmentation":       "SAM2 hiera_base_plus",
            "classification":     "EfficientNet-B3 Option A",
            "size":               {"width": w, "height": h},
            "coral_coverage":     round(coverage, 2),
            "n_colonies":         len(accepted),
            "n_rejected":         n_rejected,
            "detections": [
                {k: v for k, v in d.items() if k != "mask"}
                for d in detections if not d["rejected"]
            ],
        }, f, indent=2)

    print(f"\n  Saved: {stem}_species_map.jpg | _overlay.jpg | _label_map.png")
    return [d for d in detections if not d["rejected"]]


# ── Summary CSV ───────────────────────────────────────────────────────────────

def save_summary(all_dets, output_dir, conf_threshold):
    import csv
    rows = []
    for img_name, dets in all_dets.items():
        for d in dets:
            rows.append({
                "image":           img_name,
                "id":              d["id"],
                "species":         d["species"].replace("_"," "),
                "confidence":      round(d["confidence"], 3),
                "above_threshold": d["confidence"] >= conf_threshold,
                "x1": d["bbox"][0], "y1": d["bbox"][1],
                "x2": d["bbox"][2], "y2": d["bbox"][3],
                "area_pixels":     d["area"],
                "area_fraction":   d["area_frac"],
                "stability":       d["stability"],
                "iou":             d["iou"],
            })
    if not rows:
        return
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary: {csv_path}")
    above = [r["species"] for r in rows if r["above_threshold"]]
    if above:
        print(f"Species detected (>={conf_threshold:.0%} confidence):")
        for sp, n in Counter(above).most_common():
            print(f"  {sp:<35}  {n}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    print(f"Device         : {device}")
    if device == "cuda":
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Segmentation   : SAM2 hiera_base_plus")
    print(f"Classification : EfficientNet-B3 Option A")
    print(f"Not-coral filter: conf < {args.not_coral_threshold:.0%} → rejected")

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
            if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}
        )
    else:
        print(f"ERROR: {input_path} not found"); return

    if not img_paths:
        print("No images found."); return

    # Resume support
    already_done = {p.stem.replace("_results","")
                    for p in output_dir.glob("*_results.json")}
    to_process   = [p for p in img_paths if p.stem not in already_done]
    if already_done:
        print(f"Resuming — skipping {len(already_done)} already processed")
    print(f"Images to process: {len(to_process)}\n")

    # Load models
    generator          = load_sam2(args.sam2_weights, args.sam2_config, device)
    clf, class_names   = load_classifier(args.clf_checkpoint, device)

    # Process
    all_dets = {}
    for img_path in to_process:
        dets = process_image(img_path, generator, clf, class_names,
                             output_dir, args, device)
        if dets:
            all_dets[img_path.name] = dets

    save_summary(all_dets, output_dir, args.conf_threshold)

    total     = sum(len(d) for d in all_dets.values())
    confident = sum(1 for dets in all_dets.values()
                    for d in dets if d["confidence"] >= args.conf_threshold)
    print(f"\n{'='*60}")
    print(f"Images processed     : {len(all_dets)}")
    print(f"Total colonies       : {total}")
    print(f"Confidently labelled : {confident} (>={args.conf_threshold:.0%})")
    print(f"Results saved to     : {output_dir}")


if __name__ == "__main__":
    main()