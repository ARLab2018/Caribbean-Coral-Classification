"""
Coral Reef Species Mapping — Option A Inference
=================================================
This script produces a pixel-level species map of your underwater images.

How it works:
    1. MobileSAM scans the image and draws a precise polygon mask around
       every coral colony it finds (not just a bounding box — the actual shape).

    2. For each colony mask, the background is blacked out so EfficientNet
       only sees the coral pixels. This matches exactly how the Option A
       model was trained (on polygon-masked crops with zeroed backgrounds).

    3. EfficientNet-B3 classifies each masked colony to species level.

    4. Every pixel inside each colony mask gets painted with the species colour,
       producing a full pixel-level species map overlaid on the original photo.

This is the closest to true semantic segmentation we can produce with our
current models — each output pixel is either background or a named species.

How to run from the repository root:
    conda activate coralnet10
    python src/inference/infer_mobilesam_effnet.py --input examples/sample_input.jpg
    python src/inference/infer_mobilesam_effnet.py --input /path/to/folder/

Model files are not committed to GitHub. By default, place them under:
    models/checkpoints_option_a/best_model.pt
    models/mobile_sam.pt
Or pass explicit paths with --checkpoint and --sam_checkpoint.

What you get for each image:
    <name>_species_map.jpg     the original photo with species painted on each colony
    <name>_overlay.jpg         semi-transparent overlay so you can see the coral underneath
    <name>_label_map.png       pure pixel label map (each colour = one species, no photo)
    <name>_results.json        all detections with mask polygons and species labels
    summary.csv                spreadsheet of all detections across all images

Arguments:
    --input           image file or folder
    --output          where to save results (default: outputs/inference/mobilesam_effnet)
    --checkpoint      Option A checkpoint (default: models/checkpoints_option_a/best_model.pt)
    --sam_checkpoint  MobileSAM weights (default: models/mobile_sam.pt)
    --min_area        min colony size as fraction of image (default: 0.003)
    --max_area        max colony size as fraction of image (default: 0.50)
    --conf_threshold  min confidence to label a colony (default: 0.40)
    --overlay_alpha   transparency of colour overlay on the photo (default: 0.5)
"""

import argparse
import json
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


# ── Step 1: Arguments ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pixel-level coral species mapping using Option A model"
    )
    p.add_argument("--input",          required=True,
                   help="Image file or folder of images")
    p.add_argument("--output",         default=str(repo_path("outputs", "inference", "mobilesam_effnet")),
                   help="Where to save results")
    p.add_argument("--checkpoint",     default=str(MODEL_DIR / "checkpoints_option_a" / "best_model.pt"),
                   help="Path to the Option A EfficientNet-B3 checkpoint")
    p.add_argument("--sam_checkpoint", default=str(MODEL_DIR / "mobile_sam.pt"),
                   help="Path to MobileSAM weights")
    p.add_argument("--min_area",       type=float, default=0.003,
                   help="Ignore colonies smaller than this fraction of the image")
    p.add_argument("--max_area",       type=float, default=0.15,
                   help="Ignore regions larger than this fraction of the image")
    p.add_argument("--conf_threshold", type=float, default=0.40,
                   help="Min confidence to assign a species label (default: 0.40)")
    p.add_argument("--overlay_alpha",  type=float, default=0.50,
                   help="Transparency of colour overlay: 0=invisible 1=fully opaque")
    p.add_argument("--device",         default="auto")
    return p.parse_args()


# ── Step 2: Species colours ───────────────────────────────────────────────────
# Each species gets a unique RGB colour for the pixel map.
# The same colours are used consistently across all output images.

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

# Integer label index for the pure label map (0 = background)
SPECIES_LIST = [
    "Acropora_tenuifolia", "Agaricia_agaricites", "Colpophyllia_natans",
    "Lobophyllia_spp", "Madracis_auretenra", "Madracis_mirabilis",
    "Meandrina_meandrites", "Millepora_spp", "Montastraea_cavernosa",
    "Orbicella_annularis", "Orbicella_faveolata", "Orbicella_franksi",
    "Porites_astreoides", "Porites_porites", "Pseudodiploria_strigosa",
    "Siderastrea_siderea", "Stephanocoenia_intersepta",
]
SPECIES_TO_IDX = {s: i + 1 for i, s in enumerate(SPECIES_LIST)}  # 0 = background


# ── Step 3: MobileSAM — finds colony shapes ───────────────────────────────────

SAM_MAX_DIM = 1024   # resize images larger than this before SAM inference


def resize_for_sam(img_rgb):
    """
    Shrink large images so SAM fits in GPU memory.
    Returns the resized image and a scale factor to restore coordinates.
    """
    h, w  = img_rgb.shape[:2]
    scale = max(w, h) / SAM_MAX_DIM
    if scale <= 1.0:
        return img_rgb, 1.0
    resized = cv2.resize(img_rgb, (int(w / scale), int(h / scale)),
                         interpolation=cv2.INTER_AREA)
    return resized, scale


def load_sam(checkpoint_path, device, min_area_frac, max_area_frac):
    """Load MobileSAM automatic mask generator."""
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"MobileSAM checkpoint not found: {checkpoint_path}\n"
            "Download the weights and place them in models/, or pass --sam_checkpoint."
        )

    from x_segment_anything import sam_model_registry
    print(f"Loading MobileSAM on {device} ...")
    sam = sam_model_registry["vit_t"](checkpoint=str(checkpoint_path))
    sam.to(device)
    sam.eval()

    class SamWrapper:
        def __init__(self, model):
            self.sam            = model
            self._min_area_frac = min_area_frac
            self._max_area_frac = max_area_frac

        def build_generator(self, w, h):
            from x_segment_anything import SamAutomaticMaskGenerator
            # Scale sampling density to image size — avoids over-segmentation
            points = max(8, min(32, min(w, h) // 60))
            return SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=points,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=200,
            )

    print("MobileSAM loaded.")
    return SamWrapper(sam)


def remove_overlapping_detections(detections, iou_threshold=0.5):
    """
    Remove duplicate detections of the same colony.
    If two masks overlap by more than 50%, keep only the more stable one.
    """
    if not detections:
        return detections
    detections = sorted(detections, key=lambda d: -d["stability"])
    kept = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        duplicate = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k["bbox"]
            ix1 = max(x1, kx1); iy1 = max(y1, ky1)
            ix2 = min(x2, kx2); iy2 = min(y2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = ((x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter + 1e-6)
            if inter / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def find_colony_masks(sam_wrapper, img_rgb):
    """
    Use MobileSAM to find all coral colonies in the image.
    Unlike a bounding-box detector, this returns the precise pixel mask
    for each colony — the exact set of pixels that belong to it.
    Returns a list of detections, each with a boolean mask array.
    """
    h, w     = img_rgb.shape[:2]
    img_area = h * w

    # Resize for SAM
    sam_img, scale = resize_for_sam(img_rgb)
    sh, sw         = sam_img.shape[:2]
    if scale > 1.0:
        print(f"  Resized for SAM: {w}x{h} -> {sw}x{sh} (scale={scale:.2f})")

    generator = sam_wrapper.build_generator(sw, sh)
    raw_masks  = generator.generate(sam_img)

    detections = []
    for mask in raw_masks:
        area_frac = mask["area"] / img_area
        if area_frac < sam_wrapper._min_area_frac:
            continue
        if area_frac > sam_wrapper._max_area_frac:
            continue
        # Skip masks spanning full image width/height — these are background, not coral
        _bx, _by, _bw, _bh = mask["bbox"]
        if _bw >= img_rgb.shape[1] * 0.90 or _bh >= img_rgb.shape[0] * 0.90:
            continue

        # Scale bounding box back to original image coordinates
        x, y, bw, bh = mask["bbox"]
        x1 = int(x * scale);        y1 = int(y * scale)
        x2 = int((x + bw) * scale); y2 = int((y + bh) * scale)

        # Scale the boolean mask back to original image size
        # (SAM generated it at the reduced resolution)
        small_mask = mask["segmentation"].astype(np.uint8)
        full_mask  = cv2.resize(small_mask, (w, h),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

        detections.append({
            "bbox":      (x1, y1, x2, y2),
            "mask":      full_mask,          # True for pixels inside the colony
            "area":      int(np.sum(full_mask)),
            "area_frac": round(float(np.sum(full_mask)) / img_area, 4),
            "stability": round(float(mask.get("stability_score", 0)), 3),
            "iou":       round(float(mask.get("predicted_iou", 0)), 3),
        })

    before     = len(detections)
    detections = remove_overlapping_detections(detections)
    print(f"  SAM: {before} regions -> {len(detections)} after overlap removal")
    return detections


# ── Step 4: EfficientNet-B3 classifier ───────────────────────────────────────

# Image pre-processing must match Option A training exactly:
# the crop has background zeroed out, then resized to 224x224
CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize(int(224 * 1.1)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_classifier(checkpoint_path, device):
    """Load the trained Option A EfficientNet-B3 classifier."""
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {checkpoint_path}\n"
            "Download the trained checkpoint and place it in models/, or pass --checkpoint."
        )

    print(f"Loading Option A classifier from {checkpoint_path} ...")
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
    Produce the same kind of crop the Option A model was trained on:
      1. Extract the bounding box region from the image
      2. Zero out all pixels that are NOT part of the coral mask
    This ensures the classifier sees the same input format it was trained on.
    """
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None

    # Crop the image and mask to the bounding box
    crop      = img_rgb[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]

    # Zero out background pixels (turn them black)
    crop[~mask_crop] = 0

    return crop


def classify_colony(model, class_names, crop_rgb, device):
    """
    Classify a single Option-A-style crop (background zeroed).
    Returns predicted species, confidence (0-1), and top-3 guesses.
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


# ── Step 5: Build output images ───────────────────────────────────────────────

def build_species_map(img_rgb, detections, conf_threshold, overlay_alpha):
    """
    Paint each detected colony in its species colour.

    Produces three outputs:
      species_map  : original photo with solid colour painted over each colony
      overlay      : semi-transparent version (you can see the coral underneath)
      label_map    : pure integer map (0=background, 1-17=species index)
    """
    h, w = img_rgb.shape[:2]

    # Start with a black canvas for the colour map
    colour_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Integer label map — 0 means background
    label_map = np.zeros((h, w), dtype=np.uint8)

    # Track which colonies were painted (for the legend)
    painted_species = {}

    for det in detections:
        species    = det["species"]
        confidence = det["confidence"]
        mask       = det["mask"]

        # Only paint if we are confident enough
        if confidence < conf_threshold:
            species = "Unknown"

        rgb = SPECIES_COLOURS.get(species, SPECIES_COLOURS["Unknown"])

        # Paint this colony's pixels on the colour canvas
        colour_canvas[mask] = rgb

        # Write species index into the label map
        label_idx = SPECIES_TO_IDX.get(species, 0)
        label_map[mask] = label_idx

        if confidence >= conf_threshold:
            painted_species[species] = rgb

    # Build overlay: blend original photo with colour canvas
    img_float    = img_rgb.astype(float)
    colour_float = colour_canvas.astype(float)

    # Where there is colour (i.e. a detected colony), blend with the photo
    coral_pixels = colour_canvas.any(axis=2)
    overlay      = img_rgb.copy().astype(float)
    overlay[coral_pixels] = (
        (1 - overlay_alpha) * img_float[coral_pixels] +
        overlay_alpha       * colour_float[coral_pixels]
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Build species map: full colour where coral, original photo elsewhere
    species_map = img_rgb.copy()
    species_map[coral_pixels] = colour_canvas[coral_pixels]

    return species_map, overlay, label_map, painted_species


def draw_colony_outlines(img_bgr, detections, conf_threshold):
    """
    Draw a thin outline around each detected colony and add a small label.
    Used on the overlay image so you can see individual colony boundaries.
    """
    out   = img_bgr.copy()
    h, w  = out.shape[:2]
    scale = max(w, h) / 1920
    font  = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        species    = det["species"]
        confidence = det["confidence"]
        mask       = det["mask"].astype(np.uint8)

        if confidence < conf_threshold:
            species = "Unknown"

        rgb = SPECIES_COLOURS.get(species, SPECIES_COLOURS["Unknown"])
        bgr = (rgb[2], rgb[1], rgb[0])

        # Find the contour of the mask and draw it
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, bgr,
                         max(1, int(1.5 * scale)))

        # Add a small label near the top of each colony
        x1, y1, x2, y2 = det["bbox"]
        label      = (f"{species.replace('_',' ')} {confidence:.0%}"
                      if confidence >= conf_threshold
                      else f"Coral ({confidence:.0%})")
        font_scale = max(0.3, 0.4 * scale)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        lx = max(0, min(x1, w - tw - 4))
        ly = max(th + 4, y1 - 4)
        cv2.rectangle(out, (lx, ly - th - 2), (lx + tw + 4, ly + 2),
                      bgr, -1)
        brightness  = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        text_colour = (20, 20, 20) if brightness > 160 else (255, 255, 255)
        cv2.putText(out, label, (lx + 2, ly), font, font_scale,
                    text_colour, 1, cv2.LINE_AA)

    return out


def build_legend_panel(painted_species, image_height):
    """Build a dark legend panel listing all species found in the image."""
    padding   = 12
    line_h    = 28
    panel_h   = max(image_height, padding * 2 + 40 + len(painted_species) * line_h)
    panel     = np.full((panel_h, 300, 3), 30, dtype=np.uint8)
    font      = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, "Species map legend",
                (padding, padding + 18), font, 0.55, (230, 230, 230),
                1, cv2.LINE_AA)
    cv2.putText(panel, "Unpainted = background/substrate",
                (padding, padding + 36), font, 0.35, (160, 160, 160),
                1, cv2.LINE_AA)

    for i, (species, rgb) in enumerate(sorted(painted_species.items())):
        y   = padding + 48 + i * line_h
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(panel, (padding, y - 12), (padding + 18, y + 6), bgr, -1)
        name = species.replace("_", " ")
        if len(name) > 28:
            name = name[:26] + "…"
        cv2.putText(panel, name, (padding + 24, y + 2),
                    font, 0.42, (215, 215, 215), 1, cv2.LINE_AA)

    return panel


def attach_legend(img_bgr, painted_species):
    """Join the legend panel to the right of an image."""
    if not painted_species:
        return img_bgr
    panel = build_legend_panel(painted_species, img_bgr.shape[0])
    h     = img_bgr.shape[0]
    if h < panel.shape[0]:
        img_bgr = np.pad(img_bgr,
                         ((0, panel.shape[0] - h), (0, 0), (0, 0)),
                         mode="edge")
    return np.concatenate([img_bgr, panel[:img_bgr.shape[0]]], axis=1)


# ── Step 6: Process one image ─────────────────────────────────────────────────

def process_image(img_path, sam_wrapper, classifier, class_names,
                  output_dir, conf_threshold, overlay_alpha, device):
    """Run the full pipeline on a single image and save all outputs."""
    print(f"\n{'─' * 60}")
    print(f"Image: {img_path.name}")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  ERROR: could not open image")
        return []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    print(f"  Size: {w}x{h} pixels")

    # Find colony masks with SAM
    t0         = time.time()
    detections = find_colony_masks(sam_wrapper, img_rgb)
    print(f"  Detection time: {time.time() - t0:.1f}s")

    if not detections:
        print("  No colonies found — try lowering --min_area")
        return []

    # Classify each colony using Option A style (background zeroed)
    print(f"  Classifying {len(detections)} colonies ...")
    for idx, det in enumerate(detections):
        crop = make_option_a_crop(img_rgb, det["mask"], det["bbox"])
        sp, conf, top3 = classify_colony(classifier, class_names, crop, device)
        det["id"]         = idx + 1
        det["species"]    = sp
        det["confidence"] = conf
        det["top3"]       = top3

        flag = "✓" if conf >= conf_threshold else "~"
        print(f"  [{idx+1:2d}] {flag}  "
              f"{sp.replace('_',' '):<33}  "
              f"{conf:.1%}  "
              f"(covers {det['area_frac']:.1%} of image)")

    # Build output images
    species_map, overlay, label_map, painted = build_species_map(
        img_rgb, detections, conf_threshold, overlay_alpha)

    # Draw colony outlines on the overlay
    overlay_bgr  = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    outlined     = draw_colony_outlines(overlay_bgr, detections, conf_threshold)

    # Add legend to both output images
    species_map_bgr = cv2.cvtColor(species_map, cv2.COLOR_RGB2BGR)
    species_map_out = attach_legend(species_map_bgr, painted)
    overlay_out     = attach_legend(outlined, painted)

    # Save outputs
    stem = img_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_species_map.jpg"),
                species_map_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(output_dir / f"{stem}_overlay.jpg"),
                overlay_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(output_dir / f"{stem}_label_map.png"), label_map)

    print(f"\n  Saved:")
    print(f"    {stem}_species_map.jpg  "
          f"(solid colour per species)")
    print(f"    {stem}_overlay.jpg      "
          f"(transparent overlay with outlines)")
    print(f"    {stem}_label_map.png    "
          f"(integer label per pixel — 0=background)")

    # Coverage statistics
    total_coral_px = sum(d["area"] for d in detections
                         if d["confidence"] >= conf_threshold)
    coral_coverage = total_coral_px / (h * w) * 100
    print(f"\n  Coral coverage : {coral_coverage:.1f}% of image")
    if painted:
        print(f"  Species found  :")
        sp_counts = Counter(d["species"] for d in detections
                            if d["confidence"] >= conf_threshold)
        for sp, n in sp_counts.most_common():
            sp_area = sum(d["area"] for d in detections
                          if d["species"] == sp and
                          d["confidence"] >= conf_threshold)
            sp_pct  = sp_area / (h * w) * 100
            print(f"    {sp.replace('_',' '):<35}  "
                  f"{n} colonies  {sp_pct:.1f}% cover")

    # Save JSON (exclude numpy mask arrays — not JSON serialisable)
    json_data = {
        "image":           img_path.name,
        "size":            {"width": w, "height": h},
        "coral_coverage":  round(coral_coverage, 2),
        "detections": [
            {k: v for k, v in d.items() if k != "mask"}
            for d in detections
        ],
    }
    with open(output_dir / f"{stem}_results.json", "w") as f:
        json.dump(json_data, f, indent=2)

    return detections


# ── Step 7: Summary CSV ───────────────────────────────────────────────────────

def save_summary_csv(all_detections, output_dir, conf_threshold):
    """Write one CSV row per detection across all processed images."""
    import csv
    rows = []
    for img_name, dets in all_detections.items():
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
            })
    if not rows:
        return
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary spreadsheet saved: {csv_path}")

    confident = [r["species"] for r in rows if r["above_threshold"]]
    if confident:
        print(f"\nSpecies breakdown (above {conf_threshold:.0%} confidence):")
        for sp, n in Counter(confident).most_common():
            print(f"  {sp:<35}  {n} detection(s)")


# ── Step 8: Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    print(f"Device  : {device}")
    if device == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")

    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results : {output_dir}")

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
    print(f"Found {len(img_paths)} image(s) to process\n")

    # Load models
    sam_wrapper             = load_sam(args.sam_checkpoint, device,
                                       args.min_area, args.max_area)
    classifier, class_names = load_classifier(args.checkpoint, device)

    # Process all images — skip any already processed (resume support)
    already_done = {p.stem.replace("_results", "")
                    for p in output_dir.glob("*_results.json")}
    skipped      = sum(1 for p in img_paths if p.stem in already_done)
    if skipped:
        print(f"Resuming — skipping {skipped} already processed images")

    all_detections = {}
    for img_path in img_paths:
        if img_path.stem in already_done:
            continue   # already processed in a previous run
        dets = process_image(
            img_path, sam_wrapper, classifier, class_names,
            output_dir, args.conf_threshold, args.overlay_alpha, device
        )
        if dets:
            all_detections[img_path.name] = dets

    # Save combined summary
    save_summary_csv(all_detections, output_dir, args.conf_threshold)

    # Final stats
    total     = sum(len(d) for d in all_detections.values())
    confident = sum(1 for dets in all_detections.values()
                    for d in dets if d["confidence"] >= args.conf_threshold)
    print(f"\n{'='*60}")
    print(f"Images processed    : {len(all_detections)}")
    print(f"Total colonies found: {total}")
    print(f"Confidently labelled: {confident}  "
          f"(>= {args.conf_threshold:.0%})")
    print(f"Results saved to    : {output_dir}")


if __name__ == "__main__":
    main()
