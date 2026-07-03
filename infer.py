"""
Coral Reef Species Detector
============================
This script takes your own underwater photos, finds all the coral colonies
in each photo using MobileSAM (an AI segmentation model), draws a coloured
bounding box around each one, and labels it with the coral species name and
a confidence percentage using our trained EfficientNet-B3 classifier.

How to run:
    conda activate coralnet10

    # Run on a single image:
    python infer.py --input /path/to/photo.jpg

    # Run on a whole folder of images:
    python infer.py --input /path/to/folder/

    # Adjust sensitivity (fewer but more confident detections):
    python infer.py --input /path/to/photo.jpg --min_area 0.01 --conf_threshold 0.6

What you get for each image:
    <name>_annotated.jpg   the photo with boxes and labels drawn on it
    <name>_results.json    all detections in a machine-readable format
    summary.csv            a spreadsheet of all detections across all images

Arguments you can adjust:
    --min_area        minimum colony size as a fraction of the image (default 0.003 = 0.3%)
                      increase this (e.g. 0.01) if you get too many tiny detections
    --max_area        maximum colony size as a fraction of the image (default 0.50)
    --conf_threshold  minimum confidence to show a species label (default 0.40 = 40%)
                      detections below this show as "Coral (xx%)" without a species name
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


# ── Step 1: Read command-line arguments ───────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect and classify coral species in underwater images"
    )
    p.add_argument("--input",           required=True,
                   help="Image file or folder of images to analyse")
    p.add_argument("--output",          default=str(Path.home() / "Independent_study" / "inference_output"),
                   help="Where to save the annotated images and results")
    p.add_argument("--checkpoint",      default=str(Path.home() / "Independent_study" / "checkpoints_option_f" / "best_model.pt"),
                   help="Path to the trained EfficientNet-B3 model checkpoint")
    p.add_argument("--sam_checkpoint",  default=str(Path.home() / "Independent_study" / "mobile_sam.pt"),
                   help="Path to MobileSAM weights")
    p.add_argument("--min_area",        type=float, default=0.003,
                   help="Ignore colonies smaller than this fraction of the image (default: 0.003)")
    p.add_argument("--max_area",        type=float, default=0.50,
                   help="Ignore regions larger than this fraction of the image (default: 0.50)")
    p.add_argument("--conf_threshold",  type=float, default=0.40,
                   help="Only show species label if confidence is above this value (default: 0.40)")
    p.add_argument("--device",          default="auto",
                   help="cuda, cpu, or auto (default: auto-detect GPU)")
    return p.parse_args()


# ── Step 2: Colour palette — each species gets its own colour ─────────────────

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
    "Unknown":                   (180, 180, 180),  # grey for low confidence
}


# ── Step 3: Load MobileSAM — the colony finder ────────────────────────────────

# MobileSAM is too slow to process large high-resolution images whole.
# We resize anything bigger than 1024px down to 1024px for SAM,
# then scale the detected bounding boxes back up to the original size.
SAM_MAX_DIM = 1024


def resize_for_sam(img_rgb):
    """
    Shrink large images so SAM can process them without running out of GPU memory.
    Returns the resized image and a scale factor to convert boxes back to original coords.
    """
    h, w   = img_rgb.shape[:2]
    scale  = max(w, h) / SAM_MAX_DIM
    if scale <= 1.0:
        return img_rgb, 1.0   # already small enough, no resize needed
    new_w  = int(w / scale)
    new_h  = int(h / scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def load_sam(checkpoint_path, device, min_area_frac, max_area_frac):
    """Load MobileSAM and return a wrapper that builds generators per image."""
    from x_segment_anything import sam_model_registry

    print(f"Loading MobileSAM on {device} ...")
    sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval()

    class SamWrapper:
        """Holds the SAM model and creates a fresh detector for each image size."""
        def __init__(self, sam_model):
            self.sam            = sam_model
            self._min_area_frac = min_area_frac
            self._max_area_frac = max_area_frac

        def build_generator(self, img_width, img_height):
            """
            Build an automatic mask generator scaled to this image's resolution.
            Smaller images need fewer sample points — using too many on a small
            image produces hundreds of overlapping detections.
            """
            from x_segment_anything import SamAutomaticMaskGenerator
            # Rule of thumb: roughly one sampling point per 60x60 pixel region
            points = max(8, min(32, min(img_width, img_height) // 60))
            return SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=points,
                pred_iou_thresh=0.86,           # filter out low-quality masks
                stability_score_thresh=0.92,    # filter out unstable masks
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=200,        # ignore tiny fragments
            )

    print("MobileSAM loaded.")
    return SamWrapper(sam)


def remove_overlapping_boxes(boxes, overlap_threshold=0.5):
    """
    When two bounding boxes overlap heavily, keep only the better one.
    This prevents the same coral colony from being detected multiple times.
    Boxes are ranked by their stability score (how consistently SAM drew them).
    """
    if not boxes:
        return boxes

    # Sort best first
    boxes  = sorted(boxes, key=lambda b: -b["stability"])
    kept   = []

    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        too_similar     = False

        for kept_box in kept:
            kx1, ky1, kx2, ky2 = kept_box["bbox"]

            # Calculate how much the two boxes overlap
            inter_x1    = max(x1, kx1);    inter_y1 = max(y1, ky1)
            inter_x2    = min(x2, kx2);    inter_y2 = min(y2, ky2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue   # no overlap at all

            inter_area  = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            union_area  = ((x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1)
                           - inter_area + 1e-6)
            iou         = inter_area / union_area

            if iou > overlap_threshold:
                too_similar = True
                break

        if not too_similar:
            kept.append(box)

    return kept


def find_colonies(sam_wrapper, img_rgb):
    """
    Use MobileSAM to find all coral colonies in an image.
    Returns a list of detections, each with a bounding box and size info.
    """
    h, w     = img_rgb.shape[:2]
    img_area = h * w

    # Resize to SAM-friendly resolution
    sam_img, scale = resize_for_sam(img_rgb)
    sh, sw         = sam_img.shape[:2]
    if scale > 1.0:
        print(f"  Resized for SAM: {w}x{h} -> {sw}x{sh} (scale={scale:.2f})")

    # Run SAM
    generator = sam_wrapper.build_generator(sw, sh)
    raw_masks  = generator.generate(sam_img)

    colonies = []
    for mask in raw_masks:
        # Filter by size — skip things too small or too large to be a colony
        area_fraction = mask["area"] / img_area
        if area_fraction < sam_wrapper._min_area_frac:
            continue
        if area_fraction > sam_wrapper._max_area_frac:
            continue

        # SAM gives bbox as [x, y, width, height] — convert to [x1, y1, x2, y2]
        x, y, bw, bh = mask["bbox"]
        # Scale coordinates back to original image resolution
        x1 = int(x * scale);        y1 = int(y * scale)
        x2 = int((x + bw) * scale); y2 = int((y + bh) * scale)

        colonies.append({
            "bbox":      (x1, y1, x2, y2),
            "area":      mask["area"],
            "area_frac": round(area_fraction, 4),
            "stability": round(float(mask.get("stability_score", 0)), 3),
            "iou":       round(float(mask.get("predicted_iou", 0)), 3),
        })

    # Remove heavily overlapping boxes
    before   = len(colonies)
    colonies = remove_overlapping_boxes(colonies, overlap_threshold=0.5)
    print(f"  SAM found {before} regions → {len(colonies)} after removing overlaps")

    return colonies


# ── Step 4: Load the trained coral classifier ─────────────────────────────────

# Same image pre-processing that was used during training validation
CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize(int(224 * 1.1)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_classifier(checkpoint_path, device):
    """Load the trained EfficientNet-B3 model and its species name list."""
    print(f"Loading coral classifier from {checkpoint_path} ...")
    checkpoint  = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", [])
    num_classes = len(class_names) if class_names else 17

    model = timm.create_model("efficientnet_b3", pretrained=False,
                               num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print(f"Classifier loaded — {num_classes} species, "
          f"best val F1={checkpoint.get('val_f1', 'unknown'):.3f}")
    return model, class_names


def classify_colony(model, class_names, crop_rgb, device):
    """
    Classify a single cropped coral colony image.
    Returns the predicted species name, confidence (0-1), and top-3 guesses.
    """
    if crop_rgb.size == 0 or min(crop_rgb.shape[:2]) < 10:
        return "Unknown", 0.0, []

    # Prepare the crop the same way training images were prepared
    tensor = CLASSIFIER_TRANSFORM(
        Image.fromarray(crop_rgb)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        # Convert raw scores to probabilities (0-100%)
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    best_idx    = int(np.argmax(probabilities))
    best_conf   = float(probabilities[best_idx])
    best_name   = class_names[best_idx] if class_names else str(best_idx)

    # Top 3 predictions with their probabilities
    top3 = [
        (class_names[i] if class_names else str(i), float(c))
        for i, c in sorted(enumerate(probabilities), key=lambda x: -x[1])[:3]
    ]

    return best_name, best_conf, top3


# ── Step 5: Draw boxes and labels on the image ────────────────────────────────

def draw_boxes_and_labels(img_bgr, detections, conf_threshold):
    """
    Draw a coloured bounding box and species label on each detected colony.
    High-confidence detections show the species name.
    Low-confidence ones just say "Coral (xx%)".
    """
    annotated = img_bgr.copy()
    h, w      = annotated.shape[:2]

    # Scale text and box thickness to the image resolution
    scale      = max(w, h) / 1920
    line_thick = max(2, int(2 * scale))
    font_size  = max(0.4, 0.5 * scale)
    font       = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        species         = det["species"]
        confidence      = det["confidence"]

        # Pick colour: species colour if confident, grey if not
        if confidence >= conf_threshold:
            rgb = SPECIES_COLOURS.get(species, SPECIES_COLOURS["Unknown"])
        else:
            rgb = (160, 160, 160)
        bgr = (rgb[2], rgb[1], rgb[0])   # OpenCV uses BGR not RGB

        # Draw the bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, line_thick)

        # Build the label text
        if confidence >= conf_threshold:
            label = f"{species.replace('_', ' ')} {confidence:.0%}"
        else:
            label = f"Coral ({confidence:.0%})"

        # Draw a filled rectangle behind the text so it's readable
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_size, max(1, line_thick - 1))
        label_y = max(y1 - 6, text_h + baseline + 4)
        cv2.rectangle(annotated,
                      (x1, label_y - text_h - baseline - 4),
                      (x1 + text_w + 4, label_y + 2),
                      bgr, -1)

        # White text on dark colours, dark text on light colours
        brightness = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        text_colour = (20, 20, 20) if brightness > 160 else (255, 255, 255)
        cv2.putText(annotated, label, (x1 + 2, label_y - baseline),
                    font, font_size, text_colour,
                    max(1, line_thick - 1), cv2.LINE_AA)

        # Small detection number inside the box
        cv2.putText(annotated, str(det["id"]),
                    (x1 + 4, y1 + int(18 * scale)),
                    font, font_size * 0.7, bgr,
                    max(1, line_thick - 1), cv2.LINE_AA)

    return annotated


def add_species_legend(img_bgr, detections, conf_threshold):
    """
    Add a dark panel on the right side listing all species found in the image.
    """
    # Collect unique species above the confidence threshold
    found_species = {}
    for det in detections:
        if det["confidence"] >= conf_threshold:
            sp = det["species"]
            if sp not in found_species:
                found_species[sp] = True

    if not found_species:
        return img_bgr

    panel_width = 280
    padding     = 12
    line_height = 28
    panel_height = max(img_bgr.shape[0],
                       padding * 2 + 30 + len(found_species) * line_height)

    # Dark background panel
    panel = np.full((panel_height, panel_width, 3), 30, dtype=np.uint8)
    font  = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, "Detected species",
                (padding, padding + 16),
                font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    for i, species in enumerate(sorted(found_species)):
        y   = padding + 30 + i * line_height
        rgb = SPECIES_COLOURS.get(species, SPECIES_COLOURS["Unknown"])
        bgr = (rgb[2], rgb[1], rgb[0])

        # Colour swatch
        cv2.rectangle(panel, (padding, y - 12), (padding + 16, y + 4), bgr, -1)

        # Species name (truncate if too long for the panel)
        name = species.replace("_", " ")[:26]
        cv2.putText(panel, name, (padding + 22, y),
                    font, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

    # Make both panels the same height before joining
    h = img_bgr.shape[0]
    if h < panel.shape[0]:
        img_bgr = np.pad(img_bgr, ((0, panel.shape[0] - h), (0, 0), (0, 0)),
                         mode="edge")
    panel = panel[:img_bgr.shape[0]]

    # Join image and legend side by side
    return np.concatenate([img_bgr, panel], axis=1)


# ── Step 6: Process one image from start to finish ────────────────────────────

def process_image(img_path, sam_wrapper, classifier, class_names,
                  output_dir, conf_threshold, device):
    """Run the full detection + classification pipeline on a single image."""
    print(f"\n{'─' * 60}")
    print(f"Image: {img_path.name}")

    # Load the image
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  ERROR: could not open image")
        return []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]
    print(f"  Size: {w}×{h} pixels")

    # Find coral colonies with SAM
    t0       = time.time()
    colonies = find_colonies(sam_wrapper, img_rgb)
    print(f"  Detection time: {time.time() - t0:.1f}s")

    if not colonies:
        print(f"  No colonies found — try lowering --min_area")
        return []

    # Classify each colony with EfficientNet-B3
    detections = []
    for idx, colony in enumerate(colonies):
        x1, y1, x2, y2 = colony["bbox"]
        bw, bh = x2 - x1, y2 - y1

        # Crop with 15% padding on each side (matches Option F training style)
        px = int(bw * 0.15)
        py = int(bh * 0.15)
        crop = img_rgb[max(0, y1 - py):min(h, y2 + py),
                       max(0, x1 - px):min(w, x2 + px)]

        species, confidence, top3 = classify_colony(
            classifier, class_names, crop, device)

        detection = {
            "id":         idx + 1,
            "species":    species,
            "confidence": confidence,
            "top3":       top3,
            "bbox":       colony["bbox"],
            "area_px":    colony["area"],
            "area_frac":  colony["area_frac"],
            "stability":  colony["stability"],
        }
        detections.append(detection)

        # Print a summary line for each detection
        flag = "✓" if confidence >= conf_threshold else "~"
        print(f"  [{idx+1:2d}] {flag}  "
              f"{species.replace('_', ' '):<33}  "
              f"{confidence:.1%}  "
              f"(colony covers {colony['area_frac']:.1%} of image)")

    # Draw annotations on the image
    annotated = draw_boxes_and_labels(img_bgr, detections, conf_threshold)
    annotated = add_species_legend(annotated, detections, conf_threshold)

    # Save annotated image
    out_img = output_dir / f"{img_path.stem}_annotated.jpg"
    cv2.imwrite(str(out_img), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Saved: {out_img.name}")

    # Save JSON results (useful for downstream analysis)
    out_json = output_dir / f"{img_path.stem}_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "image":      img_path.name,
            "size":       {"width": w, "height": h},
            "detections": detections,
        }, f, indent=2)

    return detections


# ── Step 7: Save a summary spreadsheet ───────────────────────────────────────

def save_summary_csv(all_detections, output_dir, conf_threshold):
    """Write a CSV file with one row per detection across all processed images."""
    import csv
    rows = []
    for img_name, dets in all_detections.items():
        for d in dets:
            rows.append({
                "image":            img_name,
                "detection_id":     d["id"],
                "species":          d["species"].replace("_", " "),
                "confidence":       round(d["confidence"], 3),
                "above_threshold":  d["confidence"] >= conf_threshold,
                "x1": d["bbox"][0], "y1": d["bbox"][1],
                "x2": d["bbox"][2], "y2": d["bbox"][3],
                "area_pixels":      d["area_px"],
                "area_fraction":    d["area_frac"],
            })

    if not rows:
        return

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary spreadsheet saved: {csv_path}")

    # Print breakdown of species found above the confidence threshold
    confident_detections = [r["species"] for r in rows if r["above_threshold"]]
    if confident_detections:
        print(f"\nSpecies found (above {conf_threshold:.0%} confidence threshold):")
        for species, count in Counter(confident_detections).most_common():
            print(f"  {species:<35}  {count} detection(s)")


# ── Step 8: Main entry point ──────────────────────────────────────────────────

def main():
    args = parse_args()

    # Set up device
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # Create output folder
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Collect images to process
    input_path = Path(args.input)
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
        print("No images found at the specified path.")
        return
    print(f"Found {len(img_paths)} image(s) to process")

    # Load both models
    sam_wrapper             = load_sam(args.sam_checkpoint, device,
                                       args.min_area, args.max_area)
    classifier, class_names = load_classifier(args.checkpoint, device)

    # Process every image
    all_detections = {}
    for img_path in img_paths:
        dets = process_image(img_path, sam_wrapper, classifier, class_names,
                             output_dir, args.conf_threshold, device)
        if dets:
            all_detections[img_path.name] = dets

    # Save combined results
    save_summary_csv(all_detections, output_dir, args.conf_threshold)

    # Final summary
    total   = sum(len(d) for d in all_detections.values())
    confident = sum(1 for dets in all_detections.values()
                    for d in dets if d["confidence"] >= args.conf_threshold)
    print(f"\n{'='*60}")
    print(f"Images processed : {len(all_detections)}")
    print(f"Total detections : {total}")
    print(f"Confident labels : {confident}  "
          f"(>= {args.conf_threshold:.0%} confidence)")
    print(f"Results saved to : {output_dir}")


if __name__ == "__main__":
    main()
