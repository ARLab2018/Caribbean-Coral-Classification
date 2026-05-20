"""
Coral Species Manual Labelling Tool
=====================================
A desktop GUI for manually assigning species labels to coral colony detections
that were flagged as low-confidence or "Unknown" by the inference pipeline.

Reads *_results.json files from an inference output folder, shows each flagged
colony in context on the original or overlay image, and lets you assign the
correct species label from a button panel.

This version is GitHub/repo friendly:
  - It does not assume ~/Independent_study paths.
  - Defaults are relative to the repository root.
  - External data/images should be downloaded into data/ or selected with args.

Default expected repo layout:
    caribbean-coral-classification/
    ├── data/CoralNet_Data/
    ├── outputs/inference/
    ├── outputs/manual_labels/
    └── tools/label_tool.py

Usage from the repository root:
    python tools/label_tool.py

    # Use a specific inference output folder:
    python tools/label_tool.py --inference_dir outputs/inference/sam2_effnet

    # Use a specific original image folder:
    python tools/label_tool.py \
        --inference_dir outputs/inference/sam2_effnet \
        --image_dir data/CoralNet_Data

    # Review only detections below a confidence threshold:
    python tools/label_tool.py --conf_threshold 0.50

    # Resume a previous labelling session:
    python tools/label_tool.py --resume

Controls:
    Click a species button    assign that species label
    Skip (S key)              skip this detection without labelling
    Back (B key)              go back to previous detection
    Zoom +/- (scroll wheel)   zoom in/out on the image
    Q key                     quit and save

Output:
    outputs/manual_labels/labels.csv
"""

import argparse
import csv
import json
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from datetime import datetime
import threading

import cv2
import numpy as np
from PIL import Image, ImageTk

# ── Configuration ─────────────────────────────────────────────────────────────

SPECIES_LIST = [
    "Acropora_tenuifolia",
    "Agaricia_agaricites",
    "Colpophyllia_natans",
    "Lobophyllia_spp",
    "Madracis_auretenra",
    "Madracis_mirabilis",
    "Meandrina_meandrites",
    "Millepora_spp",
    "Montastraea_cavernosa",
    "Orbicella_annularis",
    "Orbicella_faveolata",
    "Orbicella_franksi",
    "Porites_astreoides",
    "Porites_porites",
    "Pseudodiploria_strigosa",
    "Siderastrea_siderea",
    "Stephanocoenia_intersepta",
    "Not_coral",        # for false positives
    "Cannot_determine", # for ambiguous cases
]

# Colour for each species (BGR for OpenCV, then converted)
SPECIES_COLOURS = {
    "Acropora_tenuifolia":       "#FF6464",
    "Agaricia_agaricites":       "#64C8FF",
    "Colpophyllia_natans":       "#FFB432",
    "Lobophyllia_spp":           "#96FF96",
    "Madracis_auretenra":        "#C864FF",
    "Madracis_mirabilis":        "#FF32C8",
    "Meandrina_meandrites":      "#32DCB4",
    "Millepora_spp":             "#DCDC32",
    "Montastraea_cavernosa":     "#50A0FF",
    "Orbicella_annularis":       "#FF8C28",
    "Orbicella_faveolata":       "#B4FF50",
    "Orbicella_franksi":         "#FF5050",
    "Porites_astreoides":        "#50FFDC",
    "Porites_porites":           "#C8FF64",
    "Pseudodiploria_strigosa":   "#FFA0C8",
    "Siderastrea_siderea":       "#A078FF",
    "Stephanocoenia_intersepta": "#FFC864",
    "Not_coral":                 "#AAAAAA",
    "Cannot_determine":          "#666666",
}



def find_project_root() -> Path:
    """Return the repository root when this file is stored in tools/."""
    script_path = Path(__file__).resolve()

    # Normal GitHub layout: repo_root/tools/label_tool.py
    if script_path.parent.name == "tools":
        return script_path.parents[1]

    # Fallback for direct execution/copies: use the current working directory.
    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()
DEFAULT_INFERENCE_DIR = PROJECT_ROOT / "outputs" / "inference"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "CoralNet_Data"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "manual_labels" / "labels.csv"


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Manual coral species labelling tool")
    p.add_argument("--inference_dir", default=str(DEFAULT_INFERENCE_DIR),
                   help=("Folder containing *_results.json files from inference "
                         f"(default: {DEFAULT_INFERENCE_DIR})"))
    p.add_argument("--image_dir",     default=str(DEFAULT_IMAGE_DIR),
                   help=("Root folder containing original images "
                         f"(default: {DEFAULT_IMAGE_DIR})"))
    p.add_argument("--conf_threshold", type=float, default=0.40,
                   help="Review detections BELOW this confidence (default: 0.40)")
    p.add_argument("--resume",        action="store_true",
                   help="Skip already-labelled detections from previous session")
    p.add_argument("--output",        default=str(DEFAULT_OUTPUT_CSV),
                   help=("Where to save labels CSV "
                         f"(default: {DEFAULT_OUTPUT_CSV})"))
    return p.parse_args()


# ── Load detections from JSON files ───────────────────────────────────────────

def load_detections(inference_dir, image_dir, conf_threshold):
    """
    Load all detections below conf_threshold from _results.json files.
    Resolves original image paths by searching image_dir.
    """
    inference_path = Path(inference_dir).expanduser().resolve()
    image_root     = Path(image_dir).expanduser().resolve()

    if not inference_path.exists():
        raise FileNotFoundError(
            f"Inference folder not found: {inference_path}\n"
            "Run an inference script first or pass --inference_dir explicitly."
        )

    # Build image index (stem -> path) for quick lookup.
    # If the original image folder is unavailable, we still try overlay/local images.
    print("Building image index ...")
    img_index = {}
    if image_root.exists():
        for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
            for p in image_root.rglob(ext):
                img_index[p.stem.lower()] = p
                img_index[p.name.lower()] = p
        print(f"  Found {len(img_index):,} images in {image_root}")
    else:
        print(f"  WARNING: image_dir not found: {image_root}")
        print("  Will try to use overlay/original files from the inference folder.")

    detections = []
    json_files  = sorted(inference_path.glob("*_results.json"))
    print(f"  Found {len(json_files)} result JSON files")

    for json_path in json_files:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Could not read {json_path.name}: {e}")
            continue

        img_name = data.get("image") or json_path.name.replace("_results.json", "")
        img_stem = Path(img_name).stem.lower()

        # Find the original image
        orig_img_path = (img_index.get(img_name.lower()) or
                         img_index.get(img_stem))
        if orig_img_path is None:
            # Try finding by name in inference dir itself (infer saved annotated copy)
            local_orig = inference_path / img_name
            if local_orig.exists():
                orig_img_path = local_orig
            else:
                print(f"  WARNING: original image not found for {img_name}")
                continue

        for det in data.get("detections", []):
            conf    = det.get("confidence", 0)
            species = det.get("species", "Unknown")

            # Flag low-confidence or unknown detections
            if conf < conf_threshold or species in ("Unknown", "Coral"):
                detections.append({
                    "json_path":    str(json_path),
                    "overlay_path": str(json_path).replace("_results.json", "_overlay.jpg"),
                    "img_name":     img_name,
                    "orig_img":     str(orig_img_path),
                    "det_id":       det.get("id", 0),
                    "bbox":         det.get("bbox", [0, 0, 100, 100]),
                    "pred_species": species,
                    "confidence":   round(conf, 3),
                    "area_frac":    det.get("area_frac", 0),
                    "stability":    det.get("stability", 0),
                    "top3":         det.get("top3", []),
                })

    print(f"  Total detections to review: {len(detections):,}")
    return detections


# ── Main GUI ──────────────────────────────────────────────────────────────────

class LabellingTool:

    def __init__(self, root, detections, args):
        self.root        = root
        self.detections  = detections
        self.args        = args
        self.current_idx = 0
        self.labels      = {}   # det_key -> assigned_species
        self.zoom_level  = 1.0
        self.pan_x       = 0
        self.pan_y       = 0
        self.current_img = None  # PIL Image of current view
        self.sam_predictor = None
        self._load_sam()

        # Load previously saved labels for resuming
        self.output_path = Path(args.output)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_existing_labels()

        if args.resume:
            self._skip_labelled()

        self._setup_ui()
        self._bind_keys()
        self._load_detection(self.current_idx)

    # ── SAM mask generation ───────────────────────────────────────────────

    # ── No SAM needed — we use pre-computed overlay images ──────────────

    def _load_sam(self):
        """SAM not needed — overlay images are pre-computed by inference."""
        pass

    def _load_existing_labels(self):
        """Load labels from a previous session."""
        if not self.output_path.exists():
            return
        with open(self.output_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['img_name']}_{row['det_id']}"
                self.labels[key] = row["assigned_species"]
        print(f"Loaded {len(self.labels)} existing labels from {self.output_path}")

    def _skip_labelled(self):
        """Skip to first unlabelled detection."""
        for i, det in enumerate(self.detections):
            key = f"{det['img_name']}_{det['det_id']}"
            if key not in self.labels:
                self.current_idx = i
                return
        self.current_idx = len(self.detections)

    def _save_label(self, det, species):
        """Append a label to the CSV file."""
        key      = f"{det['img_name']}_{det['det_id']}"
        self.labels[key] = species

        # Append to CSV (or create with header)
        write_header = not self.output_path.exists()
        with open(self.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "img_name", "det_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                "pred_species", "confidence", "assigned_species",
                "area_frac", "timestamp"
            ])
            if write_header:
                writer.writeheader()
            x1, y1, x2, y2 = det["bbox"]
            writer.writerow({
                "img_name":        det["img_name"],
                "det_id":          det["det_id"],
                "bbox_x1":         x1, "bbox_y1": y1,
                "bbox_x2":         x2, "bbox_y2": y2,
                "pred_species":    det["pred_species"],
                "confidence":      det["confidence"],
                "assigned_species":species,
                "area_frac":       det["area_frac"],
                "timestamp":       datetime.now().isoformat(),
            })

    # ── UI Setup ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        self.root.title("Coral Species Labelling Tool")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 750)

        # ── Top bar ──────────────────────────────────────────────────────
        top_bar = tk.Frame(self.root, bg="#16213e", height=56)
        top_bar.pack(fill=tk.X, side=tk.TOP)
        top_bar.pack_propagate(False)

        tk.Label(top_bar, text="🪸  Coral Labelling Tool",
                 bg="#16213e", fg="#e0e0ff",
                 font=("Georgia", 16, "bold")).pack(side=tk.LEFT, padx=20, pady=14)

        self.progress_var = tk.StringVar(value="0 / 0")
        tk.Label(top_bar, textvariable=self.progress_var,
                 bg="#16213e", fg="#7090cc",
                 font=("Courier", 12)).pack(side=tk.RIGHT, padx=20)

        self.pct_var = tk.StringVar(value="0%")
        tk.Label(top_bar, textvariable=self.pct_var,
                 bg="#16213e", fg="#50cc90",
                 font=("Courier", 12, "bold")).pack(side=tk.RIGHT, padx=4)

        # ── Main layout ───────────────────────────────────────────────────
        main = tk.Frame(self.root, bg="#1a1a2e")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Left: image panel
        left = tk.Frame(main, bg="#1a1a2e")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Image info label
        self.info_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self.info_var,
                 bg="#1a1a2e", fg="#8090a0",
                 font=("Courier", 10), anchor="w").pack(fill=tk.X, pady=(0, 4))

        # Canvas
        self.canvas = tk.Canvas(left, bg="#0d0d1a", highlightthickness=0,
                                 cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>",   self._on_scroll)
        self.canvas.bind("<Button-5>",   self._on_scroll)
        # Pan with left drag (click and drag)
        self.canvas.bind("<ButtonPress-1>",   self._pan_start)
        self.canvas.bind("<B1-Motion>",        self._pan_move)
        self._pan_start_x = 0
        self._pan_start_y = 0

        # Zoom controls under canvas
        zoom_bar = tk.Frame(left, bg="#1a1a2e")
        zoom_bar.pack(fill=tk.X, pady=4)
        tk.Label(zoom_bar, text="Zoom:", bg="#1a1a2e",
                 fg="#7090cc", font=("Courier", 10)).pack(side=tk.LEFT)
        self.zoom_var = tk.StringVar(value="100%")
        tk.Label(zoom_bar, textvariable=self.zoom_var,
                 bg="#1a1a2e", fg="#50cc90",
                 font=("Courier", 10, "bold")).pack(side=tk.LEFT, padx=8)
        tk.Button(zoom_bar, text="Reset Zoom", command=self._reset_zoom,
                  bg="#16213e", fg="#8090b0",
                  font=("Courier", 9), relief=tk.FLAT,
                  padx=8, pady=2).pack(side=tk.LEFT)

        # Right: label panel
        right = tk.Frame(main, bg="#16213e", width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)

        # Detection info box
        info_frame = tk.Frame(right, bg="#0d1929", pady=10, padx=14)
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(info_frame, text="CURRENT DETECTION",
                 bg="#0d1929", fg="#5070a0",
                 font=("Courier", 9, "bold")).pack(anchor="w")

        self.det_info_var = tk.StringVar(value="")
        tk.Label(info_frame, textvariable=self.det_info_var,
                 bg="#0d1929", fg="#c0d0e0",
                 font=("Courier", 10), justify=tk.LEFT,
                 wraplength=270).pack(anchor="w", pady=(6, 0))

        # Top-3 predictions
        tk.Label(info_frame, text="\nMODEL TOP-3:",
                 bg="#0d1929", fg="#5070a0",
                 font=("Courier", 9, "bold")).pack(anchor="w")
        self.top3_var = tk.StringVar(value="")
        tk.Label(info_frame, textvariable=self.top3_var,
                 bg="#0d1929", fg="#90b0d0",
                 font=("Courier", 9), justify=tk.LEFT,
                 wraplength=270).pack(anchor="w")

        # Species buttons
        tk.Label(right, text="ASSIGN SPECIES LABEL",
                 bg="#16213e", fg="#5070a0",
                 font=("Courier", 9, "bold")).pack(padx=10, anchor="w", pady=(8, 4))

        btn_frame = tk.Frame(right, bg="#16213e")
        btn_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        self.species_buttons = {}
        for i, sp in enumerate(SPECIES_LIST):
            colour = SPECIES_COLOURS.get(sp, "#888888")
            display = sp.replace("_", " ")
            btn = tk.Button(
                btn_frame,
                text=display,
                bg="#0d1929",
                fg=colour,
                activebackground=colour,
                activeforeground="#000000",
                font=("Courier", 10),
                relief=tk.FLAT,
                anchor="w",
                padx=10, pady=4,
                cursor="hand2",
                command=lambda s=sp: self._assign_label(s)
            )
            btn.pack(fill=tk.X, pady=1)
            self.species_buttons[sp] = btn

            # Hover effects
            btn.bind("<Enter>", lambda e, b=btn, c=colour: b.configure(bg=c, fg="#000000"))
            btn.bind("<Leave>", lambda e, b=btn, c=colour: b.configure(bg="#0d1929", fg=c))

        # Action buttons
        action_frame = tk.Frame(right, bg="#16213e")
        action_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(action_frame, text="← Back  (B)",
                  command=self._go_back,
                  bg="#1e2a3e", fg="#7090cc",
                  font=("Courier", 10), relief=tk.FLAT,
                  padx=10, pady=6, cursor="hand2").pack(fill=tk.X, pady=2)

        tk.Button(action_frame, text="Skip  (S)",
                  command=self._skip,
                  bg="#1e2a3e", fg="#cc9050",
                  font=("Courier", 10), relief=tk.FLAT,
                  padx=10, pady=6, cursor="hand2").pack(fill=tk.X, pady=2)

        tk.Button(action_frame, text="Quit & Save  (Q)",
                  command=self._quit,
                  bg="#3e1e1e", fg="#cc5050",
                  font=("Courier", 10), relief=tk.FLAT,
                  padx=10, pady=6, cursor="hand2").pack(fill=tk.X, pady=2)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var,
                 bg="#0d0d1a", fg="#5070a0",
                 font=("Courier", 9), anchor="w",
                 relief=tk.FLAT).pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=2)

    def _bind_keys(self):
        self.root.bind("<s>", lambda e: self._skip())
        self.root.bind("<S>", lambda e: self._skip())
        self.root.bind("<b>", lambda e: self._go_back())
        self.root.bind("<B>", lambda e: self._go_back())
        self.root.bind("<q>", lambda e: self._quit())
        self.root.bind("<Q>", lambda e: self._quit())
        # Number keys 1-9 for quick species selection
        for i in range(min(9, len(SPECIES_LIST))):
            self.root.bind(str(i+1),
                lambda e, idx=i: self._assign_label(SPECIES_LIST[idx]))

    # ── Image rendering ───────────────────────────────────────────────────

    def _load_detection(self, idx):
        """Load and display detection at index idx."""
        if idx >= len(self.detections):
            self._show_finished()
            return

        det = self.detections[idx]
        n   = len(self.detections)
        labelled = sum(1 for k in self.labels if k)
        self.progress_var.set(f"{idx + 1} / {n}")
        pct = int(len(self.labels) / n * 100)
        self.pct_var.set(f"{pct}% done")

        # Detection info
        self.det_info_var.set(
            f"Image: {det['img_name']}\n"
            f"Colony ID: {det['det_id']}\n"
            f"Predicted: {det['pred_species'].replace('_',' ')}\n"
            f"Confidence: {det['confidence']:.1%}\n"
            f"Area: {det['area_frac']:.1%} of image"
        )

        # Top-3 predictions
        top3_text = ""
        for sp, conf in det.get("top3", [])[:3]:
            top3_text += f"  {sp.replace('_',' ')[:22]:<22} {conf:.1%}\n"
        self.top3_var.set(top3_text.strip())

        self.info_var.set(f"  {det['orig_img']}")
        self.status_var.set(f"Loading {det['img_name']} ...")

        # Load image in background thread to keep UI responsive
        threading.Thread(
            target=self._render_image,
            args=(det,),
            daemon=True
        ).start()

        # Highlight button for previously assigned label
        key = f"{det['img_name']}_{det['det_id']}"
        prev = self.labels.get(key)
        for sp, btn in self.species_buttons.items():
            colour = SPECIES_COLOURS.get(sp, "#888888")
            if sp == prev:
                btn.configure(bg=colour, fg="#000000",
                               relief=tk.SOLID)
            else:
                btn.configure(bg="#0d1929", fg=colour,
                               relief=tk.FLAT)

    def _render_image(self, det):
        """
        Load the pre-computed _overlay.jpg and highlight the specific colony.
        The overlay already has all masks painted — we just add a bright
        outline around the colony currently being reviewed.
        """
        try:
            overlay_path = det.get("overlay_path", "")

            # Try overlay image first, fall back to original if not found
            if overlay_path and Path(overlay_path).exists():
                img_bgr     = cv2.imread(overlay_path)
                source_desc = "overlay"
            else:
                # Find original image
                img_bgr     = cv2.imread(det["orig_img"])
                source_desc = "original"
                if img_bgr is None:
                    self.status_var.set(
                        f"ERROR: neither overlay nor original found\n"
                        f"  overlay: {overlay_path}\n"
                        f"  orig:    {det['orig_img']}"
                    )
                    return

            if img_bgr is None:
                self.status_var.set(f"ERROR: Could not load image")
                return

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w    = img_rgb.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

            img_display = img_rgb.copy()

            # Draw a bright animated-style double outline to highlight
            # the colony currently being reviewed
            colour  = SPECIES_COLOURS.get(det["pred_species"], "#888888")
            r, g, b = int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)

            # Outer white outline for contrast
            cv2.rectangle(img_display,
                          (x1-3, y1-3), (x2+3, y2+3),
                          (255, 255, 255), 3)
            # Inner species-colour outline
            cv2.rectangle(img_display,
                          (x1, y1), (x2, y2),
                          (r, g, b), 3)

            # Corner marks for precision
            corner = max(20, int(min(x2-x1, y2-y1) * 0.15))
            for cx, cy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),
                                    (x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(img_display, (cx, cy),
                         (cx + dx*corner, cy), (255,255,255), 4)
                cv2.line(img_display, (cx, cy),
                         (cx, cy + dy*corner), (255,255,255), 4)
                cv2.line(img_display, (cx, cy),
                         (cx + dx*corner, cy), (r,g,b), 2)
                cv2.line(img_display, (cx, cy),
                         (cx, cy + dy*corner), (r,g,b), 2)

            # Label above the box
            label      = (f"{det['pred_species'].replace('_',' ')}  "
                          f"{det['confidence']:.0%}")
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.6, min(1.4, w / 2000))
            thick      = max(1, int(font_scale * 2))
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thick)
            ly = max(y1 - 8, th + 8)
            # White backing
            cv2.rectangle(img_display,
                          (x1-1, ly-th-7), (x1+tw+11, ly+5),
                          (255,255,255), -1)
            cv2.rectangle(img_display,
                          (x1, ly-th-6), (x1+tw+10, ly+4),
                          (r,g,b), -1)
            bright = 0.299*r + 0.587*g + 0.114*b
            tc = (20,20,20) if bright > 128 else (255,255,255)
            cv2.putText(img_display, label, (x1+4, ly),
                        font, font_scale, tc, thick, cv2.LINE_AA)

            # Store full image
            self.current_img     = Image.fromarray(img_display)
            self.current_img_roi = (0, 0, w, h)
            self.pan_x = 0
            self.pan_y = 0

            # Auto-zoom to colony
            self.root.after(0, lambda: self._auto_zoom(x1, y1, x2, y2, w, h))
            self.root.after(0, self._display_image)
            self.status_var.set(
                f"Colony {det['det_id']} — {det['img_name']}  |  "
                f"predicted: {det['pred_species'].replace('_',' ')}  "
                f"({det['confidence']:.1%})  |  source: {source_desc}"
            )

        except Exception as e:
            self.status_var.set(f"Error: {e}")
            import traceback; traceback.print_exc()

    def _display_image(self):
        """
        Render the current image onto the canvas.
        At high zoom levels, only the visible portion is drawn for speed.
        """
        if self.current_img is None:
            return

        cw = self.canvas.winfo_width()  or 900
        ch = self.canvas.winfo_height() or 650

        iw, ih = self.current_img.size

        # Scaled image dimensions
        nw = max(1, int(iw * self.zoom_level))
        nh = max(1, int(ih * self.zoom_level))

        if not hasattr(self, 'pan_x'): self.pan_x = cw / 2
        if not hasattr(self, 'pan_y'): self.pan_y = ch / 2

        # Top-left corner of the scaled image in canvas coords
        img_x0 = int(self.pan_x)
        img_y0 = int(self.pan_y)

        # Visible region in canvas coords
        vis_x1 = max(0, -img_x0)
        vis_y1 = max(0, -img_y0)
        vis_x2 = min(nw, cw - img_x0)
        vis_y2 = min(nh, ch - img_y0)

        if vis_x2 <= vis_x1 or vis_y2 <= vis_y1:
            self.canvas.delete("all")
            self.zoom_var.set(f"{int(self.zoom_level * 100)}%")
            return

        # Source region in original image coords
        src_x1 = int(vis_x1 / self.zoom_level)
        src_y1 = int(vis_y1 / self.zoom_level)
        src_x2 = min(iw, int(vis_x2 / self.zoom_level) + 1)
        src_y2 = min(ih, int(vis_y2 / self.zoom_level) + 1)

        # Crop to visible region and scale
        crop    = self.current_img.crop((src_x1, src_y1, src_x2, src_y2))
        out_w   = vis_x2 - vis_x1
        out_h   = vis_y2 - vis_y1
        resample = Image.BILINEAR if out_w * out_h > 500_000 else Image.LANCZOS
        display  = crop.resize((out_w, out_h), resample)

        self._photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        # Place top-left of visible crop at its canvas position
        canvas_x = max(0, img_x0)
        canvas_y = max(0, img_y0)
        self.canvas.create_image(canvas_x, canvas_y,
                                  image=self._photo, anchor=tk.NW)
        self.zoom_var.set(f"{int(self.zoom_level * 100)}%")

    def _pan_start(self, event):
        self._pan_start_x = event.x
        self._pan_start_y = event.y

    def _pan_move(self, event):
        if not hasattr(self, 'pan_x'): self.pan_x = 0
        if not hasattr(self, 'pan_y'): self.pan_y = 0
        self.pan_x += event.x - self._pan_start_x
        self.pan_y += event.y - self._pan_start_y
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self._display_image()

    def _on_scroll(self, event):
        if event.num == 4 or event.delta > 0:
            self.zoom_level = min(8.0, self.zoom_level * 1.15)
        else:
            self.zoom_level = max(0.1, self.zoom_level / 1.15)
        self._display_image()

    def _reset_zoom(self):
        cw = self.canvas.winfo_width()  or 900
        ch = self.canvas.winfo_height() or 650
        if self.current_img:
            iw, ih = self.current_img.size
            # Fit whole image in canvas
            self.zoom_level = min(cw / iw, ch / ih)
            self.pan_x = (cw - iw * self.zoom_level) / 2
            self.pan_y = (ch - ih * self.zoom_level) / 2
        else:
            self.zoom_level = 1.0
            self.pan_x = 0
            self.pan_y = 0
        self._display_image()

    def _auto_zoom(self, x1, y1, x2, y2, img_w, img_h):
        """
        Zoom and pan so the colony bbox is centred and fills ~60% of canvas.
        Uses top-left pan coordinates to match _display_image.
        """
        cw = self.canvas.winfo_width()  or 900
        ch = self.canvas.winfo_height() or 650

        pad   = max(100, int(min(img_w, img_h) * 0.10))
        bw    = (x2 - x1) + pad * 2
        bh    = (y2 - y1) + pad * 2

        zoom_x = (cw * 0.65) / bw
        zoom_y = (ch * 0.65) / bh
        self.zoom_level = max(0.1, min(8.0, min(zoom_x, zoom_y)))

        # Centre of bbox in scaled image coords
        bbox_cx = (x1 + x2) / 2 * self.zoom_level
        bbox_cy = (y1 + y2) / 2 * self.zoom_level

        # pan_x/y = top-left corner of scaled image in canvas coords
        # set so bbox centre lands at canvas centre
        self.pan_x = cw / 2 - bbox_cx
        self.pan_y = ch / 2 - bbox_cy

        self._display_image()

    def _assign_label(self, species):
        """Assign a species label to the current detection and advance."""
        if self.current_idx >= len(self.detections):
            return
        det = self.detections[self.current_idx]
        self._save_label(det, species)

        # Flash the selected button
        btn = self.species_buttons.get(species)
        if btn:
            colour = SPECIES_COLOURS.get(species, "#888888")
            btn.configure(bg=colour, fg="#000000")
            self.root.after(200, lambda: btn.configure(bg="#0d1929", fg=colour))

        self.status_var.set(
            f"Labelled as: {species.replace('_',' ')}  "
            f"({len(self.labels)} total labels saved)"
        )
        self.current_idx += 1
        self._load_detection(self.current_idx)

    def _skip(self):
        """Skip current detection without labelling."""
        self.current_idx += 1
        self._load_detection(self.current_idx)
        self.status_var.set("Skipped.")

    def _go_back(self):
        """Go back one detection."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self._load_detection(self.current_idx)

    def _quit(self):
        n = len(self.labels)
        if messagebox.askokcancel("Quit", f"Save and quit?\n{n} labels saved to:\n{self.output_path}"):
            self.root.quit()

    def _show_finished(self):
        """Show completion screen."""
        self.canvas.delete("all")
        cw = self.canvas.winfo_width() or 900
        ch = self.canvas.winfo_height() or 650
        self.canvas.create_text(
            cw//2, ch//2 - 30,
            text="All detections reviewed!",
            fill="#50cc90",
            font=("Georgia", 22, "bold")
        )
        self.canvas.create_text(
            cw//2, ch//2 + 20,
            text=f"{len(self.labels)} labels saved to:\n{self.output_path}",
            fill="#7090cc",
            font=("Courier", 12),
            justify=tk.CENTER
        )
        self.status_var.set(f"Complete — {len(self.labels)} labels saved.")
        self.progress_var.set(f"{len(self.detections)} / {len(self.detections)}")
        self.pct_var.set("100% done")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\nCoral Species Labelling Tool")
    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Inference dir : {args.inference_dir}")
    print(f"Image dir     : {args.image_dir}")
    print(f"Conf threshold: {args.conf_threshold} (reviewing below this)")
    print(f"Output        : {args.output}\n")

    try:
        detections = load_detections(
            args.inference_dir,
            args.image_dir,
            args.conf_threshold
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    if not detections:
        print("No detections found below the confidence threshold.")
        print("Try lowering --conf_threshold or check that --inference_dir contains *_results.json files.")
        return

    print(f"\nStarting GUI with {len(detections):,} detections to review ...")
    print("Controls: Click species button to label | S=skip | B=back | Q=quit\n")

    root = tk.Tk()
    root.resizable(True, True)
    app  = LabellingTool(root, detections, args)
    root.mainloop()

    print(f"\nSession complete. {len(app.labels)} labels saved to {app.output_path}")


if __name__ == "__main__":
    main()
