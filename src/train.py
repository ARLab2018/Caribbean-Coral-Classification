"""
Train a Coral Species Classifier
=================================
This script fine-tunes a pre-trained image recognition model on Caribbean
coral patch images.

Usage from the repository root:
    conda activate coralnet10
    python src/train.py --dataset option_f      # bounding-box crops
    python src/train.py --dataset option_a      # MobileSAM polygon-masked crops
    python src/train.py --dataset sam2          # SAM2 polygon-masked crops
    python src/train.py --dataset sam2 --backbone dinov2_s

Default local paths:
    outputs/patches_option_f/       training images for Option F
    outputs/patches_option_a/       training images for Option A
    outputs/patches_sam2/           training images for SAM2
    models/checkpoints_<run>/       saved model checkpoints, ignored by Git
    outputs/results_<run>/          training curves, confusion matrix, report

Large trained checkpoints should stay outside Git and can be hosted externally.
All defaults can be overridden with command-line arguments.
"""

import argparse
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ── Project paths ─────────────────────────────────────────────────────────────

def find_project_root(script_path: Path) -> Path:
    """Return the repository root so the script works from any current folder."""
    for parent in [script_path.parent, *script_path.parents]:
        if (parent / "README.md").exists() or (parent / ".git").exists():
            return parent

    # Expected GitHub location: src/train.py
    if script_path.parent.name == "src":
        return script_path.parent.parent

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


DEFAULT_PATCH_DIRS = {
    "option_a": OUTPUT_ROOT / "patches_option_a",
    "option_f": OUTPUT_ROOT / "patches_option_f",
    "sam2": OUTPUT_ROOT / "patches_sam2",
}


# ── Step 0: Arguments ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a coral species classifier from extracted patch folders."
    )
    parser.add_argument(
        "--dataset",
        choices=["option_a", "option_f", "sam2"],
        default="option_f",
        help="option_f = bbox crops, option_a = MobileSAM masked crops, sam2 = SAM2 masked crops",
    )
    parser.add_argument(
        "--backbone",
        choices=["efficientnet_b3", "dinov2_s", "dinov2_b"],
        default="efficientnet_b3",
        help="efficientnet_b3 = default CNN, dinov2_s = smaller ViT, dinov2_b = larger ViT",
    )
    parser.add_argument(
        "--patches-dir",
        default=None,
        help="Patch dataset folder. Defaults to outputs/patches_<dataset>.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Folder for best_model.pt and last_model.pt. Defaults to models/checkpoints_<run>.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Folder for plots and reports. Defaults to outputs/results_<run>.",
    )
    parser.add_argument("--epochs", type=int, default=60,
                        help="Maximum number of total training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Number of head-only warmup epochs")
    parser.add_argument("--patience", type=int, default=12,
                        help="Early-stopping patience measured in fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader workers. Default: 4 on CUDA, 0 on CPU")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="Learning rate for classification head")
    parser.add_argument("--lr-backbone", type=float, default=1e-4,
                        help="Learning rate for pretrained backbone")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Weight decay. Default: 3e-4 for sam2, otherwise 1e-4")
    parser.add_argument("--device", default="auto",
                        help="auto, cuda, cpu, or a device string like cuda:0")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


args = parse_args()
DATASET = args.dataset
BACKBONE = args.backbone

RUN_NAME = DATASET if BACKBONE == "efficientnet_b3" else f"{DATASET}_{BACKBONE}"
PATCHES_DIR = resolve_path(args.patches_dir) if args.patches_dir else DEFAULT_PATCH_DIRS[DATASET]
CHECKPOINT_DIR = (resolve_path(args.checkpoint_dir)
                  if args.checkpoint_dir else MODEL_DIR / f"checkpoints_{RUN_NAME}")
RESULTS_DIR = (resolve_path(args.results_dir)
               if args.results_dir else OUTPUT_ROOT / f"results_{RUN_NAME}")

NUM_CLASSES = 17
IMAGE_SIZE = args.image_size
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
WARMUP_EPOCHS = max(0, min(args.warmup_epochs, max(NUM_EPOCHS - 1, 0)))
PATIENCE = args.patience

LR_HEAD = args.lr_head
LR_BACKBONE = args.lr_backbone
WEIGHT_DECAY = args.weight_decay if args.weight_decay is not None else (
    3e-4 if DATASET == "sam2" else 1e-4
)

FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.05

if args.device == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(args.device)
NUM_WORKERS = args.num_workers if args.num_workers is not None else (
    4 if DEVICE.type == "cuda" else 0
)

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Project root        : {PROJECT_ROOT}")
print(f"Training on dataset : {DATASET}")
print(f"Backbone            : {BACKBONE}")
print(f"Patches folder      : {PATCHES_DIR}")
print(f"Checkpoint folder   : {CHECKPOINT_DIR}")
print(f"Results folder      : {RESULTS_DIR}")
print(f"Device              : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU                 : {torch.cuda.get_device_name(0)}")


# ── Step 1: Prepare image transforms ─────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

val_transforms = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.1)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def validate_patch_folder():
    """Check that the selected patch folder has ImageFolder train/val/test splits."""
    missing = [split for split in ("train", "val", "test")
               if not (PATCHES_DIR / split).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Patch dataset is missing split folder(s): {missing_str}\n"
            f"Expected ImageFolder structure under: {PATCHES_DIR}\n\n"
            "Run one of the patch extraction scripts first, for example:\n"
            "  python src/patches/extract_mobilesam.py\n"
            "  python src/patches/extract_sam2.py\n\n"
            "Or pass --patches-dir /path/to/your/patches_folder."
        )


def make_dataloaders():
    """Load images from the patches folder and prepare batches for training."""
    global NUM_CLASSES

    validate_patch_folder()

    train_data = datasets.ImageFolder(PATCHES_DIR / "train",
                                      transform=train_transforms)
    val_data = datasets.ImageFolder(PATCHES_DIR / "val",
                                    transform=val_transforms)
    test_data = datasets.ImageFolder(PATCHES_DIR / "test",
                                     transform=val_transforms)

    NUM_CLASSES = len(train_data.classes)

    print(f"\nDataset sizes:")
    print(f"  Training   : {len(train_data):,} patches")
    print(f"  Validation : {len(val_data):,} patches")
    print(f"  Test       : {len(test_data):,} patches")
    print(f"  Species    : {train_data.classes}")

    class_counts = Counter(train_data.targets)
    total_images = sum(class_counts.values())
    class_weights = []
    for i in range(NUM_CLASSES):
        count = class_counts.get(i, 0)
        if count == 0:
            raise ValueError(
                f"Class '{train_data.classes[i]}' has zero training images in {PATCHES_DIR / 'train'}."
            )
        class_weights.append(total_images / count)

    sample_weights = [class_weights[t] for t in train_data.targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_data),
        replacement=True,
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))

    return train_loader, val_loader, test_loader, train_data.classes, class_weights


# ── Step 2: Build the model ───────────────────────────────────────────────────

def is_head_param(name: str) -> bool:
    """Detect classifier-head parameters for EfficientNet and ViT/DINOv2."""
    return "classifier" in name or "head" in name


def build_model():
    """
    Build the classification model based on --backbone choice.

    efficientnet_b3 : CNN baseline.
    dinov2_s        : ViT-S/14 with DINOv2 pretrained weights.
    dinov2_b        : ViT-B/14 with DINOv2 pretrained weights.
    """
    if BACKBONE == "efficientnet_b3":
        model = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            num_classes=NUM_CLASSES,
        )
    elif BACKBONE in ("dinov2_s", "dinov2_b"):
        arch = ("vit_small_patch14_dinov2"
                if BACKBONE == "dinov2_s"
                else "vit_base_patch14_dinov2")
        model = timm.create_model(
            arch,
            pretrained=True,
            num_classes=NUM_CLASSES,
            img_size=IMAGE_SIZE,
        )
    else:
        raise ValueError(f"Unknown backbone: {BACKBONE}")

    print(f"Backbone: {BACKBONE}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")
    return model


def freeze_backbone(model):
    """Lock all layers except the final classification head."""
    for name, param in model.named_parameters():
        if not is_head_param(name):
            param.requires_grad = False


def unfreeze_backbone(model):
    """Unlock all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ── Step 3: Loss function ─────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss focuses the model on hard examples and handles class imbalance
    through per-class weights.
    """
    def __init__(self, gamma, weight, label_smoothing):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, predictions, targets):
        base_loss = nn.functional.cross_entropy(
            predictions,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        probability = torch.exp(-base_loss)
        focal = ((1 - probability) ** self.gamma) * base_loss
        return focal.mean()


# ── Step 4: Training and evaluation ──────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimiser, device):
    """Train on the full training set once."""
    model.train()
    total_loss = total_correct = total_count = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (preds.argmax(dim=1) == labels).sum().item()
        total_count += images.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Evaluate on validation or test set — no weight updates."""
    model.eval()
    total_loss = total_correct = total_count = 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = loss_fn(preds, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds.argmax(dim=1) == labels).sum().item()
        total_count += images.size(0)
        all_preds.extend(preds.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return (total_loss / total_count,
            total_correct / total_count,
            macro_f1,
            all_preds,
            all_labels)


# ── Step 5: Save graphs ───────────────────────────────────────────────────────

def save_training_curves(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss (lower is better)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="validation")
    axes[1].set_title("Accuracy (higher is better)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["val_f1"], label="val macro-F1")
    axes[2].set_title("Macro F1 (higher is better)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Training curves saved: {path}")


def save_confusion_matrix(true_labels, pred_labels, species_names, path):
    cm = confusion_matrix(true_labels, pred_labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm.astype(float), row_sums,
                       out=np.zeros_like(cm, dtype=float),
                       where=row_sums != 0) * 100
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=species_names, yticklabels=species_names, ax=ax)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    ax.set_title("Confusion Matrix — % of true class correctly identified\n"
                 "(bright diagonal = correct, off-diagonal = confused species)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Confusion matrix saved: {path}")


# ── Step 6: Main training loop ────────────────────────────────────────────────

def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, species_names, class_weights = make_dataloaders()

    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    weight_tensor = weight_tensor / weight_tensor.mean()
    loss_fn = FocalLoss(FOCAL_GAMMA, weight_tensor, LABEL_SMOOTHING)

    model = build_model().to(DEVICE)

    print(f"\nPhase 1: Warming up classification head ({WARMUP_EPOCHS} epochs) ...")
    if WARMUP_EPOCHS > 0:
        freeze_backbone(model)
        head_params = [p for p in model.parameters() if p.requires_grad]
        if not head_params:
            raise RuntimeError("No classification-head parameters found for warmup.")
        optimiser = optim.AdamW(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=max(WARMUP_EPOCHS, 1)
        )

        for epoch in range(1, WARMUP_EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn,
                                               optimiser, DEVICE)
            vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader,
                                                    loss_fn, DEVICE)
            scheduler.step()
            print(f"  Epoch {epoch:02d}  "
                  f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
                  f"val_loss={vl_loss:.4f} acc={vl_acc:.3f} f1={vl_f1:.3f}  "
                  f"({time.time() - t0:.0f}s)")
    else:
        print("  Skipped warmup because --warmup-epochs is 0.")

    print(f"\nPhase 2: Full fine-tuning "
          f"(up to {NUM_EPOCHS - WARMUP_EPOCHS} more epochs) ...")
    unfreeze_backbone(model)

    backbone_params = [p for n, p in model.named_parameters() if not is_head_param(n)]
    head_params = [p for n, p in model.named_parameters() if is_head_param(n)]
    optimiser_groups = []
    if backbone_params:
        optimiser_groups.append({"params": backbone_params, "lr": LR_BACKBONE})
    if head_params:
        optimiser_groups.append({"params": head_params, "lr": LR_HEAD})
    optimiser = optim.AdamW(optimiser_groups, weight_decay=WEIGHT_DECAY)

    remaining = max(NUM_EPOCHS - WARMUP_EPOCHS, 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=remaining, eta_min=1e-6
    )

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_epoch = None
    no_improve = 0
    last_epoch = WARMUP_EPOCHS

    for epoch in range(1, remaining + 1):
        last_epoch = epoch + WARMUP_EPOCHS
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn,
                                           optimiser, DEVICE)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(model, val_loader,
                                                loss_fn, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)

        improved = vl_f1 > best_f1
        if improved:
            best_f1 = vl_f1
            best_epoch = last_epoch
            no_improve = 0
            torch.save({
                "epoch": last_epoch,
                "model_state": model.state_dict(),
                "val_f1": vl_f1,
                "val_acc": vl_acc,
                "class_names": species_names,
                "dataset": DATASET,
                "backbone": BACKBONE,
                "image_size": IMAGE_SIZE,
            }, CHECKPOINT_DIR / "best_model.pt")
        else:
            no_improve += 1

        marker = " <- new best" if improved else ""
        print(f"  Epoch {last_epoch:02d}  "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val_loss={vl_loss:.4f} acc={vl_acc:.3f} f1={vl_f1:.3f}"
              f"{marker}  ({time.time() - t0:.0f}s)")

        if no_improve >= PATIENCE:
            print(f"\nStopping early — no improvement for {PATIENCE} epochs.")
            break

    torch.save({
        "epoch": last_epoch,
        "model_state": model.state_dict(),
        "val_f1": vl_f1,
        "class_names": species_names,
        "dataset": DATASET,
        "backbone": BACKBONE,
        "image_size": IMAGE_SIZE,
    }, CHECKPOINT_DIR / "last_model.pt")

    print(f"\nTesting the best saved model on images it has never seen ...")
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    _, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, loss_fn, DEVICE
    )

    print(f"\nTest accuracy  : {test_acc:.4f}")
    print(f"Test macro-F1  : {test_f1:.4f}")
    print()
    report = classification_report(test_labels, test_preds,
                                   target_names=species_names, digits=3)
    print(report)

    (RESULTS_DIR / "classification_report.txt").write_text(
        f"Run name        : {RUN_NAME}\n"
        f"Dataset         : {DATASET}\n"
        f"Backbone        : {BACKBONE}\n"
        f"Best epoch      : {best_epoch}\n"
        f"Best val F1     : {best_f1:.4f}\n"
        f"Test accuracy   : {test_acc:.4f}\n"
        f"Test macro-F1   : {test_f1:.4f}\n\n{report}",
        encoding="utf-8",
    )
    save_training_curves(history, RESULTS_DIR / "training_curves.png")
    save_confusion_matrix(test_labels, test_preds, species_names,
                          RESULTS_DIR / "confusion_matrix.png")

    print(f"\nBest validation F1 : {best_f1:.4f}")
    print(f"Model saved to     : {CHECKPOINT_DIR / 'best_model.pt'}")
    print(f"Results saved to   : {RESULTS_DIR}")


if __name__ == "__main__":
    main()
