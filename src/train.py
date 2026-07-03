"""
Train a Coral Species Classifier
=================================
This script teaches a neural network to recognise 17 species of Caribbean
coral from underwater photos. It works by taking a pre-trained image
recognition model (EfficientNet-B3, already good at recognising everyday
objects) and fine-tuning it on our coral patch images.

How to run:
    conda activate coralnet10
    python train.py --dataset option_f      # train on bounding-box crops
    python train.py --dataset option_a      # train on polygon-masked crops

Where your results go:
    ~/Independent_study/checkpoints_option_f/best_model.pt    <- best model
    ~/Independent_study/results_option_f/training_curves.png  <- learning graphs
    ~/Independent_study/results_option_f/confusion_matrix.png
    ~/Independent_study/results_option_f/classification_report.txt
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


# ── Step 0: Choose which dataset to train on ──────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["option_a", "option_f", "sam2"],
        default="option_f",
        help="option_f = bbox crops, option_a = masked crops, sam2 = SAM2 masked crops"
    )
    parser.add_argument(
        "--backbone",
        choices=["efficientnet_b3", "dinov2_s", "dinov2_b"],
        default="efficientnet_b3",
        help="efficientnet_b3 = default, dinov2_s = smaller ViT, dinov2_b = larger ViT"
    )
    return parser.parse_args()

args     = parse_args()
DATASET  = args.dataset
BACKBONE = args.backbone


# ── Step 1: Settings ──────────────────────────────────────────────────────────

# Where our files live
BASE_DIR       = Path.home() / "Independent_study"
PATCHES_DIR    = BASE_DIR / f"patches_{DATASET}"       # training images
CHECKPOINT_DIR = BASE_DIR / f"checkpoints_{DATASET}"   # where to save the model
RESULTS_DIR    = BASE_DIR / f"results_{DATASET}"        # where to save graphs

# sam2 dataset uses heavier weight_decay to counteract cleaner masks
if DATASET == "sam2":
    WEIGHT_DECAY = 3e-4

NUM_CLASSES     = 17    # we have 17 coral species
IMAGE_SIZE      = 224   # EfficientNet expects 224x224 pixel images
# DINOv2 also uses 224x224 by default — no change needed
BATCH_SIZE      = 32    # how many images to look at at once
NUM_EPOCHS      = 60    # maximum number of full passes through the training data
WARMUP_EPOCHS   = 3     # first few epochs: only train the new classification head
PATIENCE        = 12    # stop early if no improvement for this many epochs

# Learning rates — how big steps the model takes when updating its weights
LR_HEAD         = 1e-3  # faster for the new coral layer (starting from scratch)
LR_BACKBONE     = 1e-4  # slower for the pre-trained layers (preserve knowledge)
WEIGHT_DECAY    = 1e-4  # prevents memorising training data (regularisation)

# Loss function settings
FOCAL_GAMMA     = 2.0   # focus harder on examples the model gets wrong
LABEL_SMOOTHING = 0.05  # slight label uncertainty — improves generalisation

# Use GPU if available, otherwise fall back to CPU
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

# Fix random seeds so results are reproducible
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Training on dataset : {DATASET}")
print(f"Backbone            : {BACKBONE}")
print(f"Patches folder      : {PATCHES_DIR}")
print(f"Device              : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU                 : {torch.cuda.get_device_name(0)}")


# ── Step 2: Prepare the images ────────────────────────────────────────────────

# During training we randomly alter images so the model learns general coral
# features rather than memorising specific training photos.
# This is called "data augmentation".
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),  # random zoom
    transforms.RandomHorizontalFlip(),                            # random left/right flip
    transforms.RandomVerticalFlip(),                              # random up/down flip
    transforms.RandomRotation(30),                                # random rotation
    transforms.ColorJitter(brightness=0.4, contrast=0.4,         # random colour shift
                           saturation=0.3, hue=0.1),             # (simulates lighting)
    transforms.RandomGrayscale(p=0.05),                           # occasionally greyscale
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),    # occasional blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],             # match ImageNet stats
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),          # random small blackout
])

# For validation and testing: no augmentation, just resize and normalise
val_transforms = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.1)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def make_dataloaders():
    """Load images from the patches folder and prepare batches for training."""
    train_data = datasets.ImageFolder(PATCHES_DIR / "train",
                                       transform=train_transforms)
    val_data   = datasets.ImageFolder(PATCHES_DIR / "val",
                                       transform=val_transforms)
    test_data  = datasets.ImageFolder(PATCHES_DIR / "test",
                                       transform=val_transforms)

    print(f"\nDataset sizes:")
    print(f"  Training   : {len(train_data):,} patches")
    print(f"  Validation : {len(val_data):,} patches")
    print(f"  Test       : {len(test_data):,} patches")
    print(f"  Species    : {train_data.classes}")

    # Some species appear far more than others in our dataset.
    # WeightedRandomSampler makes the model see all species equally
    # by over-sampling rare species during training.
    class_counts   = Counter(train_data.targets)
    total_images   = sum(class_counts.values())
    class_weights  = [total_images / class_counts[i] for i in range(NUM_CLASSES)]
    sample_weights = [class_weights[t] for t in train_data.targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_data), replacement=True
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader, train_data.classes, class_weights


# ── Step 3: Build the model ───────────────────────────────────────────────────

def build_model():
    """
    Build the classification model based on --backbone choice.

    efficientnet_b3 : CNN, 12M params, proven baseline (macro-F1 0.606)
    dinov2_s        : ViT-S/14, 22M params, self-supervised on 142M images
                      Better texture features — ideal for coral species
    dinov2_b        : ViT-B/14, 86M params, highest quality but slower
    """
    if BACKBONE == "efficientnet_b3":
        model = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            num_classes=NUM_CLASSES
        )
    elif BACKBONE in ("dinov2_s", "dinov2_b"):
        # DINOv2 via timm — uses facebook/dinov2 pretrained weights
        arch = "vit_small_patch14_dinov2" if BACKBONE == "dinov2_s"                else "vit_base_patch14_dinov2"
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
        # EfficientNet uses 'classifier', DINOv2/ViT uses 'head'
        if "classifier" not in name and "head" not in name:
            param.requires_grad = False


def unfreeze_backbone(model):
    """Unlock all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ── Step 4: Loss function ─────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss focuses the model on hard examples it keeps getting wrong,
    rather than wasting effort on easy examples it already classifies correctly.
    Also handles class imbalance via per-class weights.
    """
    def __init__(self, gamma, weight, label_smoothing):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, predictions, targets):
        base_loss   = nn.functional.cross_entropy(
            predictions, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        probability = torch.exp(-base_loss)
        focal       = ((1 - probability) ** self.gamma) * base_loss
        return focal.mean()


# ── Step 5: Training and evaluation ──────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimiser, device):
    """Train on the full training set once."""
    model.train()
    total_loss = total_correct = total_count = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        preds = model(images)
        loss  = loss_fn(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss    += loss.item() * images.size(0)
        total_correct += (preds.argmax(dim=1) == labels).sum().item()
        total_count   += images.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Evaluate on validation or test set — no weight updates."""
    model.eval()
    total_loss = total_correct = total_count = 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds          = model(images)
        loss           = loss_fn(preds, labels)

        total_loss    += loss.item() * images.size(0)
        total_correct += (preds.argmax(dim=1) == labels).sum().item()
        total_count   += images.size(0)
        all_preds.extend(preds.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total_count, total_correct / total_count, \
           macro_f1, all_preds, all_labels


# ── Step 6: Save graphs ───────────────────────────────────────────────────────

def save_training_curves(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="validation")
    axes[0].set_title("Loss (lower is better)")
    axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="validation")
    axes[1].set_title("Accuracy (higher is better)")
    axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(history["val_f1"], color="green", label="val macro-F1")
    axes[2].set_title("Macro F1 (higher is better)")
    axes[2].set_xlabel("Epoch"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Training curves saved: {path}")


def save_confusion_matrix(true_labels, pred_labels, species_names, path):
    cm     = confusion_matrix(true_labels, pred_labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
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


# ── Step 7: Main training loop ────────────────────────────────────────────────

def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader, species_names, class_weights = \
        make_dataloaders()

    # Set up loss function with class weights
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    weight_tensor = weight_tensor / weight_tensor.mean()
    loss_fn = FocalLoss(FOCAL_GAMMA, weight_tensor, LABEL_SMOOTHING)

    # Build model
    model = build_model().to(DEVICE)

    # ── Phase 1: Warm up the classification head ───────────────────────────
    # Lock the backbone and only train the new coral classification layer.
    # This prevents large random gradients from the new layer from corrupting
    # the useful pre-trained features in the backbone.
    print(f"\nPhase 1: Warming up classification head ({WARMUP_EPOCHS} epochs) ...")
    freeze_backbone(model)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimiser   = optim.AdamW(head_params, lr=LR_HEAD,
                               weight_decay=WEIGHT_DECAY)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=WARMUP_EPOCHS)

    for epoch in range(1, WARMUP_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, loss_fn, optimiser, DEVICE)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(
            model, val_loader, loss_fn, DEVICE)
        scheduler.step()
        print(f"  Epoch {epoch:02d}  "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val_loss={vl_loss:.4f} acc={vl_acc:.3f} f1={vl_f1:.3f}  "
              f"({time.time()-t0:.0f}s)")

    # ── Phase 2: Fine-tune the whole network ───────────────────────────────
    # Unlock everything and train with different learning rates:
    # slow for the backbone (preserve ImageNet knowledge)
    # fast for the head (still learning coral-specific features)
    print(f"\nPhase 2: Full fine-tuning "
          f"(up to {NUM_EPOCHS - WARMUP_EPOCHS} more epochs) ...")
    unfreeze_backbone(model)
    optimiser = optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" not in n], "lr": LR_BACKBONE},
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" in n],     "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    remaining = NUM_EPOCHS - WARMUP_EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=remaining, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  [], "val_f1": []}
    best_f1    = 0.0
    no_improve = 0

    for epoch in range(1, remaining + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, loss_fn, optimiser, DEVICE)
        vl_loss, vl_acc, vl_f1, _, _ = evaluate(
            model, val_loader, loss_fn, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)

        improved = vl_f1 > best_f1
        if improved:
            best_f1    = vl_f1
            no_improve = 0
            torch.save({
                "epoch":       epoch + WARMUP_EPOCHS,
                "model_state": model.state_dict(),
                "val_f1":      vl_f1,
                "val_acc":     vl_acc,
                "class_names": species_names,
            }, CHECKPOINT_DIR / "best_model.pt")
        else:
            no_improve += 1

        marker = " <- new best" if improved else ""
        print(f"  Epoch {epoch + WARMUP_EPOCHS:02d}  "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val_loss={vl_loss:.4f} acc={vl_acc:.3f} f1={vl_f1:.3f}"
              f"{marker}  ({time.time()-t0:.0f}s)")

        if no_improve >= PATIENCE:
            print(f"\nStopping early — no improvement for {PATIENCE} epochs.")
            break

    # Save last checkpoint
    torch.save({"epoch": epoch + WARMUP_EPOCHS,
                "model_state": model.state_dict(),
                "val_f1": vl_f1,
                "class_names": species_names},
               CHECKPOINT_DIR / "last_model.pt")

    # ── Phase 3: Test set evaluation ──────────────────────────────────────
    print(f"\nTesting the best saved model on images it has never seen ...")
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    _, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, loss_fn, DEVICE)

    print(f"\nTest accuracy  : {test_acc:.4f}")
    print(f"Test macro-F1  : {test_f1:.4f}")
    print()
    report = classification_report(test_labels, test_preds,
                                    target_names=species_names, digits=3)
    print(report)

    (RESULTS_DIR / "classification_report.txt").write_text(
        f"Test accuracy  : {test_acc:.4f}\n"
        f"Test macro-F1  : {test_f1:.4f}\n\n{report}"
    )
    save_training_curves(history, RESULTS_DIR / "training_curves.png")
    save_confusion_matrix(test_labels, test_preds, species_names,
                          RESULTS_DIR / "confusion_matrix.png")

    print(f"\nBest validation F1 : {best_f1:.4f}")
    print(f"Model saved to     : {CHECKPOINT_DIR / 'best_model.pt'}")
    print(f"Results saved to   : {RESULTS_DIR}")


if __name__ == "__main__":
    main()