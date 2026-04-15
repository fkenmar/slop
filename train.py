"""
Fine-tunes a CLIP-ViT backbone with DoRA + FFT fusion for deepfake detection.

Architecture:
  - Branch 1: CLIP ViT-L/14 vision encoder (spatial features)
  - Branch 2: 2D FFT magnitude → lightweight CNN (frequency artifacts)
  - Fusion MLP classification head

Training:
  1. DoRA on all CLIP attention layers — adapts early layers for frequency cues
  2. Gradient Accumulation     — simulates batch_size=64 with less memory
  3. CosineAnnealingWarmRestarts — resume-friendly LR schedule
  4. Early Stopping            — stops if validation accuracy plateaus
  5. AMP (float16)             — 2x memory savings on MPS
  6. Albumentations pipeline   — social media degradation, JPEG, high-pass filter
  7. Label smoothing (0.1)     — prevents overconfident predictions
  8. Balanced batch sampling   — equal representation per dataset-source × class
  9. Checkpoint versioning     — saves per-epoch snapshots for rollback

Saves the merged model to ./model/ for use in app.py.
"""

import os
import multiprocessing
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, Sampler
from torchvision.datasets import ImageFolder
from transformers import CLIPVisionModel, CLIPImageProcessor
from peft import LoraConfig, get_peft_model
from model import DeepfakeDetector, CLIP_MODEL_ID
from sklearn.metrics import roc_auc_score
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm



# ── Balanced Batch Sampler ────────────────────────────────────────────────────
# Ensures each batch draws equally from every (dataset-source × class) group.
# With 2 datasets × 2 classes = 4 groups, a batch of 32 gets 8 samples per group.
# This prevents the larger dataset from dominating training and ensures the model
# sees balanced representation of each generator type per step.

class BalancedBatchSampler(Sampler):
    """Samples batches with equal representation from each group.

    Args:
        group_indices: list of lists, where each inner list contains
                       the global indices for one group.
        batch_size:    total batch size (must be divisible by number of groups).
    """

    def __init__(self, group_indices, batch_size):
        self.group_indices = [np.array(g) for g in group_indices]
        self.n_groups = len(group_indices)
        assert batch_size % self.n_groups == 0, (
            f"batch_size ({batch_size}) must be divisible by n_groups ({self.n_groups})"
        )
        self.per_group = batch_size // self.n_groups
        self.batch_size = batch_size
        # Length is determined by the smallest group (one full pass through it)
        self._min_group_size = min(len(g) for g in self.group_indices)
        self._num_batches = self._min_group_size // self.per_group

    def __iter__(self):
        # Shuffle each group independently
        shuffled = [np.random.permutation(g) for g in self.group_indices]
        # Oversample smaller groups to match the largest group for this epoch
        max_needed = self._num_batches * self.per_group
        padded = []
        for g in shuffled:
            if len(g) < max_needed:
                repeats = (max_needed // len(g)) + 1
                g = np.tile(g, repeats)[:max_needed]
            else:
                g = g[:max_needed]
            padded.append(g)

        for i in range(self._num_batches):
            batch = []
            for g in padded:
                start = i * self.per_group
                batch.extend(g[start:start + self.per_group].tolist())
            # Shuffle within the batch so groups aren't in a fixed order
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self._num_batches


# ── Supervised Contrastive Loss ───────────────────────────────────────────────
# From Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020).
# Pulls same-class embeddings together and pushes different-class embeddings
# apart in the projection space. Combined with CE for end-to-end training.

class SupConLoss(nn.Module):
    """Supervised contrastive loss on L2-normalized embeddings."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [B, D] L2-normalized projected embeddings
            labels: [B] integer class labels
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Cosine similarity matrix scaled by temperature
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # [B, B]

        # Mask: exclude self-comparisons
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # Positive mask: same label, different sample (no inplace ops for autograd)
        labels_col = labels.unsqueeze(0)  # [1, B]
        labels_row = labels.unsqueeze(1)  # [B, 1]
        pos_mask = ((labels_row == labels_col) & ~self_mask).float()  # [B, B]

        # Numerical stability: mask self-similarity with -inf, then subtract max
        neg_inf = torch.where(self_mask, torch.tensor(-1e9, device=device), torch.zeros(1, device=device))
        sim_stable = sim + neg_inf
        sim_max = sim_stable.detach().max(dim=1, keepdim=True).values
        sim_stable = sim_stable - sim_max

        # Denominator: sum of exp(sim) over all non-self pairs
        not_self = (~self_mask).float()
        exp_sim = torch.exp(sim_stable) * not_self
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Log probability of positives
        log_prob = sim_stable - log_sum_exp  # [B, B]

        # Mean log-prob over positive pairs for each anchor
        n_positives = pos_mask.sum(dim=1)  # [B]
        valid = n_positives > 0
        mean_log_prob = (log_prob * pos_mask).sum(dim=1) / (n_positives + 1e-8)

        loss = -mean_log_prob[valid].mean()
        return loss


# ── Config ────────────────────────────────────────────────────────────────────
DATASET_1       = Path("/Users/kenmarfrancisco/.cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset")
DATASET_2       = Path("/Users/kenmarfrancisco/.cache/kagglehub/datasets/xhlulu/140k-real-and-fake-faces/versions/2/real_vs_fake/real-vs-fake")
SAVE_DIR        = Path("./model")
EPOCHS          = 15
BATCH_SIZE      = 64
ACCUM_STEPS     = 1       # effective batch size = 64
LR              = 2e-4
PATIENCE        = 3       # early stop after 3 epochs without improvement
LABEL_SMOOTHING = 0.1     # prevents overconfident predictions on CE component
SUPCON_WEIGHT   = 0.7     # SupCon dominates — shapes embedding space
CE_WEIGHT       = 0.3     # CE maintains decision boundary
SUPCON_TEMP     = 0.07    # temperature for SupCon (lower = harder negatives)
PROJ_DIM        = 128     # projection head output dim for SupCon

# ── Processor ─────────────────────────────────────────────────────────────────
processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
IMG_SIZE = processor.size.get("shortest_edge", 224)


# ── High-Pass Filter (frequency-domain augmentation) ─────────────────────────
# Extracts high-frequency residuals that expose GAN/diffusion artifacts.
# Applied as a training augmentation: the model sometimes sees the raw RGB,
# sometimes the high-pass filtered version, forcing it to learn both spatial
# and frequency-domain cues.

def high_pass_filter(img_np):
    """Apply a high-pass filter to a uint8 HWC numpy image, return uint8 HWC."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Subtract a heavy blur to isolate high-frequency residuals
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    residual = gray - low
    # Normalize to 0-255 range
    residual = residual - residual.min()
    mx = residual.max()
    if mx > 0:
        residual = (residual / mx * 255).astype(np.uint8)
    else:
        residual = np.zeros_like(gray, dtype=np.uint8)
    # Stack to 3 channels so it fits the same pipeline
    return np.stack([residual, residual, residual], axis=-1)


# ── Social Media Degradation (H.264-style) ───────────────────────────────────
# Simulates the compression pipeline of platforms like Instagram, TikTok, etc.
# Real deepfakes in the wild go through upload → transcode → re-download.

class SocialMediaDegradation(A.ImageOnlyTransform):
    """Simulates social media upload: downscale → JPEG → upscale → blur."""

    def __init__(self, p=0.5):
        super().__init__(p=p)

    def apply(self, img, **params):
        h, w = img.shape[:2]

        # 1. Downscale to simulate transcoding resolution drop
        scale = np.random.uniform(0.4, 0.8)
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 2. JPEG compression (simulates H.264 I-frame quality)
        quality = np.random.randint(20, 70)
        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
        small = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        # 3. Upscale back to original size (bilinear = typical browser/app behavior)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        # 4. Light Gaussian blur (simulates sharpening → recompression softness)
        if np.random.random() < 0.5:
            ksize = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        return img

    def get_transform_init_args_names(self):
        return ()


# ── Albumentations Training Pipeline ─────────────────────────────────────────
# Replaces the old torchvision Compose. Runs entirely on CPU (numpy/OpenCV)
# so it doesn't block the MPS GPU.
#
# Order rationale:
#   1. Geometric (flip/rotate/crop) — shape changes first
#   2. Color jitter — photometric variation
#   3. Degradation (social media, JPEG, blur) — simulate real-world distribution
#   All are independent and stochastic per-sample.

train_augment = A.Compose([
    # ── Geometric ──
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.85, 1.0), p=1.0),

    # ── Photometric ──
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0, p=0.5),

    # ── Degradation (social media simulation) ──
    SocialMediaDegradation(p=0.4),
    A.ImageCompression(quality_range=(30, 95), p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.Downscale(scale_range=(0.5, 0.9), p=0.3),
])

# Probability of replacing the RGB image with its high-pass filtered version
HIGH_PASS_P = 0.15


def train_transform(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_np = np.array(img)

    # Albumentations augmentation
    img_np = train_augment(image=img_np)["image"]

    # Stochastic high-pass filter: forces model to learn frequency-domain cues
    if np.random.random() < HIGH_PASS_P:
        img_np = high_pass_filter(img_np)

    img = Image.fromarray(img_np)
    return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)


def val_transform(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    def remap_labels(ds):
        """Remap so fake->1 (Deepfake) and real->0 (Realism) regardless of folder casing."""
        class_to_idx = ds.class_to_idx
        remap = {}
        for name, idx in class_to_idx.items():
            if name.lower() == "fake":
                remap[idx] = 1   # Deepfake
            else:
                remap[idx] = 0   # Realism
        ds.targets = [remap[t] for t in ds.targets]
        ds.samples = [(path, remap[label]) for path, label in ds.samples]

    # Dataset 1: manjilkarki deepfake (Fake/Real folders)
    train_ds1 = ImageFolder(DATASET_1 / "Train",      transform=train_transform)
    val_ds1   = ImageFolder(DATASET_1 / "Validation", transform=val_transform)
    remap_labels(train_ds1)
    remap_labels(val_ds1)

    # Dataset 2: 140k StyleGAN faces (fake/real folders)
    train_ds2 = ImageFolder(DATASET_2 / "train", transform=train_transform)
    val_ds2   = ImageFolder(DATASET_2 / "valid", transform=val_transform)
    remap_labels(train_ds2)
    remap_labels(val_ds2)

    # Combine
    train_ds = ConcatDataset([train_ds1, train_ds2])
    val_ds   = ConcatDataset([val_ds1, val_ds2])

    # ── Build group indices for balanced sampling ─────────────────────────
    # 4 groups: DS1-Real, DS1-Fake, DS2-Real, DS2-Fake
    # Global index in ConcatDataset: DS1 occupies [0, len(ds1)),
    #                                DS2 occupies [len(ds1), len(ds1)+len(ds2))
    ds1_offset = 0
    ds2_offset = len(train_ds1)

    group_ds1_real = [ds1_offset + i for i, t in enumerate(train_ds1.targets) if t == 0]
    group_ds1_fake = [ds1_offset + i for i, t in enumerate(train_ds1.targets) if t == 1]
    group_ds2_real = [ds2_offset + i for i, t in enumerate(train_ds2.targets) if t == 0]
    group_ds2_fake = [ds2_offset + i for i, t in enumerate(train_ds2.targets) if t == 1]

    group_indices = [group_ds1_real, group_ds1_fake, group_ds2_real, group_ds2_fake]

    print(f"Dataset 1 — Train: {len(train_ds1)} | Val: {len(val_ds1)}")
    print(f"Dataset 2 — Train: {len(train_ds2)} | Val: {len(val_ds2)}")
    print(f"Combined  — Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Balanced sampling groups:")
    print(f"  DS1-Real: {len(group_ds1_real)} | DS1-Fake: {len(group_ds1_fake)}")
    print(f"  DS2-Real: {len(group_ds2_real)} | DS2-Fake: {len(group_ds2_fake)}")
    print(f"  Per-group per batch: {BATCH_SIZE // 4} (batch_size={BATCH_SIZE}, 4 groups)")

    balanced_sampler = BalancedBatchSampler(group_indices, batch_size=BATCH_SIZE)
    if device.type == "cuda":
        num_workers, pin_memory = 4, True
    else:
        num_workers, pin_memory = 0, False
    train_loader = DataLoader(train_ds, batch_sampler=balanced_sampler, num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # ── Model ─────────────────────────────────────────────────────────────────
    dora_checkpoint = SAVE_DIR / "dora"
    head_checkpoint = SAVE_DIR / "head_weights.pt"
    resuming = dora_checkpoint.exists() and head_checkpoint.exists()

    if resuming:
        print(f"Resuming from saved DoRA weights at {dora_checkpoint}")
        from peft import PeftModel
        print("Loading base CLIP model...")
        clip_vision = CLIPVisionModel.from_pretrained(CLIP_MODEL_ID)
        print("Loading DoRA adapter weights...")
        clip_vision = PeftModel.from_pretrained(clip_vision, dora_checkpoint, is_trainable=True)
        model = DeepfakeDetector(clip_vision)
        print("Loading head weights...")
        head_data = torch.load(head_checkpoint, map_location=device, weights_only=True)
        model.fft_branch.load_state_dict(head_data["fft_branch"])
        model.classifier.load_state_dict(head_data["classifier"])
    else:
        print(f"Downloading/loading {CLIP_MODEL_ID} (this may take a moment)...")
        clip_vision = CLIPVisionModel.from_pretrained(CLIP_MODEL_ID)

        # DoRA on ALL attention layers (including early layers for frequency cues).
        # CLIP attention uses q_proj/k_proj/v_proj/out_proj naming.
        dora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none",
            use_dora=True,
        )
        clip_vision = get_peft_model(clip_vision, dora_config)
        model = DeepfakeDetector(clip_vision)

    # Print parameter counts
    model.clip_vision.print_trainable_parameters()
    fft_params = sum(p.numel() for p in model.fft_branch.parameters())
    cls_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"FFT branch: {fft_params:,} params | Classifier: {cls_params:,} params (all trainable)")

    # ── Projection Head (training-only, for SupCon loss) ─────────────────────
    # Maps fused embeddings to a lower-dim space where contrastive loss is
    # computed. Discarded after training — the representation before it is
    # what matters for classification (Khosla et al., 2020).
    projection_head = nn.Sequential(
        nn.Linear(model.fused_dim, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, PROJ_DIM),
    )

    model = model.to(device)
    projection_head = projection_head.to(device)

    # Load projection head state if resuming
    if resuming and "projection_head" in head_data:
        projection_head.load_state_dict(head_data["projection_head"])
        print("Restored projection head state")

    proj_params = sum(p.numel() for p in projection_head.parameters())
    print(f"Projection head: {proj_params:,} params (training-only)")
    print(f"Loss: {SUPCON_WEIGHT:.0%} SupCon (τ={SUPCON_TEMP}) + {CE_WEIGHT:.0%} CE (label_smooth={LABEL_SMOOTHING})")

    # ── Optimizer + CosineAnnealingWarmRestarts ──────────────────────────────
    trainable_params = (
        [p for p in model.parameters() if p.requires_grad]
        + list(projection_head.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

    # CosineAnnealingWarmRestarts — resume-friendly, restarts LR every T_0 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,          # restart every 5 epochs
        T_mult=1,       # keep restart period constant
        eta_min=1e-6,   # minimum LR
    )

    # ── Loss Functions ───────────────────────────────────────────────────────
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    supcon_criterion = SupConLoss(temperature=SUPCON_TEMP)

    # ── Restore optimizer/scheduler state if resuming ─────────────────────────
    train_state_path = SAVE_DIR / "train_state.pt"
    start_epoch = 0
    best_val_auc = 0.0
    patience_counter = 0

    if resuming and train_state_path.exists():
        state = torch.load(train_state_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_val_auc = state["best_val_auc"]
        patience_counter = state["patience_counter"]
        print(f"Restored training state: epoch={start_epoch}, best_val_auc={best_val_auc:.4f}")
    elif resuming:
        print("Warning: DoRA weights found but no train_state.pt — optimizer/scheduler start fresh")

    # ── History for plotting ─────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_auc": []}

    # ── Training loop ─────────────────────────────────────────────────────────

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        projection_head.train()
        total_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{EPOCHS} [train]", unit="batch")
        for step, (pixels, labels) in pbar:
            pixels, labels = pixels.to(device), labels.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(pixel_values=pixels, return_fused=True)

                # CE loss on classification logits
                ce_loss = ce_criterion(outputs.logits, labels)

                # SupCon loss on projected, L2-normalized fused embeddings
                projected = projection_head(outputs.fused.float())
                projected = F.normalize(projected, dim=1)
                sc_loss = supcon_criterion(projected, labels)

                loss = (CE_WEIGHT * ce_loss + SUPCON_WEIGHT * sc_loss) / ACCUM_STEPS

            loss.backward()

            # Gradient accumulation — step every ACCUM_STEPS
            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_STEPS
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{total_loss/(step+1):.4f}",
                             acc=f"{correct/total:.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_acc = correct / total
        train_loss_avg = total_loss / len(train_loader)

        # Step scheduler per epoch (CosineAnnealingWarmRestarts is epoch-level)
        scheduler.step(epoch + 1)

        # ── Validate (AUC-ROC as primary metric) ────────────────────────────
        model.eval()
        val_all_probs, val_all_labels = [], []
        val_loss_sum = 0.0
        with torch.no_grad():
            for pixels, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", unit="batch"):
                pixels, labels = pixels.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(pixel_values=pixels)
                    val_loss = ce_criterion(outputs.logits, labels)
                val_loss_sum += val_loss.item()
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]
                val_all_probs.extend(probs.cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

        val_probs_np = np.array(val_all_probs)
        val_labels_np = np.array(val_all_labels)
        val_auc = roc_auc_score(val_labels_np, val_probs_np)
        val_acc = ((val_probs_np >= 0.5).astype(int) == val_labels_np).mean()
        val_loss_avg = val_loss_sum / len(val_loader)

        history["train_loss"].append(train_loss_avg)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss_avg)
        history["val_auc"].append(val_auc)

        print(f"\nEpoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")

        # ── Save training state every epoch (for safe resume) ────────────────
        # ── Save best + versioned snapshots + early stopping ─────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save DoRA adapter weights
            model.clip_vision.save_pretrained(SAVE_DIR / "dora")
            # Save FFT branch + classifier head + projection head
            torch.save({
                "fft_branch": model.fft_branch.state_dict(),
                "classifier": model.classifier.state_dict(),
                "id2label": model.id2label,
                "projection_head": projection_head.state_dict(),
            }, SAVE_DIR / "head_weights.pt")
            # Save optimizer/scheduler state in sync with best weights
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_auc": best_val_auc,
                "patience_counter": patience_counter,
            }, SAVE_DIR / "train_state.pt")
            # Versioned snapshot for rollback
            snapshot_dir = SAVE_DIR / f"dora_epoch{epoch+1}_auc{val_auc:.4f}"
            model.clip_vision.save_pretrained(snapshot_dir)
            print(f"  Saved best DoRA weights (val_auc={val_auc:.4f})")
            print(f"  Snapshot: {snapshot_dir.name}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("  Early stopping triggered.")
                break

        print()

    # ── Merge best DoRA weights into base model for inference ────────────
    dora_path = SAVE_DIR / "dora"
    if dora_path.exists():
        from peft import PeftModel
        base_clip = CLIPVisionModel.from_pretrained(CLIP_MODEL_ID)
        merged_clip = PeftModel.from_pretrained(base_clip, dora_path)
        merged_clip = merged_clip.merge_and_unload()
        # Build final inference model and save
        final_model = DeepfakeDetector(merged_clip)
        head_data = torch.load(SAVE_DIR / "head_weights.pt", weights_only=True)
        final_model.fft_branch.load_state_dict(head_data["fft_branch"])
        final_model.classifier.load_state_dict(head_data["classifier"])
        final_model.save_model(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)

    print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f}")
    print(f"Model saved to {SAVE_DIR}")

    # ── Plot training history ──────────────────────────────────────────────
    if len(history["train_loss"]) > 0:
        epochs_range = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs_range, history["train_loss"], "o-", label="Train Loss")
        ax1.plot(epochs_range, history["val_loss"], "o-", label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss per Epoch")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, history["train_acc"], "o-", label="Train Acc")
        ax2.plot(epochs_range, history["val_auc"], "o-", label="Val AUC")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.set_title("Train Acc / Val AUC per Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=150)
        print("Training plot saved to training_history.png")
