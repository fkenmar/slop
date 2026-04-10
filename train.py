"""
Fine-tunes prithivMLmods/Deep-Fake-Detector-v2-Model on the Kaggle deepfake dataset.

Optimizations:
  1. Layer Freezing       — embedding + first 8 blocks frozen (2x speedup)
  2. LoRA                 — 2M trainable params instead of 86M
  3. Gradient Accumulation — simulates batch_size=64 with less memory
  4. OneCycleLR           — faster convergence than CosineAnnealing
  5. Early Stopping       — stops if validation accuracy plateaus
  6. AMP (float16)        — 2x memory savings on MPS

Saves the fine-tuned model to ./model/ for use in app.py.
"""

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import LoraConfig, get_peft_model
import torch
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID        = "prithivMLmods/Deep-Fake-Detector-v2-Model"
DATASET         = Path("/Users/kenmarfrancisco/.cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset")
SAVE_DIR        = Path("./model")
EPOCHS          = 5
BATCH_SIZE      = 32
ACCUM_STEPS     = 2       # effective batch size = 32 * 2 = 64
LR              = 2e-4    # higher LR is safe with LoRA + OneCycle
FREEZE_BLOCKS   = 8       # freeze 8 of 12 blocks — only top 4 learn
PATIENCE        = 2       # early stop after 2 epochs without improvement

# ── Processor ─────────────────────────────────────────────────────────────────
processor = ViTImageProcessor.from_pretrained(MODEL_ID)

def transform(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = ImageFolder(DATASET / "Train",      transform=transform)
    val_ds   = ImageFolder(DATASET / "Validation", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {train_ds.classes}")

    # ── Model ─────────────────────────────────────────────────────────────────
    base_model = ViTForImageClassification.from_pretrained(MODEL_ID)

    # 1. Freeze embeddings + first N transformer blocks
    for param in base_model.vit.embeddings.parameters():
        param.requires_grad = False
    for i in range(FREEZE_BLOCKS):
        for param in base_model.vit.encoder.layer[i].parameters():
            param.requires_grad = False

    # 2. LoRA on the unfrozen attention layers (query + value projections)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 3. Enable gradient checkpointing via HF API
    model.gradient_checkpointing_enable()

    model = model.to(device)

    # ── Optimizer + OneCycleLR ────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

    # 4. OneCycleLR — ramps LR up then down for faster convergence
    steps_per_epoch = len(train_loader) // ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
    )

    # ── History for plotting ─────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        for step, (pixels, labels) in enumerate(train_loader):
            pixels, labels = pixels.to(device), labels.to(device)

            # 5. AMP autocast — float16 on MPS for speed + memory
            with torch.autocast(device_type="mps", dtype=torch.float16):
                outputs = model(pixel_values=pixels, labels=labels)
                loss = outputs.loss / ACCUM_STEPS

            loss.backward()

            # Gradient accumulation — step every ACCUM_STEPS
            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_STEPS
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (step + 1) % 200 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | "
                      f"Loss: {total_loss/(step+1):.4f} | Acc: {correct/total:.4f} | LR: {lr_now:.2e}")

        train_acc = correct / total

        train_loss_avg = total_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for pixels, labels in val_loader:
                pixels, labels = pixels.to(device), labels.to(device)
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    outputs = model(pixel_values=pixels, labels=labels)
                val_loss_sum += outputs.loss.item()
                preds = outputs.logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss_avg = val_loss_sum / len(val_loader)

        history["train_loss"].append(train_loss_avg)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss_avg)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # ── Save best + early stopping ────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model.save_pretrained(SAVE_DIR / "lora")
            print(f"  Saved best LoRA weights (val_acc={val_acc:.4f}) to {SAVE_DIR / 'lora'}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("  Early stopping triggered.")
                break

        print()

    # ── Merge best LoRA weights into base model for inference ────────────
    lora_path = SAVE_DIR / "lora"
    if lora_path.exists():
        from peft import PeftModel
        base_for_merge = ViTForImageClassification.from_pretrained(MODEL_ID)
        merged = PeftModel.from_pretrained(base_for_merge, lora_path)
        merged = merged.merge_and_unload()
        merged.save_pretrained(SAVE_DIR)
        processor.save_pretrained(SAVE_DIR)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
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
        ax2.plot(epochs_range, history["val_acc"], "o-", label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy per Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=150)
        print("Training plot saved to training_history.png")
