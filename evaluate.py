"""
Evaluation protocol for the deepfake detector.

Primary metrics: AUC-ROC and EER (Equal Error Rate).
Runs three evaluation modes:
  1. Combined    — all test datasets pooled
  2. Per-dataset — metrics for each dataset separately
  3. LOO         — Leave-One-Out cross-generator analysis: evaluate on each
                   dataset in isolation to measure generalization to unseen
                   generator types

Uses Test-Time Augmentation (TTA): averages original + flipped predictions.
"""

import os
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from transformers import CLIPImageProcessor
from model import DeepfakeDetector, CLIP_MODEL_ID
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import torch
import numpy as np
import matplotlib.pyplot as plt


MODEL_DIR  = Path("./model")
DATASET_1  = Path("/Users/kenmarfrancisco/.cache/kagglehub/datasets/manjilkarki/deepfake-and-real-images/versions/1/Dataset")
DATASET_2  = Path("/Users/kenmarfrancisco/.cache/kagglehub/datasets/xhlulu/140k-real-and-fake-faces/versions/2/real_vs_fake/real-vs-fake")
BATCH_SIZE = 64

processor = (
    CLIPImageProcessor.from_pretrained(MODEL_DIR)
    if (MODEL_DIR / "preprocessor_config.json").exists()
    else CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
)


def transform(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_eer(fpr, tpr):
    """Compute Equal Error Rate from ROC curve arrays."""
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer, idx


def evaluate_predictions(all_labels, all_probs, name=""):
    """Compute and print AUC-ROC, EER, classification report, confusion matrix."""
    all_preds = (all_probs >= 0.5).astype(int)

    header = f" {name} " if name else ""
    print(f"\n{'=' * 60}")
    print(f"{'=' * 20}{header:^20}{'=' * 20}")
    print(f"{'=' * 60}")

    # ── AUC-ROC & EER ─────────────────────────────────────────────
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    eer, eer_idx = compute_eer(fpr, tpr)
    eer_thresh = thresholds[eer_idx] if eer_idx < len(thresholds) else 0.5

    print(f"\n  AUC-ROC:         {auc:.4f}")
    print(f"  EER:             {eer:.4f} ({eer*100:.2f}%)")
    print(f"  EER threshold:   {eer_thresh:.4f}")

    accuracy = (all_preds == all_labels).mean()
    print(f"  Accuracy (@0.5): {accuracy:.4f} ({accuracy*100:.1f}%)")

    # ── Classification Report ─────────────────────────────────────
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Realism', 'Deepfake'])}")

    # ── Confusion Matrix ──────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    print(f"  {'':>12} {'Pred Real':>12} {'Pred Fake':>12}")
    print(f"  {'True Real':>12} {cm[0][0]:>12} {cm[0][1]:>12}")
    print(f"  {'True Fake':>12} {cm[1][0]:>12} {cm[1][1]:>12}")

    # ── Threshold Analysis ────────────────────────────────────────
    print(f"\n  {'Threshold':>10} {'Accuracy':>10} {'Fake Recall':>12} {'Real Recall':>12} {'Fake Prec':>10}")
    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        preds_t = (all_probs >= thresh).astype(int)
        acc_t = (preds_t == all_labels).mean()
        fake_mask = all_labels == 1
        real_mask = all_labels == 0
        fake_recall = (preds_t[fake_mask] == 1).mean() if fake_mask.sum() > 0 else 0
        real_recall = (preds_t[real_mask] == 0).mean() if real_mask.sum() > 0 else 0
        fake_prec_mask = preds_t == 1
        fake_prec = (all_labels[fake_prec_mask] == 1).mean() if fake_prec_mask.sum() > 0 else 0
        marker = " <-- EER" if abs(thresh - eer_thresh) < 0.05 else ""
        print(f"  {thresh:>10.2f} {acc_t:>10.4f} {fake_recall:>12.4f} {real_recall:>12.4f} {fake_prec:>10.4f}{marker}")

    return {"auc": auc, "eer": eer, "eer_thresh": eer_thresh,
            "fpr": fpr, "tpr": tpr, "accuracy": accuracy, "cm": cm,
            "all_probs": all_probs, "all_labels": all_labels}


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    """Run TTA inference (original + flipped), return (probs, labels)."""
    all_probs, all_labels = [], []
    with torch.no_grad():
        for step, (pixels, labels) in enumerate(loader):
            pixels = pixels.to(device)

            outputs_orig = model(pixel_values=pixels)
            probs_orig = torch.softmax(outputs_orig.logits, dim=1)

            pixels_flip = torch.flip(pixels, dims=[-1])
            outputs_flip = model(pixel_values=pixels_flip)
            probs_flip = torch.softmax(outputs_flip.logits, dim=1)

            probs = ((probs_orig + probs_flip) / 2).cpu().numpy()
            all_probs.extend(probs[:, 1])
            all_labels.extend(labels.numpy())

            if (step + 1) % 50 == 0:
                print(f"  Processed {(step+1) * BATCH_SIZE} images...")

    return np.array(all_probs), np.array(all_labels)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {MODEL_DIR}")

    model = DeepfakeDetector.from_pretrained(MODEL_DIR, device=str(device))
    model = model.to(device)
    model.eval()

    def remap_labels(ds):
        remap = {}
        for name, idx in ds.class_to_idx.items():
            remap[idx] = 1 if name.lower() == "fake" else 0
        ds.targets = [remap[t] for t in ds.targets]
        ds.samples = [(p, remap[l]) for p, l in ds.samples]

    # ── Load datasets ─────────────────────────────────────────────────────────
    test_ds1 = ImageFolder(DATASET_1 / "Test", transform=transform)
    remap_labels(test_ds1)
    test_ds2 = ImageFolder(DATASET_2 / "test", transform=transform)
    remap_labels(test_ds2)

    datasets = {
        "DS1 (Deepfake-Real)": test_ds1,
        "DS2 (140k StyleGAN)": test_ds2,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 1: Combined evaluation (all datasets pooled)
    # ══════════════════════════════════════════════════════════════════════════
    test_ds = ConcatDataset([test_ds1, test_ds2])
    combined_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"\nTest images: {len(test_ds)} (DS1: {len(test_ds1)}, DS2: {len(test_ds2)})")
    print("\nRunning combined evaluation...")

    combined_probs, combined_labels = run_inference(model, combined_loader, device)
    combined_results = evaluate_predictions(combined_labels, combined_probs, name="COMBINED")

    # ══════════════════════════════════════════════════════════════════════════
    # MODE 2 & 3: Per-dataset (LOO cross-generator) evaluation
    # Each dataset is evaluated in isolation — measures how well the model
    # generalizes to each generator type without seeing it alongside others.
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "#" * 60)
    print("#" + " LEAVE-ONE-OUT CROSS-GENERATOR ANALYSIS ".center(58) + "#")
    print("#" * 60)
    print("\nEvaluating model on each dataset in isolation.")
    print("This reveals whether the model generalizes across generator types,")
    print("or if high combined metrics are inflated by one easy dataset.\n")

    per_ds_results = {}
    for ds_name, ds in datasets.items():
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"\n--- {ds_name}: {len(ds)} images ---")
        probs, labels = run_inference(model, loader, device)
        per_ds_results[ds_name] = evaluate_predictions(labels, probs, name=ds_name)

    # ── LOO Summary Table ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" LOO SUMMARY ".center(60, "="))
    print("=" * 60)
    print(f"  {'Dataset':<25} {'AUC-ROC':>10} {'EER':>10} {'Acc @0.5':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for ds_name, r in per_ds_results.items():
        print(f"  {ds_name:<25} {r['auc']:>10.4f} {r['eer']:>10.4f} {r['accuracy']:>10.4f}")
    print(f"  {'COMBINED':<25} {combined_results['auc']:>10.4f} {combined_results['eer']:>10.4f} {combined_results['accuracy']:>10.4f}")

    # Check for generalization gap
    aucs = [r["auc"] for r in per_ds_results.values()]
    gap = max(aucs) - min(aucs)
    if gap > 0.05:
        print(f"\n  WARNING: AUC gap of {gap:.3f} between datasets — model may be")
        print(f"  overfitting to one generator type. Consider adding more diverse data.")
    else:
        print(f"\n  AUC gap: {gap:.3f} — good cross-generator consistency.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    n_plots = 2 + len(per_ds_results)  # ROC combined, confidence dist, + ROC per dataset
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    # 1. Combined ROC with EER point
    r = combined_results
    axes[0].plot(r["fpr"], r["tpr"], label=f'Combined AUC={r["auc"]:.3f}')
    eer_val, eer_idx = compute_eer(r["fpr"], r["tpr"])
    axes[0].plot(r["fpr"][eer_idx], r["tpr"][eer_idx], "ro", markersize=8,
                 label=f"EER={eer_val:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC — Combined")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Confidence distribution
    fake_probs = r["all_probs"][r["all_labels"] == 1]
    real_probs = r["all_probs"][r["all_labels"] == 0]
    axes[1].hist(real_probs, bins=40, alpha=0.6, label="Real", color="#4caf82")
    axes[1].hist(fake_probs, bins=40, alpha=0.6, label="Fake", color="#e05c5c")
    axes[1].axvline(0.5, color="white", linestyle="--", alpha=0.7, label="Threshold")
    axes[1].axvline(r["eer_thresh"], color="yellow", linestyle="--", alpha=0.7, label=f"EER thresh={r['eer_thresh']:.2f}")
    axes[1].set_xlabel("Deepfake Probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3+. Per-dataset ROC curves
    for i, (ds_name, r) in enumerate(per_ds_results.items()):
        ax = axes[2 + i]
        ax.plot(r["fpr"], r["tpr"], label=f'AUC={r["auc"]:.3f}')
        eer_val, eer_idx = compute_eer(r["fpr"], r["tpr"])
        ax.plot(r["fpr"][eer_idx], r["tpr"][eer_idx], "ro", markersize=8,
                label=f"EER={eer_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"ROC — {ds_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("evaluation_report.png", dpi=150)
    print("\nPlots saved to evaluation_report.png")
