"""
DeepfakeDetector: Two-branch fusion architecture for deepfake detection.

Branch 1 — CLIP ViT vision encoder:
    Extracts high-level semantic features from RGB pixels. CLIP's pretraining
    on 400M image-text pairs gives strong cross-generator generalization
    (GANs, diffusion models, face-swap).

Branch 2 — FFT magnitude → lightweight CNN:
    Computes the 2D FFT magnitude spectrum from the input, capturing
    frequency-domain artifacts invisible in pixel space (GAN spectral peaks,
    diffusion noise patterns, blending boundary artifacts).

The two branch embeddings are concatenated and classified by a fusion MLP head.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from transformers import CLIPVisionModel

# Use ViT-L/14 — same backbone as UnivFD (Ojha et al., CVPR 2023).
# Switch to "openai/clip-vit-base-patch16" if MPS memory is tight.
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"


@dataclass
class DetectorOutput:
    logits: torch.Tensor
    fused: torch.Tensor = None  # pre-classifier fused embedding (for SupCon)


class DeepfakeDetector(nn.Module):
    """Two-branch deepfake detector: CLIP (spatial) + FFT (frequency)."""

    def __init__(self, clip_vision, fft_embed_dim=128, num_classes=2):
        super().__init__()
        self.clip_vision = clip_vision

        # Resolve hidden size for both regular and peft-wrapped models
        try:
            # PeftModel stores base config at base_model.model.config
            clip_hidden = clip_vision.base_model.model.config.hidden_size
        except AttributeError:
            clip_hidden = clip_vision.config.hidden_size

        # Branch 2: lightweight CNN on FFT magnitude spectrum
        self.fft_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, fft_embed_dim),
            nn.ReLU(inplace=True),
        )

        # Fusion classification head
        self.classifier = nn.Sequential(
            nn.Linear(clip_hidden + fft_embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        self.id2label = {0: "Realism", 1: "Deepfake"}

    @property
    def fused_dim(self):
        """Dimension of the fused embedding (clip_hidden + fft_embed_dim)."""
        return self.classifier[0].in_features

    def forward(self, pixel_values, return_fused=False):
        # Branch 1: CLIP spatial features
        clip_out = self.clip_vision(pixel_values=pixel_values)
        clip_embed = clip_out.pooler_output  # [B, clip_hidden]

        # Branch 2: FFT frequency features
        gray = pixel_values.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        fft = torch.fft.fft2(gray)
        fft_mag = torch.log1p(torch.abs(torch.fft.fftshift(fft)))  # [B, 1, H, W]
        fft_embed = self.fft_branch(fft_mag)  # [B, fft_embed_dim]

        # Fusion
        fused = torch.cat([clip_embed, fft_embed], dim=1)
        logits = self.classifier(fused)
        return DetectorOutput(logits=logits, fused=fused if return_fused else None)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save_model(self, save_dir):
        """Save the full model for inference (call after DoRA merge)."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.clip_vision.save_pretrained(save_dir / "clip_vision")
        torch.save({
            "fft_branch": self.fft_branch.state_dict(),
            "classifier": self.classifier.state_dict(),
            "id2label": self.id2label,
        }, save_dir / "head_weights.pt")

    @classmethod
    def from_pretrained(cls, save_dir, device="cpu"):
        """Load a trained model for inference."""
        save_dir = Path(save_dir)
        clip_vision = CLIPVisionModel.from_pretrained(save_dir / "clip_vision")
        head_data = torch.load(
            save_dir / "head_weights.pt", map_location=device, weights_only=True
        )
        model = cls(clip_vision)
        model.fft_branch.load_state_dict(head_data["fft_branch"])
        model.classifier.load_state_dict(head_data["classifier"])
        model.id2label = head_data["id2label"]
        return model
