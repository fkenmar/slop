---
license: mit
tags:
  - deepfake-detection
  - image-classification
  - vision-transformer
  - clip
pipeline_tag: image-classification
---

# Deepfake Detector

Two-branch fusion model for detecting AI-generated / deepfake faces.

- **Branch 1** — CLIP ViT-L/14 vision encoder: high-level semantic features
- **Branch 2** — FFT magnitude CNN: frequency-domain artifacts (GAN spectral peaks, diffusion noise patterns)
- **AUC:** 0.9995

## Install

```bash
pip install torch transformers huggingface_hub pillow ultralytics opencv-python peft
```

## Usage

```python
import torch
from PIL import Image
from transformers import CLIPImageProcessor
from huggingface_hub import hf_hub_download
import sys, os

# Download model.py from the repo
model_py = hf_hub_download(repo_id="knmrfr/deepfake-detector", filename="model.py")
sys.path.insert(0, os.path.dirname(model_py))

from model import DeepfakeDetector

# Load model
model = DeepfakeDetector.from_pretrained("knmrfr/deepfake-detector")
model.eval()

# Load processor
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Run inference
image = Image.open("face.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    predicted_idx = int(torch.argmax(probs))

label = model.id2label[predicted_idx]
confidence = round(float(probs[predicted_idx]) * 100, 1)
print(f"{label} ({confidence}%)")  # e.g. "Deepfake (98.3%)"
```

## Labels

| ID | Label |
|----|-------|
| 0  | Realism (real face) |
| 1  | Deepfake (AI-generated) |

## Notes

- Input should be a cropped face image. The model is trained on face crops, not full scene images.
- The CLIP backbone (`openai/clip-vit-large-patch14`) is loaded automatically on first use (~1.7GB download).
