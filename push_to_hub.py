"""
Push fine-tuned deepfake detector weights to Hugging Face Hub.

Uploads:
  - model/dora/          (DoRA adapters, 12MB)
  - model/head_weights.pt (fusion head, 2.5MB)
  - model/model.safetensors (full fine-tuned checkpoint, 327MB)
  - model/config.json + preprocessor_config.json
  - model/blaze_face_short_range.tflite, face_landmarker.task, yolov8n-face.pt
  - model.py             (custom architecture, required for trust_remote_code)

Skips clip_vision/ — users pull openai/clip-vit-large-patch14 from HF directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

REPO_ID = "knmrfr/deepfake-detector"
MODEL_DIR = Path("model")

SKIP = {
    "clip_vision",          # already on HF as openai/clip-vit-large-patch14
    "lora",                 # dora supersedes lora
    "train_state.pt",       # training state, not needed for inference
    # epoch checkpoints
    "dora_epoch1_auc0.9984",
    "dora_epoch2_auc0.9994",
    "dora_epoch4_auc0.9994",
    "dora_epoch5_auc0.9995",
    "dora_epoch6_auc0.9979",
    "dora_epoch7_auc0.9986",
    "dora_epoch8_auc0.9989",
    "dora_epoch9_auc0.9993",
}

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create repo if it doesn't exist
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
print(f"Repo ready: https://huggingface.co/{REPO_ID}")

# Upload model.py (custom architecture)
print("Uploading model.py...")
api.upload_file(
    path_or_fileobj="model.py",
    path_in_repo="model.py",
    repo_id=REPO_ID,
)

# Upload model/ files, skipping clip_vision and old checkpoints
def iter_files(base: Path):
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        top = relative.parts[0]
        if top in SKIP:
            continue
        yield path, relative

files = list(iter_files(MODEL_DIR))
print(f"Uploading {len(files)} files from model/...")

for local_path, relative in files:
    repo_path = f"model/{relative}"
    print(f"  {repo_path} ({local_path.stat().st_size / 1e6:.1f} MB)")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=REPO_ID,
    )

print(f"\nDone! https://huggingface.co/{REPO_ID}")
