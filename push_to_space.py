"""Push the app to a Hugging Face Docker Space."""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

SPACE_ID = "knmrfr/deepfake-detector-demo"

api = HfApi(token=os.getenv("HF_TOKEN"))

api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="docker", exist_ok=True)
print(f"Space ready: https://huggingface.co/spaces/{SPACE_ID}")

files = {
    "app.py":        "app.py",
    "model.py":      "model.py",
    "requirements.txt": "requirements.txt",
    "Dockerfile":    "Dockerfile",
    ".dockerignore": ".dockerignore",
    "SPACE_README.md": "README.md",
}

for local, remote in files.items():
    api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=SPACE_ID, repo_type="space")
    print(f"  {remote} uploaded")

# Upload the frontend source so Docker can build it
api.upload_folder(
    folder_path="frontend",
    path_in_repo="frontend",
    repo_id=SPACE_ID,
    repo_type="space",
    ignore_patterns=["node_modules/**", "dist/**"],
)
print("  frontend/ uploaded")

print(f"\nDone! https://huggingface.co/spaces/{SPACE_ID}")
print("HF will build the Docker image — check the Logs tab for progress.")
