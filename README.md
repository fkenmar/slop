# Slop - a modern Deepfake Detector

An image-based deepfake detection web app that classifies face images as **Real** or **Deepfake** using a two-branch fusion model: a DoRA-fine-tuned CLIP ViT-L/14 backbone (spatial features) combined with a lightweight CNN over the 2D FFT magnitude spectrum (frequency artifacts). A Flask backend serves predictions; a React/Vite frontend provides the UI.

## Images
<img width="1512" height="857" alt="image" src="https://github.com/user-attachments/assets/9d42a6aa-5242-41f9-ab88-6c8be1085c31" />

<br />

<img width="979" height="747" alt="image" src="https://github.com/user-attachments/assets/6b0ff9f3-dc4b-42b7-99db-289b5f283c2a" />

<br />

<img width="938" height="779" alt="image" src="https://github.com/user-attachments/assets/e9a9665d-5ee1-4379-9038-fc4164849f89" />





## Setup

```bash
pip install -r requirements.txt

# Configure HuggingFace auth (required to pull CLIP weights)
cp .env.example .env
# then edit .env and set HF_TOKEN=<your-token>

# One-time: prep the OpenRL DeepFakeFace diffusion dataset
python scripts/download_deepfakeface.py

python train.py       # fine-tune the model (also pulls Kaggle datasets via kagglehub)
python evaluate.py    # evaluate on held-out test sets
python app.py         # start the Flask API at http://127.0.0.1:5001

# Frontend (optional)
cd frontend
npm install
npm run dev
```

## How It Works

1. User uploads a face image.
2. YOLOv8-face detects faces and returns padded bounding boxes.
3. Each face crop is passed through the two-branch detector:
   - **Branch 1:** CLIP ViT-L/14 vision encoder (spatial semantics).
   - **Branch 2:** 2D FFT magnitude spectrum → small CNN (frequency artifacts).
   - A fusion MLP head produces the Real / Deepfake logits.
4. Per-face label and softmax confidence are returned to the frontend, along with the bounding box for overlay rendering.

## Training Recipe

- **Backbone:** `openai/clip-vit-large-patch14`
- **Adapter:** DoRA (r=16, α=32) on all CLIP attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`), with LoRA dropout 0.1.
- **Loss:** 0.7 × Supervised Contrastive + 0.3 × Cross-Entropy (label smoothing 0.1). Projection head (128-d, L2-normalized) used only during training.
- **Optimizer:** AdamW (lr=2e-4, weight decay 0.01), gradient clipping at 1.0.
- **Schedule:** CosineAnnealingWarmRestarts (T_0=5, η_min=1e-6).
- **Precision:** AMP (float16) on MPS / CUDA.
- **Augmentation:** Albumentations pipeline — horizontal flip, rotate, random resized crop, color jitter, simulated social-media degradation (downscale → JPEG 20–70 → upscale → blur), JPEG compression, Gaussian blur, downscale, and a stochastic high-pass filter (p=0.15).
- **Batch sampling:** Balanced sampler with 6 groups (3 datasets × 2 classes), 12 samples per group at batch size 72.
- **Validation metric:** ROC-AUC; early stopping with patience 3; per-epoch versioned DoRA snapshots.

## Optimizations

### Training Efficiency

- **DoRA adapter instead of full fine-tuning** — only the decomposed low-rank updates on CLIP's `q/k/v/out_proj` layers are trained; the 300M+ backbone weights stay frozen. Massive VRAM savings, and DoRA typically closes the gap to full fine-tuning that plain LoRA leaves on the table.
- **Automatic Mixed Precision (float16 autocast)** — roughly 2× memory reduction on MPS/CUDA and faster matmuls on tensor-core hardware.
- **Gradient accumulation** (`ACCUM_STEPS`) — lets a small per-step batch simulate a much larger effective batch without the memory cost.
- **Gradient clipping at max_norm=1.0** — stabilizes DoRA + SupCon updates, which can spike early in training.
- **CosineAnnealingWarmRestarts** — resume-friendly LR schedule; periodic restarts help escape flat regions without manual LR tuning.
- **Early stopping (patience=3 on val AUC)** — avoids wasted epochs once the model plateaus.
- **Device auto-selection** — CUDA → MPS → CPU, with `num_workers=4` and `pin_memory=True` enabled automatically on CUDA only (MPS / CPU get `num_workers=0` to avoid Python multiprocessing stalls on macOS).
- **CPU-side Albumentations pipeline** — all augmentation runs in NumPy/OpenCV so it doesn't contend with the MPS GPU mid-step.

### Data Pipeline

- **Balanced batch sampler** — each batch draws equally from 6 groups (3 datasets × 2 classes). Prevents the larger dataset / majority class from dominating gradients and ensures every step sees every generator type (StyleGAN, face-swap, diffusion).
- **Stochastic high-pass filter augmentation** (p=0.15) — forces the network to learn frequency-domain cues even when the FFT branch alone would not be enough.
- **Simulated social-media degradation** — downscale → JPEG 20–70 → upscale → blur, approximating the Instagram/TikTok transcode pipeline so the model generalizes to "in-the-wild" deepfakes rather than pristine dataset images.
- **Three datasets combined** — manjilkarki deepfakes (mixed Kaggle), xhlulu 140k (StyleGAN), and OpenRL DeepFakeFace (Stable Diffusion + InsightFace face-swap). Together they span GAN, face-swap, and diffusion generators.
- **Label smoothing (0.1)** — prevents the CE head from producing overconfident logits, which also improves calibration of the softmax score shown in the UI.

### Loss & Representation

- **SupCon (0.7) + CE (0.3)** — SupCon shapes the fused embedding space so same-class samples cluster together regardless of generator, while CE maintains a clean decision boundary. The projection head is training-only and discarded for inference.
- **Two-branch fusion (CLIP + FFT)** — CLIP handles spatial semantics; the FFT CNN captures spectral peaks and blending artifacts CLIP cannot see in pixel space. Concatenated before the classifier.
- **log1p + fftshift on FFT magnitude** — compresses the dynamic range of the spectrum and centers the DC component, making the distribution easier for a small CNN to learn.

### Checkpointing & Reproducibility

- **Best-only saving, tracked by val AUC** — `head_weights.pt` and the DoRA adapter are overwritten only when validation AUC improves.
- **Versioned per-epoch snapshots** (`dora_epoch{N}_auc{X}`) — any prior epoch can be rolled back to without re-training.
- **`train_state.pt`** — stores optimizer, scheduler, epoch, best-AUC, and patience counter in sync with the best weights, so `python train.py` can safely resume from interruption.
- **DoRA merge at end of training** (`merge_and_unload`) — the final inference checkpoint is a plain `CLIPVisionModel` with the adapter folded in, so `app.py` does not need PEFT at serve time.

### Inference & Serving

- **YOLOv8-face detection with 20% bbox padding** — crops each face before classification so the model sees a consistent face-centered input instead of wide scenes.
- **Multi-face support** — every detected face is classified independently and returned as its own result with bbox + label + confidence.
- **`torch.no_grad()` everywhere in `/predict`** — no autograd graph is built for inference.

### Evaluation

- **AUC-ROC + EER** as primary metrics, plus full classification report and confusion matrix.
- **Three modes:** combined (all datasets pooled), per-dataset, and Leave-One-Out cross-generator (train on two, evaluate on the held-out third) to measure generalization to unseen generator types.
- **Test-Time Augmentation (TTA)** — predictions are averaged over the original and horizontally-flipped image.

## Citations & References

### Pretrained Backbone

- **CLIP (ViT-L/14).** Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. https://arxiv.org/abs/2103.00020 — model weights: https://huggingface.co/openai/clip-vit-large-patch14
- **Vision Transformer (ViT).** Dosovitskiy, A. et al. *An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. https://arxiv.org/abs/2010.11929

### Methods & Techniques

- **UnivFD (motivation for ViT-L/14 as deepfake backbone).** Ojha, U., Li, Y., Lee, Y. J. *Towards Universal Fake Image Detectors that Generalize Across Generative Models.* CVPR 2023. https://arxiv.org/abs/2302.10174
- **DoRA (Weight-Decomposed Low-Rank Adaptation).** Liu, S.-Y. et al. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024. https://arxiv.org/abs/2402.09353
- **LoRA.** Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. https://arxiv.org/abs/2106.09685
- **Supervised Contrastive Learning.** Khosla, P. et al. *Supervised Contrastive Learning.* NeurIPS 2020. https://arxiv.org/abs/2004.11362
- **AdamW (Decoupled Weight Decay).** Loshchilov, I., Hutter, F. *Decoupled Weight Decay Regularization.* ICLR 2019. https://arxiv.org/abs/1711.05101
- **Cosine Annealing with Warm Restarts.** Loshchilov, I., Hutter, F. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017. https://arxiv.org/abs/1608.03983
- **Label Smoothing.** Szegedy, C. et al. *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016. https://arxiv.org/abs/1512.00567 — see also Müller, R., Kornblith, S., Hinton, G. *When Does Label Smoothing Help?* NeurIPS 2019. https://arxiv.org/abs/1906.02629
- **Mixed-Precision Training (AMP).** Micikevicius, P. et al. *Mixed Precision Training.* ICLR 2018. https://arxiv.org/abs/1710.03740

### Face Detection

- **YOLOv8 (Ultralytics).** Jocher, G., Chaurasia, A., Qiu, J. *YOLO by Ultralytics* (v8), 2023. https://github.com/ultralytics/ultralytics
- **YOLO (foundational).** Redmon, J. et al. *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016. https://arxiv.org/abs/1506.02640
- **WIDER FACE benchmark** (used to train the YOLOv8-face weights). Yang, S., Luo, P., Loy, C. C., Tang, X. *WIDER FACE: A Face Detection Benchmark.* CVPR 2016. https://arxiv.org/abs/1511.06523

### Datasets

- **Deepfake and Real Images** — Manjil Karki (Kaggle). https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
- **140k Real and Fake Faces (StyleGAN)** — xhlulu (Kaggle). https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- **StyleGAN (upstream generator of the 140k dataset).** Karras, T., Laine, S., Aila, T. *A Style-Based Generator Architecture for Generative Adversarial Networks.* CVPR 2019. https://arxiv.org/abs/1812.04948
- **Flickr-Faces-HQ (FFHQ, real faces in the 140k dataset).** Karras, T., Laine, S., Aila, T., 2019. https://github.com/NVlabs/ffhq-dataset
- **DeepFakeFace** — OpenRL. https://huggingface.co/datasets/OpenRL/DeepFakeFace — combines real faces (IMDB-WIKI) with diffusion-generated and face-swap fakes.
- **Stable Diffusion v1.5 / inpainting (upstream generator of DeepFakeFace).** Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022. https://arxiv.org/abs/2112.10752
- **IMDB-WIKI (real faces in DeepFakeFace).** Rothe, R., Timofte, R., Van Gool, L. *Deep Expectation of Real and Apparent Age from a Single Image Without Facial Landmarks.* IJCV 2018. https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

### Core Libraries

- **PyTorch.** Paszke, A. et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019. https://pytorch.org
- **HuggingFace Transformers.** Wolf, T. et al. *Transformers: State-of-the-Art Natural Language Processing.* EMNLP 2020 (Demo). https://github.com/huggingface/transformers
- **HuggingFace PEFT** (DoRA / LoRA adapters). https://github.com/huggingface/peft
- **HuggingFace Hub** (`snapshot_download` for the DeepFakeFace dataset). https://github.com/huggingface/huggingface_hub
- **Ultralytics** (YOLOv8 inference framework). https://github.com/ultralytics/ultralytics
- **torchvision** (`ImageFolder`, dataset utilities). https://github.com/pytorch/vision
- **OpenCV.** Bradski, G. *The OpenCV Library.* Dr. Dobb's Journal of Software Tools, 2000. https://opencv.org
- **Albumentations.** Buslaev, A. et al. *Albumentations: Fast and Flexible Image Augmentations.* Information 11(2), 2020. https://github.com/albumentations-team/albumentations
- **scikit-learn** (`roc_auc_score`, `roc_curve`, `classification_report`, `confusion_matrix`). Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* JMLR 12, 2011. https://scikit-learn.org
- **NumPy.** Harris, C. R. et al. *Array Programming with NumPy.* Nature 585, 2020. https://numpy.org
- **Matplotlib.** Hunter, J. D. *Matplotlib: A 2D Graphics Environment.* Computing in Science & Engineering 9(3), 2007. https://matplotlib.org
- **Pillow (PIL fork).** https://python-pillow.org
- **tqdm.** https://github.com/tqdm/tqdm
- **kagglehub.** https://github.com/Kaggle/kagglehub
- **python-dotenv.** https://github.com/theskumar/python-dotenv

### Backend

- **Flask.** https://flask.palletsprojects.com
- **Flask-CORS.** https://github.com/corydolphin/flask-cors

### Frontend

- **React.** https://react.dev
- **Vite.** https://vitejs.dev
- **TypeScript.** https://www.typescriptlang.org
- **Tailwind CSS.** https://tailwindcss.com
- **lucide-react** (icons). https://lucide.dev
- **zustand** (state). https://github.com/pmndrs/zustand
- **clsx** + **tailwind-merge** (className utilities). https://github.com/lukeed/clsx · https://github.com/dcastil/tailwind-merge
- **ESLint.** https://eslint.org

## License

This repository is for research and educational use. Please respect the upstream licenses of the datasets and pretrained weights:

- CLIP ViT-L/14 weights — MIT License (OpenAI).
- Kaggle datasets — see each dataset page for terms of use.
- OpenRL DeepFakeFace — see the HuggingFace dataset page for terms of use.
- YOLOv8 (Ultralytics) — AGPL-3.0; community-trained face weights inherit any restrictions of WIDER FACE and the upstream training pipeline.
