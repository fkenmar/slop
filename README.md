# Deepfake Detector

An image-based deepfake detection web app that classifies face images as **Real** or **Deepfake** using a two-branch fusion model: a DoRA-fine-tuned CLIP ViT-L/14 backbone (spatial features) combined with a lightweight CNN over the 2D FFT magnitude spectrum (frequency artifacts). A Flask backend serves predictions, attention-rollout heatmaps, and uncanny-valley heuristics; a React/Vite frontend provides the UI.

## Setup

```bash
pip install -r requirements.txt
python train.py       # fine-tune the model (downloads datasets via kagglehub)
python evaluate.py    # evaluate on held-out test sets
python app.py         # start the Flask API at http://127.0.0.1:5001

# Frontend (optional)
cd frontend
npm install
npm run dev
```

## How It Works

1. User uploads a face image.
2. MediaPipe Face Landmarker detects and crops faces.
3. Each face is passed through the two-branch detector:
   - **Branch 1:** CLIP ViT-L/14 vision encoder (spatial semantics).
   - **Branch 2:** 2D FFT magnitude spectrum → small CNN (frequency artifacts).
   - A fusion MLP head produces the Real / Deepfake logits.
4. Attention rollout over the ViT layers produces a heatmap.
5. OpenCV-based heuristics compute an "uncanny valley" score (symmetry, eye consistency, skin tex`ture, edge naturalness, lighting consistency, sensor-noise pattern).
6. Results are returned to the frontend for display.

## Training Recipe

- **Backbone:** `openai/clip-vit-large-patch14`
- **Adapter:** DoRA (r=16, α=32) on all CLIP attention projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`), with LoRA dropout 0.1.
- **Loss:** 0.7 × Supervised Contrastive + 0.3 × Cross-Entropy (label smoothing 0.1). Projection head (128-d, L2-normalized) used only during training.
- **Optimizer:** AdamW (lr=2e-4, weight decay 0.01), gradient clipping at 1.0.
- **Schedule:** CosineAnnealingWarmRestarts (T_0=5, η_min=1e-6).
- **Precision:** AMP (float16) on MPS / CUDA.
- **Augmentation:** Albumentations pipeline — horizontal flip, rotate, random resized crop, color jitter, simulated social-media degradation (downscale → JPEG 20–70 → upscale → blur), JPEG compression, Gaussian blur, downscale, and a stochastic high-pass filter (p=0.15).
- **Batch sampling:** Balanced sampler with 4 groups (2 datasets × 2 classes), equal representation per batch.
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

- **Balanced batch sampler** — each batch draws equally from 4 groups (2 datasets × 2 classes). Prevents the larger dataset / majority class from dominating gradients and ensures every step sees every generator type.
- **Stochastic high-pass filter augmentation** (p=0.15) — forces the network to learn frequency-domain cues even when the FFT branch alone would not be enough.
- **Simulated social-media degradation** — downscale → JPEG 20–70 → upscale → blur, approximating the Instagram/TikTok transcode pipeline so the model generalizes to "in-the-wild" deepfakes rather than pristine dataset images.
- **Two datasets combined** (manjilkarki deepfakes + xhlulu 140k StyleGAN) — broader generator coverage (diffusion, face-swap, StyleGAN) than any single source.
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

- **MediaPipe Face Landmarker with 20% bbox padding** — crops each face before classification so the model sees a consistent face-centered input instead of wide scenes.
- **Multi-face support** — every detected face is classified independently and returned as its own result.
- **Attention rollout heatmap via class-swap patch** — CLIP's `CLIPSdpaAttention` ignores `output_attentions=True`; the code temporarily swaps each layer's class back to the eager `CLIPAttention` parent so attention matrices can be read without monkey-patching `forward`. Rollout adds residual and re-normalizes per layer.
- **`torch.no_grad()` everywhere in `/predict`** — no autograd graph is built for inference.
- **Base64 PNG encoding of heatmaps** — avoids a second HTTP round-trip; the frontend can render the overlay immediately.

## Citations

### Pretrained Backbone

- **CLIP (ViT-L/14).** Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. https://arxiv.org/abs/2103.00020 — model weights: https://huggingface.co/openai/clip-vit-large-patch14
- **Vision Transformer (ViT).** Dosovitskiy, A. et al. *An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. https://arxiv.org/abs/2010.11929

### Methods & Techniques

- **UnivFD (reference for ViT-L/14 choice as deepfake backbone).** Ojha, U., Li, Y., Lee, Y. J. *Towards Universal Fake Image Detectors that Generalize Across Generative Models.* CVPR 2023. https://arxiv.org/abs/2302.10174
- **DoRA (Weight-Decomposed Low-Rank Adaptation).** Liu, S.-Y. et al. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024. https://arxiv.org/abs/2402.09353
- **LoRA.** Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. https://arxiv.org/abs/2106.09685
- **Supervised Contrastive Learning.** Khosla, P. et al. *Supervised Contrastive Learning.* NeurIPS 2020. https://arxiv.org/abs/2004.11362
- **Attention Rollout (for the ViT attention heatmap).** Abnar, S. and Zuidema, W. *Quantifying Attention Flow in Transformers.* ACL 2020. https://arxiv.org/abs/2005.00928
- **AdamW.** Loshchilov, I. and Hutter, F. *Decoupled Weight Decay Regularization.* ICLR 2019. https://arxiv.org/abs/1711.05101
- **Cosine Annealing with Warm Restarts.** Loshchilov, I. and Hutter, F. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017. https://arxiv.org/abs/1608.03983
- **Label Smoothing.** Szegedy, C. et al. *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016. https://arxiv.org/abs/1512.00567 — see also Müller, R., Kornblith, S., Hinton, G. *When Does Label Smoothing Help?* NeurIPS 2019. https://arxiv.org/abs/1906.02629
- **Mixed-precision training (AMP).** Micikevicius, P. et al. *Mixed Precision Training.* ICLR 2018. https://arxiv.org/abs/1710.03740

### Datasets

- **Deepfake and Real Images** — Manjil Karki (Kaggle). https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
- **140k Real and Fake Faces (StyleGAN)** — xhlulu (Kaggle). https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- **StyleGAN (upstream generator of 140k dataset).** Karras, T., Laine, S., Aila, T. *A Style-Based Generator Architecture for Generative Adversarial Networks.* CVPR 2019. https://arxiv.org/abs/1812.04948
- **Flickr-Faces-HQ (FFHQ, real faces in 140k dataset).** Karras, T. et al., 2019. https://github.com/NVlabs/ffhq-dataset

### Core Libraries

- **PyTorch.** Paszke, A. et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019. https://pytorch.org
- **HuggingFace Transformers.** Wolf, T. et al. *Transformers: State-of-the-Art Natural Language Processing.* EMNLP 2020 (Demo). https://github.com/huggingface/transformers
- **HuggingFace PEFT** (DoRA / LoRA adapters). https://github.com/huggingface/peft
- **torchvision** (`ImageFolder`, dataset utilities). https://github.com/pytorch/vision
- **MediaPipe Face Landmarker.** Google, 2019–. https://developers.google.com/mediapipe/solutions/vision/face_landmarker — Lugaresi, C. et al. *MediaPipe: A Framework for Building Perception Pipelines.* 2019. https://arxiv.org/abs/1906.08172
- **OpenCV.** Bradski, G. *The OpenCV Library.* Dr. Dobb's Journal of Software Tools, 2000. https://opencv.org
- **Albumentations.** Buslaev, A. et al. *Albumentations: Fast and Flexible Image Augmentations.* Information 11(2), 2020. https://github.com/albumentations-team/albumentations
- **scikit-learn** (`roc_auc_score`). Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python.* JMLR 12, 2011. https://scikit-learn.org
- **NumPy.** Harris, C. R. et al. *Array Programming with NumPy.* Nature 585, 2020. https://numpy.org
- **Matplotlib.** Hunter, J. D. *Matplotlib: A 2D Graphics Environment.* Computing in Science & Engineering 9(3), 2007. https://matplotlib.org
- **Pillow (PIL fork).** https://python-pillow.org
- **tqdm.** https://github.com/tqdm/tqdm
- **kagglehub.** https://github.com/Kaggle/kagglehub

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
- **exifr** (EXIF parsing). https://github.com/MikeKovarik/exifr
- **clsx** + **tailwind-merge** (className utilities). https://github.com/lukeed/clsx · https://github.com/dcastil/tailwind-merge
- **ESLint.** https://eslint.org

## License

This repository is for research and educational use. Please respect the upstream licenses of the datasets and pretrained weights:

- CLIP ViT-L/14 weights — MIT License (OpenAI).
- Kaggle datasets — see each dataset page for terms of use.
