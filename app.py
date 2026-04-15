import os
from pathlib import Path
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from transformers import CLIPVisionModel, CLIPImageProcessor
from model import DeepfakeDetector, CLIP_MODEL_ID
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./model/face_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=10
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)


# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("./model")
if (MODEL_DIR / "clip_vision").exists():
    print(f"Loading trained model from: {MODEL_DIR}")
    model = DeepfakeDetector.from_pretrained(MODEL_DIR)
    processor = CLIPImageProcessor.from_pretrained(MODEL_DIR)
else:
    print(f"No trained model found — loading base CLIP from {CLIP_MODEL_ID}")
    clip_vision = CLIPVisionModel.from_pretrained(CLIP_MODEL_ID)
    model = DeepfakeDetector(clip_vision)
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
model.eval()

# ── Face Detection Helper ────────────────────────────────────────────────────
def detect_faces(bgr_img):
    """Detect all faces and return list of (landmarks, face_bbox).
    face_bbox is (x1, y1, x2, y2) in pixel coords with padding.
    Returns empty list if no faces found."""
    h, w = bgr_img.shape[:2]
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = face_landmarker.detect(mp_image)

    if not detection_result.face_landmarks:
        return []

    faces = []
    for landmarks in detection_result.face_landmarks:
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        fw, fh = x2 - x1, y2 - y1
        pad_x, pad_y = fw * 0.2, fh * 0.2
        x1 = int(max(0, x1 - pad_x))
        y1 = int(max(0, y1 - pad_y))
        x2 = int(min(w, x2 + pad_x))
        y2 = int(min(h, y2 + pad_y))
        faces.append((landmarks, (x1, y1, x2, y2)))

    return faces


def crop_face(bgr_img, bbox):
    """Crop face region from image using bbox."""
    x1, y1, x2, y2 = bbox
    return bgr_img[y1:y2, x1:x2]


# ── Uncanny Valley Analysis ──────────────────────────────────────────────────
def analyze_uncanny(bgr_img, landmarks, face_bbox):
    """Run uncanny valley heuristics on a BGR image using detected face region."""
    results = {}
    h, w = bgr_img.shape[:2]
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    if landmarks is None or face_bbox is None:
        return {"symmetry": None, "eye_consistency": None, "texture": None,
                "edge_natural": None, "color_consistency": None}

    lm = landmarks
    fx1, fy1, fx2, fy2 = face_bbox
    face_gray = gray[fy1:fy2, fx1:fx2]
    face_bgr = bgr_img[fy1:fy2, fx1:fx2]

    # --- 1. Facial Symmetry ---
    pairs = [(33, 263), (133, 362), (70, 300), (105, 334), (107, 336)]
    nose_x = lm[1].x
    diffs = []
    for li, ri in pairs:
        left_dist = abs(lm[li].x - nose_x)
        right_dist = abs(lm[ri].x - nose_x)
        if max(left_dist, right_dist) > 0:
            diffs.append(abs(left_dist - right_dist) / max(left_dist, right_dist))
    symmetry = 1.0 - (sum(diffs) / len(diffs)) if diffs else 1.0
    results["symmetry"] = round(symmetry * 100, 1)

    # --- 2. Eye Reflection Consistency ---
    def eye_region(indices):
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = max(0, min(xs)), min(w, max(xs))
        y1, y2 = max(0, min(ys)), min(h, max(ys))
        if x2 <= x1 or y2 <= y1:
            return None
        return gray[y1:y2, x1:x2]

    left_eye_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133]
    right_eye_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263]
    le = eye_region(left_eye_idx)
    re = eye_region(right_eye_idx)

    if le is not None and re is not None and le.size > 0 and re.size > 0:
        le_resized = cv2.resize(le, (32, 16))
        re_resized = cv2.resize(re, (32, 16))
        h_left = cv2.calcHist([le_resized], [0], None, [32], [0, 256])
        h_right = cv2.calcHist([re_resized], [0], None, [32], [0, 256])
        cv2.normalize(h_left, h_left)
        cv2.normalize(h_right, h_right)
        eye_corr = cv2.compareHist(h_left, h_right, cv2.HISTCMP_CORREL)
        results["eye_consistency"] = round(max(0, eye_corr) * 100, 1)
    else:
        results["eye_consistency"] = None

    # --- 3. Skin Texture (FFT on face region only) ---
    face_gray_f = np.float32(face_gray)
    dft = cv2.dft(face_gray_f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude = np.log(magnitude + 1)

    fh, fw = face_gray.shape[:2]
    cy, cx = fh // 2, fw // 2
    radius = min(cy, cx) // 3
    total_energy = magnitude.sum()
    mask = np.ones_like(magnitude)
    cv2.circle(mask, (cx, cy), radius, 0, -1)
    high_freq_energy = (magnitude * mask).sum()
    texture_score = high_freq_energy / total_energy if total_energy > 0 else 0
    results["texture"] = round(min(texture_score * 130, 100), 1)

    # --- 4. Edge Artifacts (Laplacian on face boundary region) ---
    # Create a ring mask around the face edge to check blending artifacts
    face_h, face_w = face_gray.shape[:2]
    ring_mask = np.zeros_like(face_gray)
    center = (face_w // 2, face_h // 2)
    outer_r = min(face_w, face_h) // 2
    inner_r = int(outer_r * 0.75)
    cv2.circle(ring_mask, center, outer_r, 255, -1)
    cv2.circle(ring_mask, center, inner_r, 0, -1)

    laplacian = cv2.Laplacian(face_gray, cv2.CV_64F)
    edge_pixels = laplacian[ring_mask > 0]
    lap_var = edge_pixels.var() if edge_pixels.size > 0 else 0
    edge_score = min(lap_var / 20, 100)
    results["edge_natural"] = round(edge_score, 1)

    # --- 5. Lighting Consistency (face halves only) ---
    face_mid = face_w // 2
    left_half = face_bgr[:, :face_mid]
    right_half = face_bgr[:, face_mid:]
    left_mean = np.mean(left_half, axis=(0, 1))
    right_mean = np.mean(right_half, axis=(0, 1))
    color_diff = np.abs(left_mean - right_mean)
    color_consistency = max(0, 100 - np.mean(color_diff) * 2)
    results["color_consistency"] = round(color_consistency, 1)

    # --- 6. Noise Pattern Analysis ---
    # Real photos have natural sensor noise; AI-generated images have uniform/no noise.
    # Extract noise by subtracting a blurred version from the original face.
    denoised = cv2.GaussianBlur(face_gray, (5, 5), 0)
    noise = face_gray.astype(np.float64) - denoised.astype(np.float64)

    # Real noise has higher variance and non-uniform distribution across the face
    noise_std = noise.std()

    # Check noise uniformity — split face into 4 quadrants and compare noise levels
    mid_y, mid_x = face_gray.shape[0] // 2, face_gray.shape[1] // 2
    quadrants = [
        noise[:mid_y, :mid_x], noise[:mid_y, mid_x:],
        noise[mid_y:, :mid_x], noise[mid_y:, mid_x:],
    ]
    quad_stds = [q.std() for q in quadrants if q.size > 0]
    # Low variation between quadrants = suspiciously uniform (AI-generated)
    # High variation = natural sensor noise affected by lighting/skin
    noise_variation = np.std(quad_stds) if len(quad_stds) > 1 else 0

    # Combine: real images score high on both noise presence and non-uniformity
    # noise_std typical range: 2-15 for real, 0-3 for AI
    presence_score = min(noise_std / 10 * 100, 100)
    uniformity_score = min(noise_variation / 2 * 100, 100)
    noise_score = presence_score * 0.6 + uniformity_score * 0.4
    results["noise_natural"] = round(min(noise_score, 100), 1)

    return results

# ── HTML UI ───────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Deepfake Detector</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0a0a0a;
      color: #f0f0f0;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 40px 20px;
    }
    h1 { font-size: 2.2rem; margin-bottom: 4px; letter-spacing: 1px; }
    p.sub { color: #666; margin-bottom: 32px; font-size: 0.95rem; }
    .card {
      background: #141414;
      border: 1px solid #222;
      border-radius: 20px;
      padding: 32px;
      width: 100%;
      max-width: 500px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 20px;
    }
    .drop-zone {
      width: 100%;
      height: 220px;
      border: 2px dashed #333;
      border-radius: 14px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.25s;
      overflow: hidden;
      position: relative;
    }
    .drop-zone:hover { border-color: #555; background: #1a1a1a; }
    .drop-zone.drag-over { border-color: #fff; background: #1a1a1a; }
    .drop-zone img { max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 12px; }
    .drop-zone .icon { font-size: 2rem; margin-bottom: 8px; opacity: 0.3; }
    .drop-zone span { color: #555; font-size: 0.85rem; }
    input[type="file"] { display: none; }

    .btn-row { display: flex; gap: 10px; width: 100%; }
    button {
      flex: 1;
      padding: 14px;
      background: #fff;
      color: #000;
      border: none;
      border-radius: 10px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    button:hover { opacity: 0.85; transform: translateY(-1px); }
    button:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }
    .btn-secondary { background: #222; color: #aaa; }
    .btn-secondary:hover { background: #2a2a2a; opacity: 1; }

    .result {
      width: 100%;
      padding: 20px;
      border-radius: 14px;
      text-align: center;
      display: none;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .result.realism { background: #0a2618; border: 1px solid #1a4a32; }
    .result.deepfake { background: #2a0a0a; border: 1px solid #4a1a1a; }
    .verdict { font-size: 1.2rem; font-weight: 700; margin-bottom: 12px; }
    .result.realism .verdict { color: #4caf82; }
    .result.deepfake .verdict { color: #e05c5c; }

    .conf-bar-wrap {
      width: 100%;
      height: 8px;
      background: #1a1a1a;
      border-radius: 4px;
      overflow: hidden;
      margin-bottom: 8px;
    }
    .conf-bar {
      height: 100%;
      border-radius: 4px;
      transition: width 0.5s ease;
    }
    .result.realism .conf-bar { background: linear-gradient(90deg, #2d7a53, #4caf82); }
    .result.deepfake .conf-bar { background: linear-gradient(90deg, #8a2c2c, #e05c5c); }
    .conf-text { font-size: 0.8rem; color: #888; }

    .analysis { width: 100%; margin-top: 4px; display: none; }
    .analysis h4 { font-size: 0.75rem; color: #555; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; }
    .metric {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }
    .metric-label {
      font-size: 0.8rem;
      color: #999;
      width: 130px;
      flex-shrink: 0;
    }
    .metric-bar-wrap {
      flex: 1;
      height: 6px;
      background: #1a1a1a;
      border-radius: 3px;
      overflow: hidden;
    }
    .metric-bar {
      height: 100%;
      border-radius: 3px;
      transition: width 0.6s ease;
    }
    .metric-bar.good { background: linear-gradient(90deg, #1a5c3a, #4caf82); }
    .metric-bar.warn { background: linear-gradient(90deg, #5c4a1a, #cfaa3e); }
    .metric-bar.bad  { background: linear-gradient(90deg, #5c1a1a, #e05c5c); }
    .metric-val {
      font-size: 0.75rem;
      color: #666;
      width: 40px;
      text-align: right;
      flex-shrink: 0;
    }
    .metric-desc {
      font-size: 0.7rem;
      color: #444;
      margin: -6px 0 10px 140px;
    }

    .history { width: 100%; max-width: 500px; margin-top: 24px; }
    .history h3 { font-size: 0.85rem; color: #444; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px; }
    .history-list { display: flex; flex-direction: column; gap: 8px; }
    .history-item {
      display: flex;
      align-items: center;
      gap: 12px;
      background: #141414;
      border: 1px solid #222;
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 0.85rem;
      animation: fadeIn 0.3s ease;
    }
    .history-item img { width: 36px; height: 36px; border-radius: 6px; object-fit: cover; }
    .history-item .name { flex: 1; color: #aaa; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .history-item .badge {
      padding: 3px 10px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .badge.realism { background: #0a2618; color: #4caf82; }
    .badge.deepfake { background: #2a0a0a; color: #e05c5c; }

    .face-results { display: flex; flex-direction: column; gap: 12px; width: 100%; }
    .face-result {
      padding: 16px;
      border-radius: 14px;
      animation: fadeIn 0.3s ease;
    }
    .face-result.realism { background: #0a2618; border: 1px solid #1a4a32; }
    .face-result.deepfake { background: #2a0a0a; border: 1px solid #4a1a1a; }
    .face-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .face-label { font-size: 0.75rem; color: #888; background: #1a1a1a; padding: 2px 8px; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>Deepfake Detector</h1>
  <p class="sub">Upload an image to check if faces are real or AI-generated</p>
  <div class="card">
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <div class="icon">+</div>
      <span>Click or drag & drop an image</span>
    </div>
    <input type="file" id="fileInput" accept="image/*" onchange="previewFile(event)" />
    <div class="btn-row">
      <button class="btn-secondary" id="clearBtn" onclick="clearImage()" disabled>Clear</button>
      <button id="analyzeBtn" onclick="analyze()" disabled>Analyze</button>
    </div>
    <div id="resultsContainer"></div>
  </div>

  <div class="history" id="historySection" style="display:none">
    <h3>History</h3>
    <div class="history-list" id="historyList"></div>
  </div>

  <script>
    let selectedFile = null;
    let thumbData = null;

    const dropZone = document.getElementById('dropZone');

    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) loadFile(file);
    });

    function previewFile(e) { loadFile(e.target.files[0]); }

    function loadFile(file) {
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = ev => {
        thumbData = ev.target.result;
        dropZone.innerHTML = '<img src="' + thumbData + '" />';
      };
      reader.readAsDataURL(file);
      document.getElementById('analyzeBtn').disabled = false;
      document.getElementById('clearBtn').disabled = false;
      document.getElementById('result').style.display = 'none';
    }

    function clearImage() {
      selectedFile = null;
      thumbData = null;
      dropZone.innerHTML = '<div class="icon">+</div><span>Click or drag & drop an image</span>';
      document.getElementById('analyzeBtn').disabled = true;
      document.getElementById('clearBtn').disabled = true;
      document.getElementById('result').style.display = 'none';
      document.getElementById('fileInput').value = '';
    }

    async function analyze() {
      if (!selectedFile) return;
      const btn = document.getElementById('analyzeBtn');
      btn.disabled = true;
      btn.textContent = 'Analyzing...';

      const form = new FormData();
      form.append('image', selectedFile);

      try {
        const res = await fetch('/predict', { method: 'POST', body: form });
        const data = await res.json();
        const container = document.getElementById('resultsContainer');
        container.innerHTML = '';

        if (!data.face_detected) {
          const face = data.faces[0];
          const cls = face.label.toLowerCase();
          container.innerHTML =
            '<div class="face-result deepfake">' +
              '<div class="verdict">NO FACE DETECTED</div>' +
              '<div class="conf-text">Upload a clear, front-facing photo for accurate results</div>' +
            '</div>';
          btn.disabled = false;
          btn.textContent = 'Analyze';
          return;
        }

        const faceCount = data.faces.length;
        data.faces.forEach((face, i) => {
          const cls = face.label.toLowerCase();
          const verdictText = cls === 'realism' ? 'REAL' : 'DEEPFAKE';
          const faceLabel = faceCount > 1 ? 'Face ' + (i + 1) : '';

          const div = document.createElement('div');
          div.className = 'face-result ' + cls;
          div.innerHTML =
            '<div class="face-header">' +
              (faceLabel ? '<span class="face-label">' + faceLabel + '</span>' : '') +
              '<div class="verdict">' + verdictText + '</div>' +
            '</div>' +
            '<div class="conf-bar-wrap"><div class="conf-bar" style="width:0%"></div></div>' +
            '<div class="conf-text">' + face.confidence + '% confidence</div>' +
            '<div class="analysis" style="display:block; margin-top:8px;">' +
              '<h4>Uncanny Valley Analysis</h4>' +
              '<div class="metrics-container"></div>' +
            '</div>';
          container.appendChild(div);

          // Animate confidence bar
          setTimeout(() => {
            div.querySelector('.conf-bar').style.width = face.confidence + '%';
          }, 50);

          // Render uncanny metrics
          renderMetricsInto(div.querySelector('.metrics-container'), face.uncanny);

          addHistory(selectedFile.name + (faceLabel ? ' (' + faceLabel + ')' : ''), cls, face.confidence, thumbData);
        });
      } catch (err) {
        alert('Analysis failed: ' + err.message);
      }

      btn.disabled = false;
      btn.textContent = 'Analyze';
    }

    const metricInfo = {
      symmetry:          { label: 'Facial Symmetry',     desc: 'How balanced left vs right face features are' },
      eye_consistency:   { label: 'Eye Reflections',     desc: 'Whether both eyes reflect light consistently' },
      texture:           { label: 'Skin Texture',        desc: 'Presence of natural micro-texture (FFT analysis)' },
      edge_natural:      { label: 'Edge Naturalness',    desc: 'Quality of edges around facial boundaries' },
      color_consistency: { label: 'Lighting Consistency', desc: 'Whether lighting is uniform across the face' },
      noise_natural:     { label: 'Noise Pattern',        desc: 'Presence of natural camera sensor noise (AI images lack this)' },
    };

    function renderMetricsInto(container, uncanny) {
      container.innerHTML = '';
      if (!uncanny) return;

      for (const [key, info] of Object.entries(metricInfo)) {
        const val = uncanny[key];
        if (val === null || val === undefined) continue;

        const grade = val >= 70 ? 'good' : val >= 40 ? 'warn' : 'bad';
        const row = document.createElement('div');
        row.innerHTML =
          '<div class="metric">' +
            '<span class="metric-label">' + info.label + '</span>' +
            '<div class="metric-bar-wrap"><div class="metric-bar ' + grade + '" style="width:0%"></div></div>' +
            '<span class="metric-val">' + val + '%</span>' +
          '</div>' +
          '<div class="metric-desc">' + info.desc + '</div>';
        container.appendChild(row);

        setTimeout(() => {
          row.querySelector('.metric-bar').style.width = val + '%';
        }, 100);
      }
    }

    function addHistory(name, cls, conf, thumb) {
      const section = document.getElementById('historySection');
      section.style.display = 'block';
      const list = document.getElementById('historyList');
      const item = document.createElement('div');
      item.className = 'history-item';
      item.innerHTML =
        '<img src="' + thumb + '" />' +
        '<span class="name">' + name + '</span>' +
        '<span class="badge ' + cls + '">' +
          (cls === 'realism' ? 'Real' : 'Fake') + ' ' + conf + '%</span>';
      list.prepend(item);
    }
  </script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    arr = np.frombuffer(file.read(), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    faces = detect_faces(bgr)

    if not faces:
        # No face detected — fall back to full image as single result
        face_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face_rgb)
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            predicted_idx = int(torch.argmax(probs).item())

        return jsonify({
            "faces": [{
                "label": model.id2label[predicted_idx],
                "confidence": round(float(probs[predicted_idx].item()) * 100, 1),
                "uncanny": analyze_uncanny(bgr, None, None),
                "bbox": None,
            }],
            "face_count": 0,
            "face_detected": False,
        })

    # Process each detected face
    results = []
    for landmarks, face_bbox in faces:
        face_bgr = crop_face(bgr, face_bbox)
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face_rgb)
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            predicted_idx = int(torch.argmax(probs).item())

        label = model.id2label[predicted_idx]
        confidence = round(float(probs[predicted_idx].item()) * 100, 1)
        uncanny = analyze_uncanny(bgr, landmarks, face_bbox)

        results.append({
            "label": label,
            "confidence": confidence,
            "uncanny": uncanny,
            "bbox": list(face_bbox),
        })

    return jsonify({
        "faces": results,
        "face_count": len(results),
        "face_detected": True,
    })

if __name__ == "__main__":
    app.run(debug=True)
