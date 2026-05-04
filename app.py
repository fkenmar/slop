import os
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from transformers import CLIPVisionModel, CLIPImageProcessor
from model import DeepfakeDetector, CLIP_MODEL_ID
from PIL import Image
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS

from dotenv import load_dotenv; load_dotenv()

app = Flask(__name__)
CORS(app, origins="*")

FRONTEND_DIST = Path("frontend/dist")
YOLO_FACE_CONF = 0.35
MODEL_DIR = Path("./model")
MODEL_REPO = "knmrfr/deepfake-detector"

# ── Model & YOLO loading ───────────────────────────────────────────────────────
if (MODEL_DIR / "clip_vision").exists():
    print(f"Loading trained model from: {MODEL_DIR}")
    yolo_path = str(MODEL_DIR / "yolov8n-face.pt")
    model = DeepfakeDetector.from_pretrained(MODEL_DIR)
    processor = CLIPImageProcessor.from_pretrained(MODEL_DIR)
else:
    print(f"Downloading model from HF Hub: {MODEL_REPO}")
    from huggingface_hub import hf_hub_download
    yolo_path = hf_hub_download(repo_id=MODEL_REPO, filename="model/yolov8n-face.pt")
    model = DeepfakeDetector.from_pretrained(MODEL_REPO)
    processor = CLIPImageProcessor.from_pretrained(MODEL_REPO, subfolder="model")

yolo_face = YOLO(yolo_path)
model.eval()

# ── Face Detection Helper ────────────────────────────────────────────────────
def detect_faces(bgr_img):
    """Detect faces with YOLOv8-face and return padded face boxes."""
    h, w = bgr_img.shape[:2]

    results = yolo_face.predict(bgr_img, conf=YOLO_FACE_CONF, verbose=False)
    boxes = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for xyxy in results[0].boxes.xyxy.cpu().numpy():
            boxes.append(tuple(xyxy.tolist()))
    print(f"[detect_faces] YOLOv8 returned {len(boxes)} face(s)")
    if not boxes:
        return []

    faces = []
    for bx1, by1, bx2, by2 in boxes:
        bw = bx2 - bx1
        bh = by2 - by1
        pad_x = bw * 0.2
        pad_y = bh * 0.2
        x1 = int(max(0, bx1 - pad_x))
        y1 = int(max(0, by1 - pad_y))
        x2 = int(min(w, bx2 + pad_x))
        y2 = int(min(h, by2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            continue

        faces.append((x1, y1, x2, y2))

    return faces


def crop_face(bgr_img, bbox):
    """Crop face region from image using bbox."""
    x1, y1, x2, y2 = bbox
    return bgr_img[y1:y2, x1:x2]

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
	      document.getElementById('resultsContainer').innerHTML = '';
	    }

	    function clearImage() {
	      selectedFile = null;
	      thumbData = null;
	      dropZone.innerHTML = '<div class="icon">+</div><span>Click or drag & drop an image</span>';
	      document.getElementById('analyzeBtn').disabled = true;
	      document.getElementById('clearBtn').disabled = true;
	      document.getElementById('resultsContainer').innerHTML = '';
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
	          container.innerHTML =
	            '<div class="face-result deepfake">' +
	              '<div class="verdict">NO FACE DETECTED</div>' +
	              '<div class="conf-text">Upload a clear face photo so the model can analyze a face crop</div>' +
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
	            '<div class="conf-text">' + face.confidence + '% model confidence</div>';
	          container.appendChild(div);

          // Animate confidence bar
          setTimeout(() => {
            div.querySelector('.conf-bar').style.width = face.confidence + '%';
          }, 50);

	          addHistory(selectedFile.name + (faceLabel ? ' (' + faceLabel + ')' : ''), cls, face.confidence, thumbData);
	        });
      } catch (err) {
        alert('Analysis failed: ' + err.message);
      }

      btn.disabled = false;
      btn.textContent = 'Analyze';
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
    if FRONTEND_DIST.exists():
        return send_from_directory(str(FRONTEND_DIST), "index.html")
    return render_template_string(HTML)

@app.route("/<path:path>")
def static_proxy(path):
    if FRONTEND_DIST.exists():
        try:
            return send_from_directory(str(FRONTEND_DIST), path)
        except Exception:
            return send_from_directory(str(FRONTEND_DIST), "index.html")
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
        return jsonify({
            "faces": [],
            "face_count": 0,
            "face_detected": False,
        })

    # Process each detected face
    results = []
    for face_bbox in faces:
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

        results.append({
            "label": label,
            "confidence": confidence,
            "bbox": list(face_bbox),
        })

    return jsonify({
        "faces": results,
        "face_count": len(results),
        "face_detected": True,
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
