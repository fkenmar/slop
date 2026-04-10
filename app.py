import os
import cv2
import numpy as np
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID = "./model" if os.path.exists("./model") else "prithivMLmods/Deep-Fake-Detector-v2-Model"
print(f"Loading model from: {MODEL_ID}")
processor = ViTImageProcessor.from_pretrained(MODEL_ID)
model = ViTForImageClassification.from_pretrained(MODEL_ID)
model.eval()

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
      background: #0f0f0f;
      color: #f0f0f0;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 40px 20px;
    }
    h1 { font-size: 2rem; margin-bottom: 8px; letter-spacing: 1px; }
    p.sub { color: #888; margin-bottom: 32px; }
    .card {
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 16px;
      padding: 32px;
      width: 100%;
      max-width: 480px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 20px;
    }
    .drop-zone {
      width: 100%;
      height: 200px;
      border: 2px dashed #444;
      border-radius: 12px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: border-color 0.2s;
      overflow: hidden;
    }
    .drop-zone:hover { border-color: #888; }
    .drop-zone img { max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 10px; }
    .drop-zone span { color: #666; font-size: 0.9rem; }
    input[type="file"] { display: none; }
    button {
      width: 100%;
      padding: 14px;
      background: #fff;
      color: #000;
      border: none;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    button:hover { opacity: 0.85; }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    .result {
      width: 100%;
      padding: 16px;
      border-radius: 10px;
      text-align: center;
      font-size: 1.1rem;
      font-weight: 600;
      display: none;
    }
    .result.realism { background: #0d3321; color: #4caf82; border: 1px solid #1a5c3a; }
    .result.deepfake { background: #3a0d0d; color: #e05c5c; border: 1px solid #5c1a1a; }
    .confidence { font-size: 0.85rem; font-weight: 400; margin-top: 4px; opacity: 0.8; }
  </style>
</head>
<body>
  <h1>Deepfake Detector</h1>
  <p class="sub">Upload a face image to check if it's real or AI-generated</p>
  <div class="card">
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <span>Click or drag &amp; drop an image</span>
    </div>
    <input type="file" id="fileInput" accept="image/*" onchange="previewFile(event)" />
    <button id="analyzeBtn" onclick="analyze()" disabled>Analyze</button>
    <div class="result" id="result">
      <div id="verdict"></div>
      <div class="confidence" id="confidence"></div>
    </div>
  </div>

  <script>
    let selectedFile = null;

    const dropZone = document.getElementById('dropZone');
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = '#fff'; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = '#444'; });
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.style.borderColor = '#444';
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) loadFile(file);
    });

    function previewFile(e) { loadFile(e.target.files[0]); }

    function loadFile(file) {
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = ev => {
        dropZone.innerHTML = '<img src="' + ev.target.result + '" />';
      };
      reader.readAsDataURL(file);
      document.getElementById('analyzeBtn').disabled = false;
      document.getElementById('result').style.display = 'none';
    }

    async function analyze() {
      if (!selectedFile) return;
      const btn = document.getElementById('analyzeBtn');
      btn.disabled = true;
      btn.textContent = 'Analyzing...';

      const form = new FormData();
      form.append('image', selectedFile);

      const res = await fetch('/predict', { method: 'POST', body: form });
      const data = await res.json();

      const resultEl = document.getElementById('result');
      const cls = data.label.toLowerCase();
      resultEl.className = 'result ' + cls;
      document.getElementById('verdict').textContent = cls === 'realism' ? 'REAL IMAGE' : 'DEEPFAKE DETECTED';
      document.getElementById('confidence').textContent = 'Confidence: ' + data.confidence + '%';
      resultEl.style.display = 'block';

      btn.disabled = false;
      btn.textContent = 'Analyze';
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
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        predicted_idx = int(torch.argmax(probs).item())

    label = model.config.id2label[predicted_idx]
    confidence = round(float(probs[predicted_idx].item()) * 100, 1)

    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
