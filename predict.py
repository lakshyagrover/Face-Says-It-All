import os, sys, io, base64, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("[startup] Loading TensorFlow...", flush=True)
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, make_response
import numpy as np
import cv2
from PIL import Image
import requests

MODEL_PATH = "emotion_model.h5"
MODEL_URL  = "https://huggingface.co/lakshyagrover/emotion_model/resolve/main/emotion_model.h5"

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
INPUT_SIZE   = 224

EMOTION_MAP = {
    0: "angry",
    1: "neutral",
    2: "fearful",
    3: "happy",
    4: "sad",
    5: "surprised",
    6: "neutral"
}

# Handle Git LFS pointer files (small placeholder files on Render/Heroku)
if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 2000:
    print(f"[startup] {MODEL_PATH} appears to be a LFS pointer -- deleting to re-download...", flush=True)
    os.remove(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print("[startup] Downloading model from Hugging Face...", flush=True)
    r = requests.get(MODEL_URL, stream=True, timeout=300)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("[startup] Download complete!", flush=True)

print("[startup] Loading model...", flush=True)
import tensorflow as tf
tf.keras.utils.get_custom_objects().clear()
model = load_model(MODEL_PATH, compile=False)

# Load face detector
cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Warm up model
print("[startup] Warming up...", flush=True)
model.predict(np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32), verbose=0)

print(f"[startup] Ready  -->  http://0.0.0.0:{os.environ.get('PORT', 5001)}", flush=True)

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/", methods=["GET"])
def index():
    """Serve the frontend; fall back to JSON health check."""
    from flask import send_file
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(html_path):
        return send_file(html_path)
    return jsonify({"status": "ready"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return add_cors(make_response()), 204

    data = request.get_json(silent=True) or {}
    b64  = data.get("image", "")

    if not b64:
        return jsonify({"status": "ready"})

    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        pil   = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        frame = np.array(pil)
        bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return jsonify({"error": "no_face"})

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    pad = int(0.1 * min(w, h))
    H, W = frame.shape[:2]
    face_img = frame[
        max(0, y - pad):min(H, y + h + pad),
        max(0, x - pad):min(W, x + w + pad)
    ]

    arr = np.array(
        Image.fromarray(face_img).resize((INPUT_SIZE, INPUT_SIZE)),
        dtype=np.float32
    ) / 255.0

    preds = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    top_i = int(np.argmax(preds))

    return jsonify({
        "emotion":    EMOTION_MAP[top_i],
        "confidence": round(float(preds[top_i]) * 100, 2),
        "scores": {
            EMOTION_MAP[i]: round(float(preds[i]) * 100, 2)
            for i in range(7)
        },
        "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
