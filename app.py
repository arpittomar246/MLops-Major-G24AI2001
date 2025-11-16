
from venv import logger
from flask import Flask, request, render_template, jsonify, send_from_directory
import joblib
import numpy as np
from PIL import Image
import io
import os
from sklearn.metrics import accuracy_score
import base64
import random

MODEL_PATH = "models/savedmodel.pth"
TESTDATA_PATH = "models/test_data.npz" 
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}

app = Flask(__name__, static_folder="static", template_folder="templates")


def safe_load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def load_test_metrics(testdata_path=TESTDATA_PATH, clf=None):
    """Return accuracy and n_classes if available, else None"""
    if not os.path.exists(testdata_path) or clf is None:
        return None, None
    d = np.load(testdata_path)
    X_test = d["X_test"]
    y_test = d["y_test"]
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    try:
        n_classes = int(len(np.unique(y_test)))
    except Exception:
        n_classes = None
    return acc, n_classes


MODEL = safe_load_model()
MODEL_INFO = {}
if MODEL is not None:
    acc, n_classes = load_test_metrics(TESTDATA_PATH, MODEL)
    MODEL_INFO = {
        "model_path": MODEL_PATH,
        "n_classes": n_classes,
        "test_accuracy": acc,
        "has_proba": hasattr(MODEL, "predict_proba"),
    }
else:
    MODEL_INFO = {
        "model_path": MODEL_PATH,
        "n_classes": None,
        "test_accuracy": None,
        "has_proba": False,
    }

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_bytes(file_bytes):
    """Preprocess to match Olivetti: convert to grayscale, resize to 64x64, normalize to 0..1, flatten"""
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = img.resize((64, 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)

@app.route("/")
def index():

    return render_template("index.html", model_info=MODEL_INFO)

@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": f"Model not available on server. Expected at {MODEL_PATH}"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Allowed: png, jpg, jpeg, bmp, gif"}), 400

    try:
        file_bytes = file.read()
        X = preprocess_image_bytes(file_bytes) 
        pred_label = int(MODEL.predict(X)[0])
        predicted_name = f"Person {pred_label}"

   
        topk = []
        if MODEL_INFO.get("has_proba", False):
            proba = MODEL.predict_proba(X)[0]
            idx_sorted = np.argsort(proba)[::-1][:3]
            for idx in idx_sorted:
                topk.append({"class": int(idx), "prob": float(proba[idx]), "name": f"Person {int(idx)}"})
        else:
            topk = [{"class": pred_label, "prob": None, "name": predicted_name}]


        rep_image_b64 = None
        true_label_of_sample = None
        if os.path.exists(TESTDATA_PATH):
            try:
                d = np.load(TESTDATA_PATH)
                X_test = d["X_test"]
                y_test = d["y_test"]
    
                idxs = np.where(y_test == pred_label)[0]
                if len(idxs) > 0:
                    idx0 = int(idxs[0])
                    arr = (X_test[idx0].reshape(64, 64) * 255.0).astype("uint8")
                    img = Image.fromarray(arr, mode="L")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    rep_image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    true_label_of_sample = int(y_test[idx0])
            except Exception:
     
                rep_image_b64 = None

        response = {
            "predicted_class": pred_label,
            "predicted_name": predicted_name,
            "topk": topk,
            "representative_image_base64": rep_image_b64,
            "representative_true_label": true_label_of_sample,
            "model_info": MODEL_INFO,
        }
        return jsonify(response)
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Server error while predicting: {str(e)}"}), 500
    

def load_sample_test_images(n_samples=12):

    if not os.path.exists(TESTDATA_PATH):
        return []

    data = np.load(TESTDATA_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"]


    idxs = random.sample(range(len(X_test)), min(n_samples, len(X_test)))

    samples = []
    for idx in idxs:
        arr = X_test[idx].reshape(64, 64) * 255.0    
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        samples.append({
            "img_base64": img_b64,
            "true_label": int(y_test[idx])
        })

    return samples


@app.route("/samples")
def samples():
    if not os.path.exists(TESTDATA_PATH):
        return "Test data not found. Please run train.py first.", 500

    test_samples = load_sample_test_images(n_samples=12)
    return render_template("samples.html", samples=test_samples)

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(os.path.join(app.root_path, "static"),
                               "favicon.ico", mimetype="image/vnd.microsoft.icon")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
