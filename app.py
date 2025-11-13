# app.py
from flask import Flask, request, render_template_string, redirect, url_for
import joblib
import numpy as np
from PIL import Image
import io

MODEL_PATH = "models/savedmodel.pth"

app = Flask(__name__)

def load_model():
    return joblib.load(MODEL_PATH)

def preprocess_image(file_stream):
    # Olivetti images are 64x64 grayscale normalized between 0 and 1.
    img = Image.open(io.BytesIO(file_stream)).convert("L")  # convert to grayscale
    img = img.resize((64,64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        '''
        <h2>Olivetti Face Classifier</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
          <input type="file" name="image"/>
          <input type="submit" value="Upload & Predict"/>
        </form>
        '''
    )

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return "No file uploaded", 400
    file_bytes = file.read()
    X = preprocess_image(file_bytes)
    clf = load_model()
    pred = int(clf.predict(X)[0])
    return f"Predicted class: {pred}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
