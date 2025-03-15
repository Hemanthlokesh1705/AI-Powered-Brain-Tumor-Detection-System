from flask import Flask, render_template, request
import os
import numpy as np
import cv2
import tensorflow as tf


MODEL_PATH = "brain_tumor_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)


app = Flask(__name__)


UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def predict_tumor(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0 
    img = img.reshape(1, 128, 128, 1)  

    prediction = model.predict(img)[0][0]
    return "Tumor Detected" if prediction > 0.5 else "No Tumor"


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    result = predict_tumor(file_path)
    return render_template("result.html", prediction=result, image_path=file_path)


if __name__ == "__main__":
    app.run(debug=True)
