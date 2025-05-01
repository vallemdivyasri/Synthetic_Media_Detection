import pickle
import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from skimage.feature import hog
import time

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TEMPLATES_AUTO_RELOAD"] = True  # Disable template caching

# Load model
with open("model.pkl", "rb") as file:
    model, scaler = pickle.load(file)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * total_frames // num_frames)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (64, 64))
        hog_features = hog(frame, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        frames.append(hog_features)

    cap.release()
    
    if frames:
        return np.mean(frames, axis=0)
    return None

@app.route("/", methods=["GET", "POST"])
def home():
    timestamp = int(time.time())  # Unique timestamp for cache busting

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(url_for("home", message="No file uploaded.", image_file=None))

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return redirect(url_for("home", message="Invalid file format.", image_file=None))

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        features = extract_features(filepath)
        if features is None:
            return redirect(url_for("home", message="Could not extract features.", image_file=None))

        features = scaler.transform([features])
        prediction = model.predict(features)[0]
        result = "Fake" if prediction == 1 else "Real"
        image_file = "Real_image.jpg" if result == "Real" else "fake_image.webp"

        # Redirect to GET with result as query parameters
        return redirect(url_for("home", message=result, image_file=image_file))

    # Handle GET request (initial load or refresh)
    message = request.args.get("message")
    image_file = request.args.get("image_file")
    response = render_template("index.html", message=message, image_file=image_file, timestamp=timestamp)
    return response

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)