from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import joblib
import os
from PIL import Image
import base64
from io import BytesIO
import urllib.request
import gdown

# Initialize Flask app
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model path
MODEL_PATH = "svm_model.pkl"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?export=download&id=1CPEZxfe9DwlPwTXiQerTOqtG17LfNiXz"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded!")

# Load the model once globally
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    model = None

# Gujarati classes
classes = {
    0: 'ક', 1: 'ખ', 2: 'ગ', 3: 'ઘ', 4: 'ચ', 5: 'છ',
    6: 'જ', 7: 'ઝ', 8: 'ટ', 9: 'ઠ', 10: 'ડ', 11: 'ઢ',
    12: 'ણ', 13: 'ત', 14: 'થ', 15: 'દ', 16: 'ધ', 17: 'ન',
    18: 'પ', 19: 'ફ', 20: 'બ', 21: 'ભ', 22: 'મ', 23: 'ય',
    24: 'ર', 25: 'લ', 26: 'વ', 27: 'શ', 28: 'ષ', 29: 'સ',
    30: 'હ', 31: 'ળ', 32: 'ક્ષ', 33: 'જ્ઞ'
}

def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)  # Grayscale
    img = cv2.resize(img, (200, 200))
    img = img.flatten() / 255.0
    return np.array([img])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return "Model not loaded. Cannot predict.", 500

        image_data = request.form.get('image_data')
        if image_data:
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert('L').resize((200, 200))
            input_data = np.array(image).flatten() / 255.0
            input_data = np.array([input_data])

            prediction = model.predict(input_data)
            predicted_label = classes.get(prediction[0], "Unknown")
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            return render_template('result.html', prediction=predicted_label, captured_image=image_base64)

        if 'file' in request.files:
            file = request.files['file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            input_data = preprocess_image(filepath)
            prediction = model.predict(input_data)
            predicted_label = classes.get(prediction[0], "Unknown")

            return render_template('result.html', prediction=predicted_label, image_path=filepath)

        return redirect(request.url)

    except Exception as e:
        print("Error during prediction:", e)
        return f"Internal Server Error: {e}", 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
