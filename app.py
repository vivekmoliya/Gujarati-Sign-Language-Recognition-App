from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import joblib
import os
from PIL import Image
import base64
from io import BytesIO
import gdown
from gtts import gTTS
import mediapipe as mp

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
AUDIO_FOLDER = './static/audio'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model download if missing
MODEL_PATH = "svm_model.pkl"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?export=download&id=1CPEZxfe9DwlPwTXiQerTOqtG17LfNiXz"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded!")

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# Gujarati character mapping
classes = {
    0: 'ક', 1: 'ખ', 2: 'ગ', 3: 'ઘ', 4: 'ચ', 5: 'છ',
    6: 'જ', 7: 'ઝ', 8: 'ટ', 9: 'ઠ', 10: 'ડ', 11: 'ઢ',
    12: 'ણ', 13: 'ત', 14: 'થ', 15: 'દ', 16: 'ધ', 17: 'ન',
    18: 'પ', 19: 'ફ', 20: 'બ', 21: 'ભ', 22: 'મ', 23: 'ય',
    24: 'ર', 25: 'લ', 26: 'વ', 27: 'શ', 28: 'ષ', 29: 'સ',
    30: 'હ', 31: 'ળ', 32: 'ક્ષ', 33: 'જ્ઞ'
}

# ========= HAND-ONLY VALIDATION ==========

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

def verify_hand_only(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Lowered confidence threshold to 0.4
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.4) as hands:
        hand_results = hands.process(image_rgb)

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.4) as faces:
        face_results = faces.process(image_rgb)

    hand_detected = hand_results.multi_hand_landmarks is not None
    face_detected = face_results.detections is not None

    return hand_detected and not face_detected

# ========= PREPROCESS ==========

def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (200, 200))
    img = img.flatten() / 255.0
    return np.array([img])

# ========= ROUTES ==========

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

        predicted_label = None
        image_base64 = None

        # ---- Camera Capture ----
        if request.form.get('image_data'):
            image_data = request.form['image_data'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            image_np = np.array(image)

            # Validate hand only
            if not verify_hand_only(image_np):
                return render_template("error.html", message="Only clear hand signs allowed. No random images or faces.")

            # Resize & predict
            image_gray = image.convert('L').resize((200, 200))
            input_data = np.array(image_gray).flatten() / 255.0
            prediction = model.predict([input_data])
            predicted_label = classes.get(prediction[0], "Unknown")
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # ---- File Upload ----
        elif 'file' in request.files:
            file = request.files['file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image_np = cv2.imread(filepath)

            # Validate hand only
            if not verify_hand_only(image_np):
                os.remove(filepath)
                return render_template("error.html", message="Only upload a clear hand sign image. No random objects, text, or face.")

            input_data = preprocess_image(filepath)
            prediction = model.predict(input_data)
            predicted_label = classes.get(prediction[0], "Unknown")

            with open(filepath, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')

        if predicted_label:
            if not os.path.exists(AUDIO_FOLDER):
                os.makedirs(AUDIO_FOLDER)

            audio_filename = f"{predicted_label}.mp3"
            audio_path = os.path.join(AUDIO_FOLDER, audio_filename)

            if not os.path.exists(audio_path):
                tts = gTTS(text=predicted_label, lang='gu')
                tts.save(audio_path)

            return render_template('result.html',
                                   prediction=predicted_label,
                                   captured_image=image_base64,
                                   audio_file=audio_filename)

        return redirect(request.url)

    except Exception as e:
        print("Error during prediction:", e)
        return f"Internal Server Error: {e}", 500

# ========= MAIN ==========

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
