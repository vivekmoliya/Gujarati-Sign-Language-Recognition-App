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

# Load the trained SVM model
# model = joblib.load('model.pkl')
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?export=download&id=1ja640I0nxK9gilwNytH7EL3bx3dJakQb"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded!")

# model = joblib.load(MODEL_PATH)
# model = joblib.load('svm_model (4).pkl')
# Now load the model

# Gujarati classes
# Gujarati classes
classes = {
    0: 'ક',   1: 'ખ',   2: 'ગ',   3: 'ઘ',   4: 'ચ',   5: 'છ',
    6: 'જ',   7: 'ઝ',   8: 'ટ',   9: 'ઠ',   10: 'ડ',   11: 'ઢ',
    12: 'ણ',   13: 'ત',   14: 'થ',   15: 'દ',   16: 'ધ',   17: 'ન',
    18: 'પ',   19: 'ફ',   20: 'બ',   21: 'ભ',   22: 'મ',   23: 'ય',
    24: 'ર',   25: 'લ',   26: 'વ',   27: 'શ',   28: 'ષ',   29: 'સ',
    30: 'હ',   31: 'ળ',   32: 'ક્ષ', 33:'જ્ઞ'
}


def preprocess_image(image_path):
    """Preprocess the uploaded image to match the model's input format."""
    img = cv2.imread(image_path, 0)  # Read in grayscale
    img = cv2.resize(img, (200, 200))  # Resize to 200x200
    img = img.flatten() / 255.0  # Flatten and normalize
    return np.array([img])

def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print("Error loading model:", e)
        return None

@app.route('/')
def index():
    return render_template('index.html')  # Main page for uploading files

@app.route('/camera')
def camera():
    return render_template('camera.html')  # Camera page

@app.route('/predict', methods=['POST'])
def predict():
    # Lazy load the model inside the route
    model = load_model()
    
    if model is None:
        return "Model not found. Cannot predict.", 500

    # Handle camera image
    image_data = request.form.get('image_data')
    if image_data:
        # Decode the base64 image
        image_data = image_data.split(',')[1]  # Remove the data URL prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert the image to a format that can be displayed as an image source
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Predict using the model
        image = Image.open(BytesIO(image_bytes))
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((200, 200))  # Resize to match model input
        input_data = np.array(image).flatten() / 255.0
        input_data = np.array([input_data])

        # Predict using the model
        prediction = model.predict(input_data)
        print("Prediction:", prediction)  # Debugging print statement
        
        # Check if the prediction is in the range of classes
        if prediction[0] in classes:
            predicted_label = classes[prediction[0]]
        else:
            predicted_label = "Unknown Prediction"

        return render_template('result.html', prediction=predicted_label, captured_image=image_base64)

    # Handle uploaded image from the main page
    if 'file' in request.files:
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        input_data = preprocess_image(filepath)
        prediction = model.predict(input_data)
        print("Prediction:", prediction)  # Debugging print statement
        
        # Check if the prediction is in the range of classes
        if prediction[0] in classes:
            predicted_label = classes[prediction[0]]
        else:
            predicted_label = "Unknown Prediction"

        return render_template('result.html', image_path=filepath, prediction=predicted_label)

    return redirect(request.url)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
