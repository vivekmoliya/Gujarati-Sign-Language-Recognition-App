# Gujarati Sign Language Recognition

This project is an AI-powered web application that recognizes **Gujarati Sign Language (Vyanjan)** gestures from camera-captured images and converts them into text. Built with **Flask**, **CNN-based model**, and optionally **MediaPipe**, the app helps improve communication for the deaf and hard-of-hearing community.


## 🚀 Features

- 📷 Live camera input for gesture recognition
- 🤖 Trained CNN model to classify Gujarati sign language letters
- 🧠 Optionally supports hand detection using MediaPipe (experimental)
- 🌐 User-friendly Flask web interface
- ✅ Real-time prediction and result display


## 🧪 Dataset Structure

dataset/ │ ├── train/ │ ├── ક/ (12 images) │ ├── ખ/ │ └── ... (34 vyanjans total) │ ├── test_data/ │ └── test/ │ ├── test(1).jpg │ ├── test(2).jpg │ └── ...


Each class has hand sign images for training. Test images are unlabeled.


## 🧠 Model Training

- Preprocessed images to standard sizes (`256x256` and `512x512`)
- Augmented using rotation (0°, 90°, 180°, 270°)
- Model trained using a CNN with high accuracy (~90% without MediaPipe)
- MediaPipe was also tested for hand segmentation, but resulted in lower accuracy (~70%) due to input mismatch.


## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Python, Flask
- **ML/DL**: TensorFlow / Keras, OpenCV

