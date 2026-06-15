# SketchNova

A Touchless Human-Computer Interaction Interface powered by Computer Vision and Hand Gesture Recognition.

## Features

- Air Drawing
- Virtual Whiteboard
- Handwriting Recognition
- Gesture-Based Controls
- Real-time Hand Tracking
- Touchless Human-Computer Interaction

## Technologies Used

- Python
- OpenCV
- MediaPipe
- FastAPI
- PyTorch
- HTML/CSS/JavaScript

## Project Structure

backend.py          # Backend server
predict_handwriting.py
train_model.py
index.html
model/handwriting_model.pth

## Dataset

The dataset is not included in this repository.

Download:
https://www.kaggle.com/code/abdallahsaadelgendy/a-z-handwritten-alphabets-recognizer-with-cnn#About-Dataset

Place it inside:

dataset/az_handwritten.csv

## Run

pip install -r requirements.txt

python backend.py
