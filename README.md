# Synthetic_Media_Detection

This project is a web-based application for detecting DeepFake videos using machine learning and computer vision techniques. It allows users to upload a video file, extracts visual features using Histogram of Oriented Gradients (HOG), and classifies the video as either Real or Fake using a pre-trained model.

## Model & Features
- **Machine Learning Model:** Trained on extracted HOG features from video frames.
- **Feature Extraction:** Uses grayscale conversion, resizing, and HOG feature extraction on multiple frames of a video.
- **Scaler:** Preprocessing using a fitted scaler for consistent feature scaling.
- **Classifier:** Binary classifier saved as model.pkl.

During experimentation, we tested four deep learning architectures for DeepFake detection:
- Xception (Deployed)
- Inception ResNet V2
- Vision Transformer (ViT)
- EfficientNet B7

Among these, Xception achieved the highest accuracy and best performance on our dataset. Therefore, the model based on Xception was selected, saved as a .pkl file, and integrated into the Flask web application for real-time predictions.

## Feature Extraction
Videos are processed to extract 5 key frames. Each frame undergoes:
- Grayscale conversion
- Resizing
- HOG (Histogram of Oriented Gradients) feature extraction
- Extracted features are averaged and passed to the trained classifier.

## How It Works
Upload a video file (.mp4, .avi, .mov, .mkv)
The backend extracts features from key frames. Features are scaled and input to the Xception-based model. Output is prediction displayed as Real or Fake, along with an illustrative image.

## Project Structure
├── app.py               # Flask app backend

├── model.pkl            # Trained Xception model + scaler

├── templates/

│   └── index.html       # Web interface

├── uploads/             # Uploaded videos (temporary)

├── FinalComparisonOfModels.ipynb  # Model evaluation notebook

├── Copy_of_FinalDeepFakeDetection.ipynb  # Full pipeline notebook

## Dependencies
Required packages include:
- Flask
- scikit-learn
- OpenCV (opencv-python)
- scikit-image
- NumPy
