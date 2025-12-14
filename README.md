# Road Object Detection

## 📌 Overview
This project focuses on detecting and classifying road objects using two approaches:
1. **Basic ML Models** (SVM, Logistic Regression, KNN, Random Forest) on cropped grayscale images.
2. **YOLOv8 Neural Network** for real-time object detection using bounding boxes.

## 📂 Dataset
- **Source:** [Roboflow Road Traffic Dataset](https://universe.roboflow.com/roboflow-100/road, 'fire hydrants', 'motorcycles', 'traffic lights', 'vehicles']
- **Image Size:** 
  - Basic ML: 64×64 grayscale cropped objects
  - YOLO: 640×640 original images
- **Split:** Train / Validation / Test

## 🛠 Models Used
- **Basic ML Models:**
  - SVM with HOG features
  - Logistic Regression with engineered features
  - KNN with color histogram
  - Random Forest with SIFT features
- **Deep Learning:**
  - YOLOv8 (PyTorch-based)

## ✅ Evaluation Metrics
- **Basic ML:** Accuracy, Precision, Recall, F1-score (Macro & Weighted)
- **YOLOv8:** mAP@0.5, mAP@0.5:0.95, Precision, Recall

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/road-object-detection.git
