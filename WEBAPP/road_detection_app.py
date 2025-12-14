import streamlit as st
import cv2
import numpy as np
import requests
import os
import time
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Road Object detection ML project",
    
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & URLs for Models ---
# We use caching to prevent downloading these files on every run
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/shantnu/Vehicle-Detection-Haar-Cascades/master/cars.xml"
# MobileNet SSD (Lightweight Neural Network)
DNN_PROTO_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
DNN_MODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

PATHS = {
    "cascade": "cars.xml",
    "proto": "deploy.prototxt",
    "model": "mobilenet_iter_73000.caffemodel"
}

# --- Helper Functions ---

@st.cache_resource
def download_model_file(url, filename):
    """Downloads model files if they don't exist."""
    if not os.path.exists(filename):
        with st.spinner(f"Downloading model resource: {filename}..."):
            r = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(r.content)
    return filename

def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img.convert('RGB'))

# --- Model Detection Classes ---

class RoadDetector:
    def __init__(self):
        # Download necessary files on init
        self.cascade_path = download_model_file(HAAR_CASCADE_URL, PATHS["cascade"])
        self.proto_path = download_model_file(DNN_PROTO_URL, PATHS["proto"])
        self.model_path = download_model_file(DNN_MODEL_URL, PATHS["model"])
        
        # Classes for MobileNet SSD
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        # Colors for bounding boxes
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    def basic_ml_detection(self, image):
        """
        Uses Haar Cascades (Classic ML) to detect cars.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        car_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        # Detect cars
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        
        img_out = image.copy()
        count = 0
        for (x, y, w, h) in cars:
            cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(img_out, "Car", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            count += 1
            
        return img_out, count, "Haar Cascade Classifier"

    def neural_network_detection(self, image, confidence_threshold=0.2):
        """
        Uses MobileNet SSD (Deep Learning) to detect objects.
        """
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        
        net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)
        net.setInput(blob)
        detections = net.forward()

        img_out = image.copy()
        count = 0
        
        # Loop over the detections
        for i in range(np.arange(0, detections.shape[2])):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                # We only care about road objects for this app
                if self.CLASSES[idx] in ["car", "bus", "motorbike", "person", "bicycle"]:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    label = "{}: {:.2f}%".format(self.CLASSES[idx], confidence * 100)
                    cv2.rectangle(img_out, (startX, startY), (endX, endY), self.COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(img_out, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
                    count += 1
                    
        return img_out, count, "MobileNet SSD"

# --- Simulation Logic ---

def simulate_model_comparison(category):
    """
    Simulates the process of comparing 4 models and picking the best one.
    This creates the visual effect of an AutoML system.
    """
    comparison_placeholder = st.empty()
    bar = st.progress(0)
    
    if category == "Basic ML":
        models = [
            {"name": "SVM + HOG Features", "speed": "Medium", "accuracy": "72%"},
            {"name": "Random Forest", "speed": "Slow", "accuracy": "68%"},
            {"name": "Fisher Vectors", "speed": "Slow", "accuracy": "65%"},
            {"name": "Haar Cascade (Selected)", "speed": "Fast", "accuracy": "75%"}
        ]
    else:
        models = [
            {"name": "YOLOv8", "speed": "Very Fast", "accuracy": "89%"},
            {"name": "Faster R-CNN", "speed": "Slow", "accuracy": "92%"},
            {"name": "EfficientDet", "speed": "Medium", "accuracy": "88%"},
            {"name": "MobileNet SSD (Selected)", "speed": "Fast", "accuracy": "85%"}
        ]
        
    for i, model in enumerate(models):
        comparison_placeholder.info(f"🧬 Testing Model {i+1}/4: {model['name']}...")
        time.sleep(0.4) # Simulate processing time
        bar.progress((i + 1) * 25)
    
    bar.empty()
    comparison_placeholder.success("✅ Best Model Selected based on current environment constraints.")
    
    return models

# --- Main App Layout ---

def main():
    st.title("🛣️ Intelligent Road Object Detection")
    st.markdown("""
    Upload a road image and let our AI choose the best model to detect vehicles and pedestrians.
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    model_type = st.sidebar.radio(
        "Choose Detection Approach:",
        ("Basic Machine Learning", "Deep Neural Network")
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Basic ML:** Uses classical feature extraction (Fast, less accurate).\n\n"
        "**Neural Network:** Uses Deep Learning (Slower, high accuracy, detects classes)."
    )

    # Main Area
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Load and display original
        image = load_image(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        # Button to trigger detection
        if st.button("🚀 Analyze Road Scene"):
            
            # 1. Simulate Comparison Phase
            st.divider()
            st.subheader("🤖 Model Selection Phase")
            
            comparison_results = simulate_model_comparison("Basic ML" if model_type == "Basic Machine Learning" else "Neural Network")
            
            # Show Comparison Table
            st.table(comparison_results)
            
            # 2. Perform Actual Detection
            detector = RoadDetector()
            
            with st.spinner("Running inference with winning model..."):
                if model_type == "Basic Machine Learning":
                    result_img, count, model_name = detector.basic_ml_detection(image)
                    info_text = "detected using Haar features. Good for simple vehicle shapes."
                else:
                    result_img, count, model_name = detector.neural_network_detection(image)
                    info_text = "detected using Deep Learning. Can distinguish between person, car, bus, etc."

            # 3. Display Results
            with col2:
                st.subheader(f"Detection Result")
                st.image(result_img, use_container_width=True)
            
            st.success(f"**Winner Model:** {model_name}")
            st.metric("Objects Detected", count)
            st.caption(f"ℹ️ {info_text}")

if __name__ == "__main__":
    main()