
--------------------------------------------------------------------------------
# 📍 Campus Image-to-GPS Localization (Project 4)
### Visual Place Recognition using Multi-Task "Trinity" Architecture

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/roeitheyosef/campus-gps-locator)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Authors:** Roei Azariya Yosef, Ayala Egoz, Yair Michael Avisar  
**Course:** Introduction to Deep Learning, Ben-Gurion University  
**Best Validation Error:** 8.68 meters

---

## 🚀 Live Demo
We have deployed our model to **Hugging Face Spaces**. You can upload any image from the BGU campus and get its predicted location on an interactive map.

👉 **[Click here to try the Live Demo](https://huggingface.co/spaces/roeitheyosef/campus-gps-locator)**

---

## 📖 Overview
Standard GPS signals in dense urban environments (like university campuses) suffer from the **"Multipath Effect"**, causing deviations of 10-20 meters. This project proposes a robust **Visual Place Recognition (VPR)** system that regresses precise GPS coordinates from a single monocular image.

Our solution implements a novel **"Multi-Head Trinity Architecture"** based on a fine-tuned ResNet50 backbone. By simultaneously optimizing for **Coordinate Regression**, **Coarse Classification** (Smart Zones), and **Metric Learning** (Triplet Loss), we achieved a state-of-the-art mean localization error of **8.68m**, outperforming standard regression baselines.

---

## 🧠 The "Trinity" Architecture
The model processes a 224x224 image through a ResNet50 backbone injected with **Spatial Dropout** layers, branching into three task-specific heads:

1.  **Regression Head (MSE):** Predicts the precise $(x, y)$ coordinates.
2.  **Classification Head (Cross-Entropy):** Classifies the image into one of **300 Smart Zones** (generated via K-Means) to provide global context and prevent "mean location collapse".
3.  **Embedding Head (Triplet Loss):** Learns a metric space where visually similar but geographically distant locations (aliasing) are pushed apart using **Hard Negative Mining**.

---

## ⚙️ Data Preprocessing & Organization

To ensure reproducibility and robustness, we organized the data into two distinct formats.

### 1. Folder Structure
*   **`data/raw` (Submission Format):** Contains the raw images and the `gt.csv` file exactly as requested in the project guidelines.
*   **`data/processed` (Training Ready):** Contains the data after our rigorous preprocessing pipeline. **We recommend training on this folder.**

### 2. Preprocessing Pipeline
Our `train.py` script utilizes the processed data, which underwent the following critical steps:

*   **🚫 GPS Denoising:** We manually corrected raw sensor drifts (which deviated up to 15m) by cross-referencing with satellite maps, creating a clean "Ground Truth".
*   **🧩 Smart Zoning (K-Means):** We clustered the campus coordinates into **300 discrete zones**. This allows the model to learn a "coarse-to-fine" approach.
*   **⚖️ Weighted Random Sampling:** To handle data imbalance (e.g., popular plazas vs. sparse corridors), we calculated sampling weights based on the inverse frequency of each zone.
*   **☀️ Day/Night Stratification:** We ensured that night images (approx. 20%) are equally represented in both training and validation splits.

---

## 🛠️ Environment Setup
To replicate our results, please strictly follow these steps to create a clean Conda environment with the required dependencies (including `utm` and `pillow-heif`).

```bash
# 1. Create a clean environment with Python 3.9
conda create -n gps_project python=3.9

# 2. Activate the environment
conda activate gps_project

# 3. Install required dependencies
pip install -r requirements.txt

--------------------------------------------------------------------------------
🚀 How to Run
1. Training
To train the model from scratch (using the "Trinity" loss and the processed dataset):
python train.py
Note: The script automatically handles the weighted sampling and data loading.
2. Inference (Evaluation)
We provide a standalone function predict_gps that accepts a numpy array image and returns coordinates.
Example usage:
import numpy as np
from PIL import Image
from predict import predict_gps

# Load an image
img_path = "data/raw/images/some_campus_image.jpg"
image = np.array(Image.open(img_path))

# Predict
# Returns: np.array([latitude, longitude], dtype=float32)
coords = predict_gps(image)

print(f"Predicted Location: {coords}")

--------------------------------------------------------------------------------
📊 Results & Analysis
• Best Validation Error: 8.68m (Epoch 127).
• Robustness: The model successfully handles night scenes and visual aliasing thanks to the Hard Negative Mining strategy.
Error Distribution Map
 Blue dots: Ground Truth | Gray dots: Predictions | Red lines: Error vectors.

--------------------------------------------------------------------------------
📂 Repository Structure
Campus_GPS_Project/
├── app.py               # Streamlit Demo Application
├── predict.py           # Inference function (Submission Requirement)
├── train.py             # Main training loop
├── model.py             # The VPSModel (Trinity Architecture) class
├── requirements.txt     # Dependencies
├── best_model.pth       # Pre-trained weights (Download link in Release)
├── README.md            # Project Documentation
└── data/
    ├── raw/             # Format as requested by TA
    └── processed/       # Denoised & Clustered data
