# Campus Image-to-GPS Localization Project

**Authors:** Roei Azariya Yosef, Ayala Egoz, Yair Michael Avisar  
**Course:** Introduction to Deep Learning, Ben-Gurion University  
**Best Validation Error:** 8.68 meters

---

## 🚀 Live Demo
We have deployed our model to **Hugging Face Spaces**. You can upload any image from the BGU campus and get its predicted location on an interactive map.

👉 **[Click here to try the Live Demo](https://huggingface.co/spaces/roeitheyosef/campus-gps-locator)**

---


## Overview
In this repository is a deep learning solution for **Image-to-GPS Regression**. The goal is to predict the precise real-world location (Latitude, Longitude) of a photo taken within the university campus, utilizing only visual features.


## The model 
processes a 224x224 image through a ResNet50 backbone injected with Spatial Dropout layers, branching into three task-specific heads:
Regression Head (MSE): Predicts the precise $(x, y)$ coordinates.
Classification Head (Cross-Entropy): Classifies the image into one of 300 Smart Zones (generated via K-Means) to provide global context.
Embedding Head (Triplet Loss): Learns a metric space where visually similar but geographically distant locations (aliasing) are pushed apart using Hard Negative Mining.

###  The model 
The model processes a standard $224 \times 224$ input image through a **ResNet50 backbone**, which we fine-tuned and injected with **Spatial Dropout** layers to enhance feature robustness. The extracted feature vector branches into three parallel, task-specific heads:

*   ** Regression Head (Geometric Precision):**
    Directly predicts the precise $(x, y)$ coordinates using **MSE Loss**. This head focuses on minimizing the meter-level distance error.

*   ** Classification Head (Global Context):**
    Classifies the image into one of **300 "Smart Zones"** (generated via K-Means clustering). This provides a coarse location estimate and prevents "mean location collapse" by enforcing commitment to a specific neighborhood [1].

*   ** Embedding Head (Metric Learning):**
    Projects the image into a metric space optimized via **Triplet Loss** with **Hard Negative Mining**. This head learns to distinguish between visually similar but geographically distant locations (solving the "Visual Aliasing" problem) [2, 3].

---

## 📥 Data & Model Setup (Crucial Step)
Due to file size limits, the dataset and model weights are hosted on Google Drive. 
**You must download and place them correctly for the code to run.**

### 1. Download Model Weights
Required for `predict.py` and `check_submission.py`.

* **Download:** `best_model.pth`
* **Link:** [INSERT_LINK_TO_BEST_MODEL_PTH]
* **Action:** Place the file in the **root directory** of the project (next to `predict.py`).

### 2. Download Dataset
We provide two options. **Option A is recommended** for immediate reproduction of training.

#### Option A: Preprocessed Data (Ready to Train)
This version includes the processed CSV (has image_name, lat, lon, utm_x, utm_y, is_night, label)

* **Link:** [INSERT_LINK_TO_PROCESSED_DATA_ZIP]
* **Action:**
    1. Download and unzip.
    2 Place the `images` folder inside `data/`.
    3. Place the `gt.csv` file inside `data/`.
    4. **Status:** You can run `python train.py` immediately.

#### Option B: Raw Data (Submission Format)
This is the original dataset format as required by the submission guidelines, and this version includes the CSV as required( has image_name, lat, lon)

* **Link:** [INSERT_LINK_TO_RAW_DATA_ZIP]
* **Action:**
    1. Download and unzip to `data/raw/`.
    2. **Preprocessing Required:** You must run the preprocessing script to generate the smart zones and handle GPS noise before training.
    ```bash
    python preprocess.py  # Generates a new preprocessed folder with the processed photos and the newly updated CSV
    ```

**Final Project Structure:**
```text
Campus_GPS_Project/
├── best_model.pth          <-- Downloaded Model
├── predict.py
├── train.py
├── preprocess.py           <-- (Only if using Raw Data)
└── data/
    ├── images/             <-- Downloaded Images
    └── gt.csv              <-- Downloaded/Generated CSV
```


## 🛠️ Environment Setup
To replicate our results, please strictly follow these steps to create a clean Conda environment with the required dependencies (including `utm` and `pillow-heif`).

```bash
# 1. Create a clean environment with Python 3.9
conda create -n gps_project python=3.9 -y

# 2. Activate the environment
conda activate gps_project

# 3. Install required dependencies
pip install -r requirements.txt
```
---

### 🚀 How to Run
#### 1. Training To train the model from scratch (ensure you followed "Data Setup Option A"):
``` bash
 python train.py
```
### 2. Inference (Evaluation)
We provide a standalone function `predict_gps` that accepts a numpy array image and returns coordinates, exactly as required.

**Example Usage:**

```python
import numpy as np
from PIL import Image
from predict import predict_gps

# 1. Load an image (Standard RGB)
img_path = "data/images/some_campus_image.jpg"
image = np.array(Image.open(img_path).convert('RGB'))

# 2. Predict Coordinates
# Returns: np.array([latitude, longitude], dtype=float32)
coords = predict_gps(image)

print(f"Predicted Location: {coords}")
# Output: [31.262345 34.803210]
```
---
