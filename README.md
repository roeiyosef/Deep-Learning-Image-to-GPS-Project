# Campus Image-to-GPS Localization Project

**Authors:** Roei Azariya Yosef, Ayala Egoz, Yair Michael Avisar  
**Course:** Introduction to Deep Learning, Ben-Gurion University  
**Best Validation Error:** 8.68 meters

---

## Live Demo
We have deployed our model to **Hugging Face Spaces**. You can upload any image from the BGU campus and get its predicted location on an interactive map.

**[Click here to try the Live Demo](https://huggingface.co/spaces/roeitheyosef/campus-gps-locator)**

---


## Overview
In this repository is a deep learning solution for **Image-to-GPS Regression**. The goal is to predict the precise real-world location (Latitude, Longitude) of a photo taken within the university campus, utilizing only visual features.

###  The model 
The model processes a standard $224 \times 224$ input image through a **ResNet50 backbone**, which we fine-tuned and injected with **Spatial Dropout** layers to enhance feature robustness. The extracted feature vector branches into three parallel, task-specific heads:

*   **Regression Head (Geometric Precision):**
    Directly predicts the precise $(x, y)$ coordinates using **MSE Loss**. This head focuses on minimizing the meter-level distance error.

*   **Classification Head (Global Context):**
    Classifies the image into one of **300 "Smart Zones"** (generated via K-Means clustering). This provides a global context.

*   **Embedding Head:**
    Extracts compact embeddings that serve as input for the **Triplet Loss**. We use **Hard Negative Mining** on these embeddings to separate confusing scenes that look alike but are far apart.
---

## Data & Model Setup 
Due to file size limits, the dataset and model weights are hosted on Google Drive. 
**You must download and place them correctly for the code to run.**

### 1. Download Model Weights
Required for `predict.py` and `check_submission.py`.

* **Download:** `best_model.pth`
* **Link:** [INSERT_LINK_TO_BEST_MODEL_PTH]
* **Action:** Place the file in the **root directory** of the project (next to `predict.py`).

### 2. Download Dataset
First, in the Project Folder, there is an empty folder called Data. this folder should include the images folder and the CSV.

We provide two options. **Option A is recommended** for immediate reproduction of training.

#### Option A: Preprocessed Data (Ready to Train)
This version includes an images folder with the original images resized to 224*224 and contains the processed CSV (has image_name, lat, lon, utm_x, utm_y, is_night, label)

* **Link:** [INSERT_LINK_TO_PROCESSED_DATA_ZIP]
* **Action:**
    1. Download and unzip, this folder will contain an images folder and a gt.csv.
    2 Place the `images` folder inside `data/`.
    3. Place the `gt.csv` file inside `data/`.
    4. **Status:** You can run `python train.py` immediately.


#### 🟠 Option B: Raw Data (Submission Format)
This option follows the strict submission guidelines but requires an additional preprocessing step before training.

* **Link:** [INSERT_LINK_TO_RAW_DATA_ZIP]
* **Structure:**
    1. Download and unzip the data into the project folder.
    2. **Run Preprocessing:** Execute the following script to denoise GPS labels and generate Smart Zones:
    
    ```bash
    python preprocess.py
    ```
    
    > **Output:** This script creates a new folder named `processed_data/` containing:
    > * `images/` (Optimized images)
    > * `gt.csv` (Updated Ground Truth with "Smart Zone" labels)

    3. move the images folder and the gt.csv from the processed_data to the data folder
    4.  And now you can run `python train.py`.

**Final Project Structure:**
```text
Campus_GPS_Project/
├── best_model.pth          <-- Downloaded Model
├── predict.py
├── model.py
├── train.py
├── preprocess.py           <-- (Only if using Raw Data)
├── requirements.txt
└── data/
    ├── images/             <-- Images
    └── gt.csv              <-- Downloaded/Generated CSV
```


## Environment Setup
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

### How to Run
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
