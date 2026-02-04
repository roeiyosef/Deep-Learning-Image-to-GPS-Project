Campus Image-to-GPS Localization (Project 4)
### Visual Place Recognition using Multi-Task "Trinity" Architecture

**Authors:** Roei Azariya Yosef, Ayala Egoz, Yair Michael Avisar  
**Course:** Introduction to Deep Learning, Ben-Gurion University  
**Best Validation Error:** 8.68 meters

---

## 🚀 Live Demo
We have deployed our model to **Hugging Face Spaces**. You can upload any image from the BGU campus and get its predicted location on an interactive map.

👉 **[Click here to try the Live Demo](https://huggingface.co/spaces/roeitheyosef/campus-gps-locator)**

---


## 📖 Overview
This repository contains a deep learning solution for **Image-to-GPS Regression**. The goal is to predict the precise real-world location (Latitude, Longitude) of a photo taken within the university campus, utilizing only visual features.

Addressing the "Multipath Effect" challenge in urban canyons—where standard GPS signals deviate by 10-20 meters—our model achieves a state-of-the-art mean error of **8.68 meters**, effectively overcoming visual aliasing and extreme lighting variations.

## 🔬 Method

### **Model: The "Trinity" Architecture**
We propose a custom **Multi-Head Network** designed to solve three complementary tasks simultaneously:
1.  **Geometric Precision:** Exact coordinate regression.
2.  **Global Context:** Coarse classification into topological zones.
3.  **Visual Identity:** Metric learning to distinguish similar-looking locations (aliasing).

### **Architecture**
*   **Backbone:** ResNet50 (pre-trained on ImageNet), fine-tuned with injected **Spatial Dropout ($p=0.1$)** layers to prevent texture overfitting [1].
*   **Heads:** The feature vector (2048-dim) branches into three parallel heads:
    *   `Regressor`: FC layers $\to$ $(x, y)$ coordinates.
    *   `Classifier`: FC layers $\to$ 300 "Smart Zones" (generated via K-Means) [2].
    *   `Embedding`: FC layers $\to$ 256-dim unit vector for Triplet Learning [2].

### **Training Strategy**
*   **Dataset:** ~4,000 images collected using a grid-based "Center + Offset" protocol [3].
*   **Preprocessing:** Manual GPS denoising (correction of sensor drift) and Smart Zoning via K-Means ($K=300$) [4, 5].
*   **Sampling:** Implemented a **Weighted Random Sampler** to balance rare locations and Day/Night cycles [6].
*   **Augmentations:** Adaptive Dual-View Sampling including Random Erasing (occlusions), Color Jitter (lighting), and Geometric Warping [7, 8].
*   **Hard Negative Mining:** For the Triplet Loss, negatives are selected dynamically based on visual similarity but geographic distance [9].

### **Loss Function**
The model minimizes a weighted multi-task objective [10]:
$$ L_{total} = 200 \cdot L_{MSE} + 0.3 \cdot L_{CE} + 10 \cdot L_{Triplet} $$

*   **$L_{MSE}$ (Regression):** Penalizes geometric distance errors.
*   **$L_{CE}$ (Classification):** Prevents "mean location collapse" by enforcing zone commitment.
*   **$L_{Triplet}$ (Metric Learning):** Pushes visually similar but distinct locations apart (Margin=1.5).

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

#### 🟢 Option A: Preprocessed Data (Ready to Train)
This version includes the processed CSV with "Smart Zones" and corrected GPS labels.

* **Link:** [INSERT_LINK_TO_PROCESSED_DATA_ZIP]
* **Action:** 1. Download and unzip.
    2. Place the `images` folder inside `data/`.
    3. Place the `gt.csv` file inside `data/`.
    4. **Status:** You can run `python train.py` immediately.

#### 🟠 Option B: Raw Data (Submission Format)
This is the original dataset format as required by the submission guidelines.

* **Link:** [INSERT_LINK_TO_RAW_DATA_ZIP]
* **Action:**
    1. Download and unzip to `data/raw/`.
    2. **Preprocessing Required:** You must run the preprocessing script to generate the smart zones and handle GPS noise before training.
    ```bash
    python preprocess.py  # Generates the compatible gt.csv
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

# 4. Sanity Check (Optional)
python -c "import utm; import pillow_heif; print('✅ Setup Complete!')"
```
---
## 🧠 The "Trinity" ArchitectureThe model 
processes a 224x224 image through a ResNet50 backbone injected with Spatial Dropout layers, branching into three task-specific heads:
Regression Head (MSE): Predicts the precise $(x, y)$ coordinates.
Classification Head (Cross-Entropy): Classifies the image into one of 300 Smart Zones (generated via K-Means) to provide global context.
Embedding Head (Triplet Loss): Learns a metric space where visually similar but geographically distant locations (aliasing) are pushed apart using Hard Negative Mining.

### 🚀 How to Run
#### 1. Training To train the model from scratch (ensure you followed "Data Setup Option A"):
``` bash train.py
```
Note: The script automatically handles weighted sampling, data loading, and validation checks.
#### 2. Inference (Evaluation)We provide a standalone function predict_gps that accepts a numpy array image and returns coordinates.

Example usage (Python):Pythonimport numpy as np
from PIL import Image
from predict import predict_gps

### Load an image
img_path = "data/images/some_campus_image.jpg"
image = np.array(Image.open(img_path))

### Predict
Returns: np.array([latitude, longitude], dtype=float32)
coords = predict_gps(image)

print(f"Predicted Location: {coords}")

---

Quick Sanity Check:Run our built-in test script to verify the model and environment:Bashpython check_submission.py
📊 Results & AnalysisBest Validation Error: 8.68m (Epoch 127).Robustness: The model successfully handles night scenes and visual aliasing thanks to the Hard Negative Mining strategy.Error Distribution Map Blue dots: Ground Truth | Gray dots: Predictions | Red lines: Error vectors.(See full report for visualization)
