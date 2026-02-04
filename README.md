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

Our solution implements a novel **"Multi-Head Trinity Architecture"** based on a fine-tuned ResNet50 backbone. By simultaneously optimizing for **Coordinate Regression**, **Coarse Classification** (Smart Zones), and **Metric Learning** (Triplet Loss), we achieved a state-of-the-art mean localization error of **8.68m**.

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

**Final Directory Structure:**
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


🛠️ Environment SetupTo replicate our results, please strictly follow these steps to create a clean Conda environment with the required dependencies (including utm and pillow-heif).Bash# 1. Create a clean environment with Python 3.9
conda create -n gps_project python=3.9 -y

# 2. Activate the environment
conda activate gps_project

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Sanity Check (Optional)
python -c "import utm; import pillow_heif; print('✅ Setup Complete!')"
🧠 The "Trinity" ArchitectureThe model processes a 224x224 image through a ResNet50 backbone injected with Spatial Dropout layers, branching into three task-specific heads:Regression Head (MSE): Predicts the precise $(x, y)$ coordinates.Classification Head (Cross-Entropy): Classifies the image into one of 300 Smart Zones (generated via K-Means) to provide global context.Embedding Head (Triplet Loss): Learns a metric space where visually similar but geographically distant locations (aliasing) are pushed apart using Hard Negative Mining.🚀 How to Run1. TrainingTo train the model from scratch (ensure you followed "Data Setup Option A"):Bashpython train.py
Note: The script automatically handles weighted sampling, data loading, and validation checks.2. Inference (Evaluation)We provide a standalone function predict_gps that accepts a numpy array image and returns coordinates.Example usage (Python):Pythonimport numpy as np
from PIL import Image
from predict import predict_gps

# Load an image
img_path = "data/images/some_campus_image.jpg"
image = np.array(Image.open(img_path))

# Predict
# Returns: np.array([latitude, longitude], dtype=float32)
coords = predict_gps(image)

print(f"Predicted Location: {coords}")
Quick Sanity Check:Run our built-in test script to verify the model and environment:Bashpython check_submission.py
📊 Results & AnalysisBest Validation Error: 8.68m (Epoch 127).Robustness: The model successfully handles night scenes and visual aliasing thanks to the Hard Negative Mining strategy.Error Distribution Map Blue dots: Ground Truth | Gray dots: Predictions | Red lines: Error vectors.(See full report for visualization)
### מה הוספתי ושיניתי (הסבר קצר):

1.  **Data & Model Setup (הכי חשוב):** הוספתי סקציה ענקית וברורה בהתחלה שמחלקת את ההורדה ל-3 חלקים:
    * **מודל:** הבהרתי שחייבים להוריד אותו ידנית לתיקייה הראשית.
    * **Option A (המומלץ):** מורידים, שמים בתיקייה ורצים.
    * **Option B (הגלם/Raw):** הסברתי שאם בוחרים בזה, **חייבים** להריץ `python preprocess.py` קודם (הנחתי שיש לך סקריפט כזה או שאתה יכול לייצר אחד, אם אין לך סקריפט כזה, תגיד לי ואני אכתוב לך אותו בשנייה, או שתמחק את השורה הזו).
2.  **מבנה התיקיות:** הוספתי ציור של עץ התיקיות (`tree`) כדי שהבודק יראה בדיוק איפה כל קובץ צריך לשבת.
3.  **לינקים:** השארתי לך מקום (`[INSERT_LINK...]`) לשים את הלינקים האמיתיים שלך.
