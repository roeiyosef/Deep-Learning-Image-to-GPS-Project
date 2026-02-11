import os
import csv
import utm
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps
from sklearn.cluster import KMeans

# Attempt to register HEIF opener for iPhone photos
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# -------------------------
# Configuration (Updated)
# -------------------------
# Get the directory where this script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: DATA_SET_RAW/images
INPUT_DIR = os.path.join(BASE_DIR, "DATA_SET_RAW", "images")

# Output: DATA_SET_PROCESSED
OUTPUT_ROOT = os.path.join(BASE_DIR, "DATA_SET_PROCESSED")
PROCESSED_IMG_DIR = os.path.join(OUTPUT_ROOT, "images")
CSV_PATH = os.path.join(OUTPUT_ROOT, "gt.csv")

IMG_SIZE = (224, 224)
N_CLUSTERS = 300

# EXIF Tags
EXIF_DATE_TAG = 36867 
GPS_IFD_TAG = 34853

# Create the output directories if they don't exist
os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)

# -------------------------
# Helper Functions
# -------------------------
def to_decimal(coords, ref):
    if not coords: return 0.0
    try:
        d = float(coords[0])
        m = float(coords[1])
        s = float(coords[2])
        sign = -1 if ref in ['S', 'W'] else 1
        return (d + m / 60.0 + s / 3600.0) * sign
    except: return 0.0

def get_exif_data(img):
    try:
        exif = img._getexif()
        if not exif: return None, None, None
        
        # 1. Date Time
        dt_str = exif.get(EXIF_DATE_TAG)
        
        # 2. GPS
        gps_info = exif.get(GPS_IFD_TAG)
        lat, lon = None, None
        if gps_info:
            lat = to_decimal(gps_info.get(2), gps_info.get(1))
            lon = to_decimal(gps_info.get(4), gps_info.get(3))
            
        return dt_str, lat, lon
    except:
        return None, None, None

# -------------------------
# Phase 1: Metadata Extraction & Image Processing
# -------------------------
raw_entries = []
print(f"🔎 Scanning folder: {INPUT_DIR}")

if not os.path.exists(INPUT_DIR):
    print(f"❌ Error: Input directory not found: {INPUT_DIR}")
    exit(1)

valid_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]

print(f"📸 Found {len(valid_files)} images. Starting processing...")

for fn in tqdm(valid_files, desc="Processing Images"):
    full_path = os.path.join(INPUT_DIR, fn)
    try:
        with Image.open(full_path) as img:
            dt_str, lat, lon = get_exif_data(img)
            
            # Skip images with no GPS data
            if lat is None or lon is None or lat == 0.0:
                continue 

            # --- Logic: Night/Day via Hour ---
            # Format: 'YYYY:MM:DD HH:MM:SS'
            is_night = 0
            if dt_str:
                try:
                    hour = int(dt_str.split(' ')[1].split(':')[0])
                    if hour >= 17:
                        is_night = 1
                except:
                    pass # Keep 0 if date parsing fails
            
            # --- UTM Projection ---
            utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)
            
            # --- Image Processing (Resize & Save) ---
            img = ImageOps.exif_transpose(img) # Fix rotation
            img = img.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
            
            # Rename file to be safe
            out_name = f"proc_{fn.split('.')[0]}.jpg"
            save_path = os.path.join(PROCESSED_IMG_DIR, out_name)
            
            img.save(save_path, quality=95)
            
            raw_entries.append({
                "filename": out_name, # Changed from "image_name" to "filename" to match typical train.py
                "Latitude": lat,
                "Longitude": lon,
                "utm_x": utm_x,
                "utm_y": utm_y,
                "label": 0, # Placeholder, will be updated by K-Means
                "is_night": is_night
            })
    except Exception as e:
        print(f"Error {fn}: {e}")

# -------------------------
# Phase 2: K-Means Clustering
# -------------------------
df = pd.DataFrame(raw_entries)

if not df.empty:
    if len(df) >= N_CLUSTERS:
        print(f"🧩 Generating {N_CLUSTERS} Smart Zones (K-Means)...")
        coords = df[['utm_x', 'utm_y']].values
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
        df['label'] = kmeans.fit_predict(coords)
    else:
        print(f"⚠️ Not enough images for {N_CLUSTERS} clusters. Setting all labels to 0.")
        df['label'] = 0

    # -------------------------
    # Final Output
    # -------------------------
    df.to_csv(CSV_PATH, index=False)

    print("\n========== FINAL SUMMARY ==========")
    print(f"✅ Total Processed: {len(df)}")
    print(f"🌙 Night Photos (>= 17:00): {df['is_night'].sum()}")
    print(f"☀️ Day Photos (< 17:00): {len(df) - df['is_night'].sum()}")
    print(f"📄 CSV saved at: {CSV_PATH}")
    print(f"📂 Processed images saved at: {PROCESSED_IMG_DIR}")

else:
    print("❌ No valid images with GPS found.")