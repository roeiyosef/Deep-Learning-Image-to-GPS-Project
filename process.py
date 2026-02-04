import os
import csv
import utm
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps
from sklearn.cluster import KMeans

# Optional HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# -------------------------
# Config & Paths
# -------------------------
INPUT_DIR = "."  # The folder containing all your photos
PROCESSED_DIR = "../train_data_processed_all"
CSV_PATH = "train_data_processed_all.csv"
IMG_SIZE = (224, 224)
N_CLUSTERS = 300

# EXIF Tags
EXIF_DATE_TAG = 36867 
GPS_IFD_TAG = 34853

os.makedirs(PROCESSED_DIR, exist_ok=True)

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
# Phase 1: Metadata Extraction
# -------------------------
raw_entries = []
print(f"🔎 Scanning folder: {INPUT_DIR}")

valid_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))]

for fn in tqdm(valid_files, desc="Extracting Metadata"):
    full_path = os.path.join(INPUT_DIR, fn)
    try:
        with Image.open(full_path) as img:
            dt_str, lat, lon = get_exif_data(img)
            
            if lat is None or lon is None or lat == 0.0:
                continue # Skip images with no GPS ground truth

            # --- Logic: Night/Day via Hour ---
            # Format: 'YYYY:MM:DD HH:MM:SS'
            is_night = 0
            if dt_str:
                hour = int(dt_str.split(' ')[1].split(':')[0])
                if hour >= 17:
                    is_night = 1
            
            # --- UTM Projection ---
            utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)
            
            # --- Image Processing ---
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
            
            out_name = f"proc_{fn.split('.')[0]}.jpg"
            img.save(os.path.join(PROCESSED_DIR, out_name), quality=95)
            
            raw_entries.append({
                "image_name": out_name,
                "Latitude": lat,
                "Longitude": lon,
                "utm_x": utm_x,
                "utm_y": utm_y,
                "is_night": is_night
            })
    except Exception as e:
        print(f"Error {fn}: {e}")

# -------------------------
# Phase 2: K-Means Clustering
# -------------------------
df = pd.DataFrame(raw_entries)

if len(df) >= N_CLUSTERS:
    print(f"🧩 Generating {N_CLUSTERS} Smart Zones (K-Means)...")
    coords = df[['utm_x', 'utm_y']].values
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    df['label'] = kmeans.fit_predict(coords)
else:
    print("⚠️ Not enough images for 300 clusters. Setting label to 0.")
    df['label'] = 0

# -------------------------
# Final Output
# -------------------------
df.to_csv(CSV_PATH, index=False)

print("\n========== FINAL SUMMARY ==========")
print(f"✅ Total Processed: {len(df)}")
print(f"🌙 Night Photos (>= 17:00): {df['is_night'].sum()}")
print(f"☀️ Day Photos (< 17:00): {len(df) - df['is_night'].sum()}")
print(f"📄 CSV saved: {CSV_PATH}")
print(f"📂 Images resized and saved in: {PROCESSED_DIR}")