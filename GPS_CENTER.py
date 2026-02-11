import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader
from torchvision import transforms

# GPS Statistics & Center Calculation

def calculate_gps_center(csv_path):
    """
    Calculates the center (mean) of UTM coordinates from the CSV.
    Returns: center_x, center_y
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ Warning: File {csv_path} not found.")
        return 0, 0

    df = pd.read_csv(csv_path)

    center_x = df['utm_x'].mean()
    center_y = df['utm_y'].mean()

    min_x, max_x = df['utm_x'].min(), df['utm_x'].max()
    min_y, max_y = df['utm_y'].min(), df['utm_y'].max()

    range_x = max_x - min_x
    range_y = max_y - min_y

    std_x = df['utm_x'].std()
    std_y = df['utm_y'].std()
    
    unique_coords = len(df.groupby(['utm_x', 'utm_y']))

    print("-" * 40)
    print("GPS STATISTICS")
    print("-" * 40)
    print(f"Total Samples: {len(df)}")
    print(f"Unique Coordinates: {unique_coords}")
    print(f"\nCENTER (Mean):")
    print(f"  X: {center_x:.4f}")
    print(f"  Y: {center_y:.4f}")
    print(f"\nRANGE:")
    print(f"  X: {min_x:.2f} to {max_x:.2f} (Span: {range_x:.2f}m)")
    print(f"  Y: {min_y:.2f} to {max_y:.2f} (Span: {range_y:.2f}m)")
    print(f"\nSTD DEV:")
    print(f"  X: {std_x:.2f}m")
    print(f"  Y: {std_y:.2f}m")
    print("-" * 40)

    return center_x, center_y





if __name__ == "__main__":
    CSV_PATH = 'data/gt.csv'
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        exit()

    print("\n" + "="*40)
    print("STEP 1: GPS Statistics")
    print("="*40)
    center_x, center_y = calculate_gps_center(CSV_PATH)
    GPS_CENTER = [center_x, center_y]
