import torch
import numpy as np
import utm
from PIL import Image
from torchvision import transforms
from model import VPSModel 

CENTER_X = 671669.6027
CENTER_Y = 3460032.8143
GPS_SCALE = 100.0

CAMPUS_MEAN = [0.42974764108657837, 0.41656675934791565, 0.37728580832481384]
CAMPUS_STD = [0.1985701620578766, 0.19456541538238525, 0.2009020745754242]

def predict_gps(image: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VPSModel(num_classes=300)
    try:
        checkpoint = torch.load('best_model.pth', map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print("Error: 'best_model.pth' not found.")
        return np.array([0.0, 0.0], dtype=np.float32)

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CAMPUS_MEAN, std=CAMPUS_STD)
    ])
    
    if isinstance(image, np.ndarray):
        input_tensor = transform(image).unsqueeze(0).to(device)
    else:
        print("Error: Input is not a numpy array")
        return np.array([0.0, 0.0], dtype=np.float32)

    with torch.no_grad():
        pred_norm, _, _ = model(input_tensor)

    pred_flat = pred_norm.detach().cpu().numpy().flatten()


    norm_x = float(pred_flat[0]) 
    norm_y = float(pred_flat[1])

    utm_x = (norm_x * GPS_SCALE) + CENTER_X
    utm_y = (norm_y * GPS_SCALE) + CENTER_Y

    try:
        lat, lon = utm.to_latlon(utm_x, utm_y, 36, 'R')
    except Exception as e:
        print(f"Error converting UTM to LatLon: {e}")
        return np.array([0.0, 0.0], dtype=np.float32)

    return np.array([lat, lon], dtype=np.float32)