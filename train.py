
import os
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from model import VPSModel  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_DIR = 'data/images'      
CSV_PATH = 'data/gt.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = 'results' 

CENTER_X = 671669.6027
CENTER_Y = 3460032.8143
GPS_SCALE = 100.0
CAMPUS_MEAN = [0.42974764108657837, 0.41656675934791565, 0.37728580832481384]
CAMPUS_STD = [0.1985701620578766, 0.19456541538238525, 0.2009020745754242]

BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 0.05
EPOCHS = 200
PATIENCE = 15

ALPHA = 200.0 # Regression
BETA = 0.3    # Classification
GAMMA = 10.0  # Triplet




torch.manual_seed(0)
np.random.seed(0)

#This is what w used to calculate the Campus Mean and Std
def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images
    return mean.tolist(), std.tolist()



class DualViewDataset(Dataset):
    def __init__(self, dataframe, weak_transform, strong_transform=None, is_validation=False):
        self.data = dataframe.reset_index(drop=True)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.is_validation = is_validation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(DATA_DIR, row['image_name'])
        
        
        img = Image.open(img_path).convert('RGB')
       

        gps = torch.tensor([
            (row['utm_x'] - CENTER_X) / GPS_SCALE,
            (row['utm_y'] - CENTER_Y) / GPS_SCALE
        ], dtype=torch.float32)

        label = torch.tensor(row['label'], dtype=torch.long)

        if self.is_validation:
            img_clean = self.weak_transform(img)
            return img_clean, gps

        is_night = (row['is_night'] == 1)
        img_anc = self.weak_transform(img)

        if is_night:
            img_pos = self.weak_transform(img)
        elif self.strong_transform:
            img_pos = self.strong_transform(img)
        else:
            img_pos = img_anc

        return img_anc, img_pos, gps, label

def get_hard_negatives(embeddings, gps, safe_radius=0.2): 
    dist_emb = torch.cdist(embeddings, embeddings, p=2)
    dist_gps = torch.cdist(gps, gps, p=2)
    
    valid_mask = (dist_gps > safe_radius)
    
    dist_emb_masked = dist_emb.clone()
    dist_emb_masked[~valid_mask] = float('inf')
    
    hard_neg_indices = dist_emb_masked.argmin(dim=1)
    
    return embeddings[hard_neg_indices]

def get_transforms():
    regular = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CAMPUS_MEAN, std=CAMPUS_STD)
    ])

    weak = transforms.Compose([
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=CAMPUS_MEAN, std=CAMPUS_STD)
])
    
    strong = transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=CAMPUS_MEAN, std=CAMPUS_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2),value="random") 
    ])
    
    return regular,weak, strong

def evaluate_and_plot_localization_error(model, val_loader):
    """
    Runs inference, denormalizes coordinates, calculates errors, 
    plots results AND saves the plot to an image file.
    """
    model.eval()
    preds_list = []
    actuals_list = []
    GPS_CENTER = np.array([CENTER_X,CENTER_Y])


    print("Running inference on validation set:")

    with torch.no_grad():
        for imgs, gps_targets in tqdm(val_loader, desc="Inference"):
            imgs = imgs.to(DEVICE)
            
            preds = model(imgs)
            
            if isinstance(preds, tuple):
                preds = preds[0]

            preds_list.append(preds.cpu().numpy())
            actuals_list.append(gps_targets.numpy())

    all_preds = np.concatenate(preds_list, axis=0)
    all_actuals = np.concatenate(actuals_list, axis=0)

    all_preds_denorm = (all_preds * GPS_SCALE) + GPS_CENTER
    all_actuals_denorm = (all_actuals * GPS_SCALE) + GPS_CENTER

    errors = np.linalg.norm(all_actuals_denorm - all_preds_denorm, axis=1)

    mean_error = np.mean(errors)
    median_error = np.median(errors)
    num_green = np.sum(errors < 5.0)
    num_orange = np.sum((errors >= 5.0) & (errors < 15.0))
    num_red = np.sum(errors >= 15.0)

    print(f"\nEvaluation Complete.")
    print(f"   Mean Error:   {mean_error:.2f}m")
    print(f"   Median Error: {median_error:.2f}m")

    plt.figure(figsize=(12, 10))

    plt.scatter(all_actuals_denorm[:, 0], all_actuals_denorm[:, 1],
                label='Ground Truth', color='blue', alpha=0.5, s=30, edgecolors='k', zorder=2)
    plt.scatter(all_preds_denorm[:, 0], all_preds_denorm[:, 1],
                label='Predicted', color='gray', alpha=0.5, s=20, zorder=1)

    for i in range(len(all_actuals_denorm)):
        err = errors[i]
        if err < 5.0:
            color = 'green'; lw = 1.0; alpha = 0.4
        elif err < 15.0:
            color = 'orange'; lw = 1.5; alpha = 0.6
        else:
            color = 'red'; lw = 2.0; alpha = 0.8

        plt.plot([all_actuals_denorm[i, 0], all_preds_denorm[i, 0]],
                 [all_actuals_denorm[i, 1], all_preds_denorm[i, 1]],
                 color=color, linewidth=lw, alpha=alpha, zorder=1)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Ground Truth',
               markerfacecolor='blue', markersize=10, markeredgecolor='k', alpha=0.6),
        Line2D([0], [0], marker='o', color='w', label='Predicted',
               markerfacecolor='gray', markersize=8, alpha=0.6),
        Line2D([0], [0], color='none', label=' '),
        Line2D([0], [0], color='green', lw=2, label=f'Good (< 5m): {num_green}'),
        Line2D([0], [0], color='orange', lw=2, label=f'Fair (5-15m): {num_orange}'),
        Line2D([0], [0], color='red', lw=2, label=f'Poor (> 15m): {num_red}')
    ]

    plt.legend(handles=legend_elements, loc='upper right', framealpha=0.95, fontsize=11, title="Legend")
    plt.title(f'Localization Error Analysis\nMean Error: {mean_error:.2f}m', fontsize=14, fontweight='bold')
    plt.xlabel('UTM Easting (X)')
    plt.ylabel('UTM Northing (Y)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    
    print(f"Saving evaluation plot to: localization_error.png")
    plt.savefig("localization_error.png", dpi=300) 
    
    plt.show()
    plt.close()

def plot_distance_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_dist'], label='Train Avg Distance (m)', color='blue', linestyle='--')
    plt.plot(history['val_dist'], label='Val Avg Distance (m)', color='red', linewidth=2)

    plt.title('Average Distance: Train vs Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Distance (Meters)')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=10, color='green', linestyle=':', label='10m Target Threshold')
    plt.savefig("training_plot.png")
    plt.close()
    print(f"Training plot saved!")
    
def train():
    print(f"Starting Training on {DEVICE}...")
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}. Please download 'Preprocessed Data' from Drive.")
        
    full_df = pd.read_csv(CSV_PATH)
    
    
    train_df, val_df = train_test_split(
    full_df,
    test_size=0.2,
    random_state=42,
    stratify=full_df['is_night']
    )    
    
    train_df['group_id'] = train_df['label'].astype(str) + "_" + train_df['is_night'].astype(str)

    group_counts = train_df['group_id'].value_counts()
    group_weights = 1.0 / group_counts

    sample_weights = train_df['group_id'].map(group_weights).values

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_df),
        replacement=True
    )

    print(f"Data Ready! Train: {len(train_df)}, Val: {len(val_df)}")
    
    #Data Loaders
    regular_transform,weak_transform, train_transform = get_transforms()
    
    train_ds = DualViewDataset(train_df,weak_transform,train_transform,is_validation=False)
    val_ds = DualViewDataset(val_df,regular_transform,strong_transform=None,is_validation=True)

    train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2)
   

    #######################################################################################
    #SHOULD YOU CHOOSE TO RECALCULATE THE CAMPUS MEAN AND STD , THIS IS WHERE YOU DO IT.
    # temp_ds = DualViewDataset(train_df, weak_transform=transforms.Compose([
    # transforms.ToTensor()]), is_validation=True)
    # temp_loader = DataLoader(temp_ds, batch_size=64, shuffle=False)

    # campus_mean, campus_std = get_mean_and_std(temp_loader)
    # print(f"Calculated Mean: {campus_mean}")
    # print(f"Calculated Std: {campus_std}")
    ########################################################################################


    model = VPSModel(num_classes=300).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    criterion_mse = nn.MSELoss()
    criterion_ce =  nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_triplet = nn.TripletMarginLoss(margin=1.5, p=2)
    
    history = {'train_dist': [], 'val_dist': []}


    #TRAIN LOOP
    best_val_dist = float('inf')
    patience_counter = 0
    

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        train_dist_sum = 0
        epoch_correct = 0
        epoch_total = 0
        epoch_loss = 0.0


        loop = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=True, ncols=100, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for img_anc, img_pos, gps,label in loop:
            img_anc, img_pos, gps,label= img_anc.to(DEVICE), img_pos.to(DEVICE), gps.to(DEVICE),label.to(DEVICE)

            optimizer.zero_grad()

            pred_gps_anc, emb_anc, pred_cls_anc = model(img_anc)
            pred_gps_pos, emb_pos, pred_cls_pos = model(img_pos)

            emb_neg = get_hard_negatives(emb_anc, gps, safe_radius=20.0/100.0)
            
            loss_mse = (0.3 * criterion_mse(pred_gps_anc, gps) + 0.7 * criterion_mse(pred_gps_pos, gps))
            
            loss_triplet = criterion_triplet(emb_anc, emb_pos, emb_neg)
            
            loss_cls =(0.3 * criterion_ce(pred_cls_anc, label) + 0.7 *  criterion_ce(pred_cls_pos, label))
            
            total_loss = ALPHA*loss_mse  + (BETA* loss_cls) + (GAMMA * loss_triplet)
            
            total_loss.backward()
            optimizer.step()

            batch_size = len(img_anc)
            epoch_loss += total_loss.item() * batch_size


            dist_error = torch.norm((pred_gps_anc - gps) * GPS_SCALE, dim=1)
            train_dist_sum += dist_error.sum().item()
            epoch_correct += (dist_error < 10.0).sum().item() 
            epoch_total += batch_size

            loop.set_postfix(
                MSE=f"{loss_mse.item():.4f}",
                Tri=f"{loss_triplet.item():.2f}",
                Cls=f"{loss_cls.item():.2f}",
                Acc=f"{100*epoch_correct/epoch_total:.1f}%"
            )

        avg_loss = epoch_loss / epoch_total
        avg_train_dist = train_dist_sum / epoch_total
        train_acc = 100 * epoch_correct / epoch_total

        # Validation
        model.eval()
        val_dist_sum = 0.0
        val_correct = 0
        val_total = 0
        val_loss_sum = 0

        with torch.no_grad():
            for img, gps in val_loader:
                img, gps = img.to(DEVICE), gps.to(DEVICE)
                pred, _ ,_= model(img)

                v_loss = criterion_mse(pred, gps)
                val_loss_sum += v_loss.item() * len(img)

                dist = torch.norm((pred - gps) * GPS_SCALE, dim=1)
                val_dist_sum += dist.sum().item()

                val_correct += (dist < 10.0).sum().item()
                val_total += len(img)



        val_acc = 100 * val_correct / val_total
        val_avg_loss = val_loss_sum / val_total
        avg_val_dist = val_dist_sum / val_total

        history['train_dist'].append(avg_train_dist)
        history['val_dist'].append(avg_val_dist)

        print("-" * 60)
        print(f"Epoch {epoch+1} Summary:")
        print(f"   TRAIN | Avg Distance: {avg_train_dist:.2f}m | Acc (<10m): {train_acc:.1f}%")
        print(f"   VAL   | Avg Distance: {avg_val_dist:.2f}m | Acc (<10m): {val_acc:.1f}%")
        print("-" * 60)

        if avg_val_dist < best_val_dist:
            best_val_dist = avg_val_dist
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New Best Model Saved! ({best_val_dist:.2f}m)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training Finished")
    print(f"The Best Model is with Validation avg distance of ({best_val_dist:.2f}m)")
    plot_distance_history(history)
    print("Loading Best Model weights for evaluation")
    model.load_state_dict(torch.load("best_model.pth"))
    evaluate_and_plot_localization_error(model, val_loader)

if __name__ == "__main__":
    train()
