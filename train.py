import torch
import torch.nn as nn
from model import ResNetLiteDualHeadSkip, UNetDualHead
from dataset import EdgeCornerDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from train_utils import train_one_epoch, validate, plot_metrics, export_metrics_csv, compare_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4

dataset = EdgeCornerDataset(root_dir="../data/filtered_crops/")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model_1 = ResNetLiteDualHeadSkip().to(device)
model_2 = UNetDualHead().to(device)

loss_fn = nn.BCELoss()
optimizer_1 = torch.optim.AdamW(model_1.parameters(), lr = 1e-3)
scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size = 10, gamma = 0.5)

optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr = 1e-3)
scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size = 10, gamma = 0.5)

resnet_history = {key: [] for key in ["train_loss", "val_loss", "iou_edge", "iou_corner", "f1_edge", "f1_corner"]}
unet_history = {key: [] for key in ["train_loss", "val_loss", "iou_edge", "iou_corner", "f1_edge", "f1_corner"]}

# ===================== TRAINING - ResNet-18 Backbone Skip Model =====================
best_val_loss = float("inf")
os.makedirs("../models", exist_ok = True)
epochs = 100

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs} [ResNetLiteDualHeadSkip]")
    train_loss = train_one_epoch(model_1, train_loader, loss_fn, optimizer_1, device)
    val_metrics = validate(model_1, val_loader, loss_fn, device)
    scheduler_1.step()

    for key in resnet_history:
        resnet_history[key].append(val_metrics.get(key, train_loss if key == "train_loss" else None))
    resnet_history["train_loss"][epoch] = train_loss

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | IoU(E): {val_metrics['iou_edge']:.3f} | F1(E): {val_metrics['f1_edge']:.3f} | IoU(C): {val_metrics['iou_corner']:.3f} | F1(C): {val_metrics['f1_corner']:.3f}")

    if val_metrics["val_loss"] < best_val_loss:
        best_val_loss = val_metrics["val_loss"]
        torch.save(model_1.state_dict(), "../models/resnet_skip_best.pth")
        print("Saved new best ResNet model")

# ===================== TRAINING - UNET Double Head Model =====================
best_val_loss = float("inf")
epochs = 35

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs} [UNetDualHead]")
    train_loss = train_one_epoch(model_2, train_loader, loss_fn, optimizer_2, device)
    val_metrics = validate(model_2, val_loader, loss_fn, device)
    scheduler_2.step()

    for key in unet_history:
        unet_history[key].append(val_metrics.get(key, train_loss if key == "train_loss" else None))
    unet_history["train_loss"][epoch] = train_loss

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | IoU(E): {val_metrics['iou_edge']:.3f} | F1(E): {val_metrics['f1_edge']:.3f} | IoU(C): {val_metrics['iou_corner']:.3f} | F1(C): {val_metrics['f1_corner']:.3f}")
    if val_metrics["val_loss"] < best_val_loss:
        best_val_loss = val_metrics["val_loss"]
        torch.save(model_2.state_dict(), "../models/unet_dual_head_best.pth")
        print("Saved new best UNet model")

plot_metrics(resnet_history, "ResNetLiteDualHeadSkip")
export_metrics_csv(resnet_history, "ResNetLiteDualHeadSkip")

plot_metrics(unet_history, "UNetDualHead")
export_metrics_csv(unet_history, "UNetDualHead")

compare_predictions(model_1, model_2, val_loader, device)