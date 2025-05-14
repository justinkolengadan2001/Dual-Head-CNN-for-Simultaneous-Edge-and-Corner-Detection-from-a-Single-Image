import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
import os
from tqdm import tqdm

def compute_iou(pred, target, threshold = 0.5):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection + 1e-6
    return (intersection / union).item()

def compute_f1(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return (2 * precision * recall / (precision + recall + 1e-6)).item()

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    iou_edge = iou_corner = f1_edge = f1_corner = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            for i in range(images.size(0)):
                pred, ground_truth = outputs[i], labels[i]
                iou_edge += compute_iou(pred[0], ground_truth[0])
                iou_corner += compute_iou(pred[1], ground_truth[1])
                f1_edge += compute_f1(pred[0], ground_truth[0])
                f1_corner += compute_f1(pred[1], ground_truth[1])
                count += 1
    return {
        "val_loss": total_loss / len(dataloader),
        "iou_edge": iou_edge / count,
        "iou_corner": iou_corner / count,
        "f1_edge": f1_edge / count,
        "f1_corner": f1_corner / count,
    }

def plot_metrics(history, model_name):
    epochs = range(1, len(history["val_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["val_loss"], label="Val Loss", linestyle='--', marker='o')
    plt.plot(epochs, history["train_loss"], label="Train Loss", linestyle='-', marker='x')
    plt.title(f"{model_name} Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["iou_edge"], label="IoU Edge", linestyle='-', marker='D')
    plt.plot(epochs, history["iou_corner"], label="IoU Corner", linestyle='--', marker='s')
    plt.plot(epochs, history["f1_edge"], label="F1 Edge", linestyle='-.', marker='^')
    plt.plot(epochs, history["f1_corner"], label="F1 Corner", linestyle=':', marker='v')
    plt.title(f"{model_name} Metrics")

    for ax in plt.gcf().axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle=':', linewidth = 0.5)
        ax.legend()
    
    plt.tight_layout()
    plt.grid(True)
    os.makedirs("../plots", exist_ok=True)
    plt.savefig(f"../plots/{model_name.lower()}_metrics.png")
    plt.show()

def export_metrics_csv(history, model_name):
    os.makedirs("../metrics", exist_ok = True)
    file_path = f"../metrics/{model_name.lower()}_metrics.csv"
    
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "IoU Edge", "IoU Corner", "F1 Edge", "F1 Corner"])
        for i in range(len(history["val_loss"])):
            writer.writerow([
                i+1,
                history["train_loss"][i], history["val_loss"][i],
                history["iou_edge"][i], history["iou_corner"][i],
                history["f1_edge"][i], history["f1_corner"][i]
            ])

def compare_predictions(model1, model2, dataloader, device):
    model1.eval(); model2.eval()
    imgs, labels = next(iter(dataloader))
    imgs = imgs.to(device)
    
    with torch.no_grad():
        preds1 = model1(imgs)
        preds2 = model2(imgs)
        
    for i in range(1):
        img = imgs[i].cpu().permute(1, 2, 0).numpy()
        edge1 = preds1[i, 0].cpu().numpy()
        corner1 = preds1[i, 1].cpu().numpy()
        edge2 = preds2[i, 0].cpu().numpy()
        corner2 = preds2[i, 1].cpu().numpy()
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes[0, 0].imshow(img)
        axes[0, 1].imshow(edge1, cmap='gray')
        axes[0, 2].imshow(corner1, cmap='gray')
        axes[1, 1].imshow(edge2, cmap='gray')
        axes[1, 2].imshow(corner2, cmap='gray')
        
        for ax in axes.flat: ax.axis("off")
        axes[0, 0].set_title("Input")
        axes[0, 1].set_title("ResNet Edge")
        axes[0, 2].set_title("ResNet Corner")
        axes[1, 1].set_title("UNet Edge")
        axes[1, 2].set_title("UNet Corner")
        
        plt.tight_layout()
        plt.show()
