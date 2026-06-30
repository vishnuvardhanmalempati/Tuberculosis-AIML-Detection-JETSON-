import os
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_loader import get_dataloaders
from src.model import get_tuberculosis_model

def plot_training_history(history, save_path):
    """
    Plots training & validation loss and accuracy.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, save_path):
    """
    Plots confusion matrix using Seaborn.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Tuberculosis'], 
                yticklabels=['Normal', 'Tuberculosis'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, auc_score, save_path):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(model_name="mobilenet_v3", epochs=15, batch_size=16, lr=1e-4, data_dir="data", save_dir="models", device=None):
    """
    Runs the training pipeline and evaluates the best model on the test set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load data
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(data_dir, batch_size)
    
    # 2. Instantiate model with frozen backbone for transfer learning
    model = get_tuberculosis_model(model_name, pretrained=True, freeze_backbone=True)
    model = model.to(device)
    
    # 3. Loss, optimizer, and scheduler
    # We use BCEWithLogitsLoss because model outputs raw logits
    criterion = nn.BCEWithLogitsLoss()
    # Only optimize parameters that require gradients
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Directory to save models and plots
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / f"best_tb_model_{model_name}.pth"
    
    # Track training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    early_stop_patience = 15
    epochs_no_improve = 0
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_corrects = 0
        total_train_samples = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1) # shape: [B, 1]
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            # Compute accuracy
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            train_corrects += torch.sum(preds == labels).item()
            total_train_samples += images.size(0)
            
        epoch_train_loss = train_loss / total_train_samples
        epoch_train_acc = train_corrects / total_train_samples
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                val_corrects += torch.sum(preds == labels).item()
                total_val_samples += images.size(0)
                
        epoch_val_loss = val_loss / total_val_samples
        epoch_val_acc = val_corrects / total_val_samples
        
        # Step the learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
              
        # Checkpoint based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved new best model weights to {best_model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    print("Training finished. Loading best weights for final evaluation...")
    
    # 4. Final Evaluation on Test Set
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs)
            
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    y_preds = (y_probs >= 0.5).astype(float)
    
    # Compute metrics
    acc = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds, zero_division=0)
    recall = recall_score(y_true, y_preds, zero_division=0) # Recall = Sensitivity
    f1 = f1_score(y_true, y_preds, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    
    # Compute Specificity
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "model_name": model_name,
        "test_loss": test_loss / len(y_true),
        "accuracy": acc,
        "precision": precision,
        "sensitivity_recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "auc_roc": auc
    }
    
    # Print metrics
    print("\n================== TEST SET PERFORMANCE ==================")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")
    print("==========================================================\n")
    
    # Save metrics JSON
    with open(save_dir / f"metrics_{model_name}.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save plots
    plot_training_history(history, save_dir / f"training_history_{model_name}.png")
    plot_confusion_matrix(cm, save_dir / f"confusion_matrix_{model_name}.png")
    plot_roc_curve(y_true, y_probs, auc, save_dir / f"roc_curve_{model_name}.png")
    
    print("Generated all plots and metrics successfully.")
    return metrics

if __name__ == "__main__":
    # Test script with 2 epochs
    train_model(model_name="mobilenet_v3", epochs=2, batch_size=8)
