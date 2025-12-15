import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import time
import os
from collections import Counter

# --- CONFIGURATION ---
DATA_DIR = './gtsrb-german-traffic-sign/Train' 
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
IMG_HEIGHT = 224  # ResNet expects 224x224
IMG_WIDTH = 224
NUM_CLASSES = 43
PATIENCE = 5  # Early stopping patience

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. ENHANCED DATA PIPELINE ---
train_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Simulate camera defocus
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dataloaders(data_dir):
    """Loads data with train/val/test split and handles class imbalance"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    
    # 70/15/15 Train/Val/Test Split
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply different transforms to val/test
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = val_transforms
    
    # Handle class imbalance with weighted sampling
    train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = Counter(train_labels)
    class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    return train_loader, val_loader, test_loader, full_dataset.classes

# --- 2. TRANSFER LEARNING WITH RESNET18 ---
def create_model():
    """Creates ResNet18 with pretrained ImageNet weights"""
    model = models.resnet18(pretrained=True)
    
    # Freeze early layers for faster training
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    
    # Replace final layer for 43 classes
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, NUM_CLASSES)
    )
    
    return model.to(device)

# --- 3. TRAINING WITH EARLY STOPPING & LR SCHEDULING ---
def train_model():
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(DATA_DIR)
    
    # Create ensemble of 3 models
    models_list = [create_model() for _ in range(3)]
    
    histories = []
    
    for model_idx, model in enumerate(models_list):
        print(f"\n{'='*60}")
        print(f"Training Model {model_idx + 1}/3")
        print(f"{'='*60}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(EPOCHS):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch [{epoch+1}/{EPOCHS}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f"traffic_sign_model_{model_idx}.pth")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        print(f"Model {model_idx + 1} finished in {time.time() - start_time:.2f}s")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Load best weights
        model.load_state_dict(torch.load(f"traffic_sign_model_{model_idx}.pth"))
        histories.append(history)
    
    return models_list, histories, test_loader, class_names

# --- 4. EVALUATION & VISUALIZATION ---
def evaluate_ensemble(models_list, test_loader, class_names):
    """Evaluate ensemble of models on test set"""
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION ON TEST SET")
    print("="*60)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for model in models_list:
        model.eval()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Ensemble prediction (average probabilities)
            ensemble_output = torch.zeros(images.size(0), NUM_CLASSES).to(device)
            for model in models_list:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                ensemble_output += probs
            
            ensemble_output /= len(models_list)
            _, predicted = torch.max(ensemble_output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(ensemble_output.cpu().numpy())
    
    # Calculate metrics
    test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(NUM_CLASSES)]))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.title('Confusion Matrix - Traffic Sign Classification', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(15, 6))
    plt.bar(range(NUM_CLASSES), per_class_acc * 100)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14)
    plt.axhline(y=test_acc, color='r', linestyle='--', label=f'Overall Acc: {test_acc:.2f}%')
    plt.legend()
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print("Per-class accuracy plot saved to per_class_accuracy.png")
    
    # Analyze misclassifications
    print("\n" + "="*60)
    print("MOST COMMON MISCLASSIFICATIONS (Safety Analysis)")
    print("="*60)
    
    misclass = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                misclass.append((cm[i, j], i, j))
    
    misclass.sort(reverse=True)
    for count, true_label, pred_label in misclass[:10]:
        print(f"Class {true_label} misclassified as {pred_label}: {count} times")
    
    return test_acc, all_preds, all_labels, all_probs

def plot_training_history(histories):
    """Plot training curves for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, history in enumerate(histories):
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label=f'Model {idx+1} Train', alpha=0.7)
        axes[0, 0].plot(history['val_loss'], label=f'Model {idx+1} Val', alpha=0.7, linestyle='--')
    
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    for idx, history in enumerate(histories):
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label=f'Model {idx+1} Train', alpha=0.7)
        axes[0, 1].plot(history['val_acc'], label=f'Model {idx+1} Val', alpha=0.7, linestyle='--')
    
    axes[0, 1].set_title('Training & Validation Accuracy', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average ensemble performance
    avg_train_loss = np.mean([h['train_loss'] for h in histories], axis=0)
    avg_val_loss = np.mean([h['val_loss'] for h in histories], axis=0)
    avg_train_acc = np.mean([h['train_acc'] for h in histories], axis=0)
    avg_val_acc = np.mean([h['val_acc'] for h in histories], axis=0)
    
    axes[1, 0].plot(avg_train_loss, label='Ensemble Train', linewidth=2)
    axes[1, 0].plot(avg_val_loss, label='Ensemble Val', linewidth=2, linestyle='--')
    axes[1, 0].set_title('Ensemble Average Loss', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(avg_train_acc, label='Ensemble Train', linewidth=2)
    axes[1, 1].plot(avg_val_acc, label='Ensemble Val', linewidth=2, linestyle='--')
    axes[1, 1].set_title('Ensemble Average Accuracy', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history saved to training_history.png")

def benchmark_inference_speed(model, test_loader):
    """Benchmark inference speed on CPU and GPU"""
    print("\n" + "="*60)
    print("INFERENCE SPEED BENCHMARK")
    print("="*60)
    
    model.eval()
    
    # GPU benchmark
    if torch.cuda.is_available():
        model = model.to('cuda')
        images, _ = next(iter(test_loader))
        images = images.to('cuda')
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(images)
        
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(images)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 100
        
        print(f"GPU Inference Time: {gpu_time*1000:.2f}ms per batch ({BATCH_SIZE} images)")
        print(f"GPU Throughput: {BATCH_SIZE/gpu_time:.2f} images/second")
    
    # CPU benchmark
    model = model.to('cpu')
    images, _ = next(iter(test_loader))
    images = images.to('cpu')
    
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(images)
    cpu_time = (time.time() - start) / 100
    
    print(f"CPU Inference Time: {cpu_time*1000:.2f}ms per batch ({BATCH_SIZE} images)")
    print(f"CPU Throughput: {BATCH_SIZE/cpu_time:.2f} images/second")

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data not found at {DATA_DIR}")
        print("Please download the GTSRB dataset from Kaggle and unzip it.")
    else:
        # Train ensemble
        models_list, histories, test_loader, class_names = train_model()
        
        # Plot training curves
        plot_training_history(histories)
        
        # Evaluate on test set
        test_acc, preds, labels, probs = evaluate_ensemble(models_list, test_loader, class_names)
        
        # Benchmark inference speed
        benchmark_inference_speed(models_list[0], test_loader)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print("\nGenerated files:")
        print("  - traffic_sign_model_0.pth, traffic_sign_model_1.pth, traffic_sign_model_2.pth")
        print("  - confusion_matrix.png")
        print("  - per_class_accuracy.png")
        print("  - training_history.png")
