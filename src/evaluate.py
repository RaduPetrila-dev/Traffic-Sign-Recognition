"""Evaluation, visualisation, and inference benchmarking."""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.config import BATCH_SIZE, DEVICE, NUM_CLASSES, OUTPUT_DIR
from src.model import ensemble_predict


def evaluate_ensemble(models_list, test_loader):
    """Evaluate the ensemble on the test set.

    Returns:
        tuple: (test_accuracy, predictions, labels, probabilities)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            probs = ensemble_predict(models_list, images)
            _, predicted = torch.max(probs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = 100.0 * np.mean(all_preds == all_labels)

    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=[str(i) for i in range(NUM_CLASSES)],
    ))

    _plot_confusion_matrix(all_labels, all_preds, test_acc)
    _plot_per_class_accuracy(all_labels, all_preds, test_acc)
    _print_safety_analysis(all_labels, all_preds)

    return test_acc, all_preds, all_labels, np.array(all_probs)


def _plot_confusion_matrix(labels, preds, test_acc):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.title(f"Confusion Matrix (Accuracy: {test_acc:.2f}%)", fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def _plot_per_class_accuracy(labels, preds, test_acc):
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(15, 6))
    colours = ["#e74c3c" if acc < 0.9 else "#2ecc71" for acc in per_class_acc]
    plt.bar(range(NUM_CLASSES), per_class_acc * 100, color=colours)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Per-Class Accuracy (red = below 90%)", fontsize=14)
    plt.axhline(y=test_acc, color="k", linestyle="--",
                label=f"Overall: {test_acc:.2f}%")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "per_class_accuracy.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def _print_safety_analysis(labels, preds):
    """Print the top misclassification pairs for safety review."""
    cm = confusion_matrix(labels, preds)
    misclass = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                misclass.append((cm[i, j], i, j))
    misclass.sort(reverse=True)

    print(f"\n{'=' * 60}")
    print("TOP MISCLASSIFICATIONS (Safety Analysis)")
    print(f"{'=' * 60}")
    for count, true_label, pred_label in misclass[:10]:
        print(f"  Class {true_label} -> {pred_label}: {count} times")


def plot_training_history(histories):
    """Plot training curves for all ensemble members."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for idx, h in enumerate(histories):
        axes[0, 0].plot(h["train_loss"], label=f"Model {idx+1} Train", alpha=0.7)
        axes[0, 0].plot(h["val_loss"], label=f"Model {idx+1} Val",
                        alpha=0.7, linestyle="--")
        axes[0, 1].plot(h["train_acc"], label=f"Model {idx+1} Train", alpha=0.7)
        axes[0, 1].plot(h["val_acc"], label=f"Model {idx+1} Val",
                        alpha=0.7, linestyle="--")

    for ax, title, ylabel in [
        (axes[0, 0], "Training & Validation Loss", "Loss"),
        (axes[0, 1], "Training & Validation Accuracy", "Accuracy (%)"),
    ]:
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Ensemble averages
    avg_loss_t = np.mean([h["train_loss"] for h in histories], axis=0)
    avg_loss_v = np.mean([h["val_loss"] for h in histories], axis=0)
    avg_acc_t = np.mean([h["train_acc"] for h in histories], axis=0)
    avg_acc_v = np.mean([h["val_acc"] for h in histories], axis=0)

    axes[1, 0].plot(avg_loss_t, label="Ensemble Train", linewidth=2)
    axes[1, 0].plot(avg_loss_v, label="Ensemble Val", linewidth=2, linestyle="--")
    axes[1, 1].plot(avg_acc_t, label="Ensemble Train", linewidth=2)
    axes[1, 1].plot(avg_acc_v, label="Ensemble Val", linewidth=2, linestyle="--")

    for ax, title, ylabel in [
        (axes[1, 0], "Ensemble Average Loss", "Loss"),
        (axes[1, 1], "Ensemble Average Accuracy", "Accuracy (%)"),
    ]:
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_history.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def benchmark_inference(model, test_loader, n_runs: int = 100):
    """Benchmark single-model inference on CPU and GPU."""
    print(f"\n{'=' * 60}")
    print("INFERENCE BENCHMARK")
    print(f"{'=' * 60}")

    model.eval()
    images, _ = next(iter(test_loader))

    if torch.cuda.is_available():
        model_gpu = model.to("cuda")
        imgs_gpu = images.to("cuda")
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model_gpu(imgs_gpu)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                model_gpu(imgs_gpu)
        torch.cuda.synchronize()
        gpu_ms = (time.time() - start) / n_runs * 1000
        print(f"GPU: {gpu_ms:.2f} ms/batch ({BATCH_SIZE} images)")
        print(f"GPU throughput: {BATCH_SIZE / (gpu_ms / 1000):.0f} images/sec")

    model_cpu = model.to("cpu")
    imgs_cpu = images.to("cpu")
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model_cpu(imgs_cpu)
    cpu_ms = (time.time() - start) / n_runs * 1000
    print(f"CPU: {cpu_ms:.2f} ms/batch ({BATCH_SIZE} images)")
    print(f"CPU throughput: {BATCH_SIZE / (cpu_ms / 1000):.0f} images/sec")
