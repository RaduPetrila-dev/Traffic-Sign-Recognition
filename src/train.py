"""Training loop with early stopping, LR scheduling, and checkpointing."""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_DIR, PATIENCE


def train_single_model(model, train_loader, val_loader, model_idx: int) -> dict:
    """Train a single model with early stopping and LR scheduling.

    Args:
        model: nn.Module to train
        train_loader: training DataLoader
        val_loader: validation DataLoader
        model_idx: index for checkpoint naming

    Returns:
        dict with keys: train_loss, val_loss, train_acc, val_acc (lists per epoch)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(MODEL_DIR, f"traffic_sign_model_{model_idx}.pth")
    start_time = time.time()

    for epoch in range(EPOCHS):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

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
        train_acc = 100.0 * correct / total

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        scheduler.step(val_acc)

        # --- Early stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    elapsed = time.time() - start_time
    print(f"Model {model_idx + 1} finished in {elapsed:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Reload best weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    return history


def train_ensemble(models_list, train_loader, val_loader) -> list:
    """Train all models in the ensemble sequentially.

    Returns:
        list of history dicts, one per model.
    """
    histories = []
    for idx, model in enumerate(models_list):
        print(f"\n{'=' * 60}")
        print(f"Training Model {idx + 1}/{len(models_list)}")
        print(f"{'=' * 60}")
        history = train_single_model(model, train_loader, val_loader, idx)
        histories.append(history)
    return histories
