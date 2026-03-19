"""Main entry point: train ensemble, evaluate, and benchmark."""

from src.config import DEVICE
from src.data import get_dataloaders
from src.evaluate import benchmark_inference, evaluate_ensemble, plot_training_history
from src.model import create_ensemble
from src.train import train_ensemble


def main():
    print(f"Using device: {DEVICE}\n")

    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    models_list = create_ensemble()
    histories = train_ensemble(models_list, train_loader, val_loader)

    plot_training_history(histories)
    test_acc, _, _, _ = evaluate_ensemble(models_list, test_loader)
    benchmark_inference(models_list[0], test_loader)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE  |  Test Accuracy: {test_acc:.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
