from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .dataset import CIFAR10Dataset
from .model import CNN


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(loader, model, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)

def compute_metrics(labels, preds):

    acc = accuracy_score(labels, preds) # Proporción de predicciones correctas sobre el total de muestras.
    f1_macro = f1_score(labels, preds, average="macro") # Promedio F1 sin ponderar por tamaño de clase.
    f1_weighted = f1_score(labels, preds, average="weighted") # Pondera F1por número de muestras por clase.

    return {
        "Accuracy": acc,
        "F1_macro": f1_macro,
        "F1_weighted": f1_weighted,
    }


def plot_confusion_matrix(labels, preds, class_names, filepath):

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d',
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


if __name__ == "__main__":

    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)

    device = get_device()
    print("Using device:", device)

    # DATASETS
    full_train = CIFAR10Dataset("./data", train=True)
    test_dataset = CIFAR10Dataset("./data", train=False)

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # MODEL 
    model = CNN(num_classes=10)
    model.load_state_dict(torch.load(output_folder / "best_model.pth"))
    model.to(device)

    metrics_dict = {}

    for name, loader in {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader
    }.items():

        labels, preds = evaluate_model(loader, model, device)

        metrics = compute_metrics(labels, preds)
        metrics_dict[name] = metrics

        print(f"\n=== {name.upper()} METRICS ===")
        print(metrics)
        print(classification_report(labels, preds))

        plot_confusion_matrix(
            labels,
            preds,
            full_train.data.classes,
            output_folder / f"{name}_confusion_matrix.png"
        )

    # Save metrics as CSV
    df_metrics = pd.DataFrame(metrics_dict).T
    df_metrics.to_csv(output_folder / "metrics.csv")

    # Save metrics as image
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    table = ax.table(
        cellText=df_metrics.values,
        colLabels=df_metrics.columns,
        rowLabels=df_metrics.index,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    plt.savefig(output_folder / "metrics.png")
    plt.close()

    print("\nEvaluation complete.")