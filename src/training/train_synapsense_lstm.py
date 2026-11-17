from focal_loss import FocalLoss
from pathlib import Path
import json
import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    ROOT / "datasets" / "processed" /
    "features" / "mobilenet_v3_small"
)

OUTPUT_DIR = ROOT / "models" / "visual_focal"

LABEL_COLUMNS = [
    "Boredom",
    "Engagement",
    "Confusion",
    "Frustration",
]

INPUT_SIZE = 576
HIDDEN_SIZE = 128
NUM_LAYERS = 1
NUM_CLASSES = 4

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

PATIENCE = 6
NUM_WORKERS = 0

DEVICE = torch.device("cpu")

# Limit CPU thread usage for predictable performance.
torch.set_num_threads(4)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(split):

    folder = FEATURE_ROOT / split

    features = np.load(folder / "features.npy")
    labels = np.load(folder / "labels.npy")

    return (
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def normalize_features(train_x, val_x, test_x):

    # Calculate statistics using training data only.
    mean = train_x.mean(dim=(0, 1))
    std = train_x.std(dim=(0, 1))

    std = torch.where(std < 1e-6, 1.0, std)

    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    return train_x, val_x, test_x, mean, std


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(labels):

    weights = []

    for task_index in range(labels.shape[1]):

        counts = torch.bincount(
            labels[:, task_index],
            minlength=NUM_CLASSES,
        ).float()

        # Inverse-square-root weighting is less aggressive
        # than ordinary inverse-frequency weighting.
        task_weights = torch.sqrt(
            counts.sum() / (NUM_CLASSES * counts.clamp(min=1))
        )

        # Normalize and limit extreme weights.
        task_weights = task_weights / task_weights.mean()
        task_weights = torch.clamp(task_weights, max=3.0)

        weights.append(task_weights)

        print(
            f"{LABEL_COLUMNS[task_index]} class counts:",
            counts.int().tolist(),
        )

        print(
            f"{LABEL_COLUMNS[task_index]} weights:",
            task_weights.round(decimals=3).tolist(),
        )

    return weights


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class SynapSenseLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        self.shared = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.heads = nn.ModuleList([
            nn.Linear(128, NUM_CLASSES)
            for _ in LABEL_COLUMNS
        ])

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        # Final forward and backward hidden states.
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1,
        )

        shared_features = self.shared(combined)

        outputs = [
            head(shared_features)
            for head in self.heads
        ]

        return outputs


# ============================================================
# LOSS FUNCTION
# ============================================================

def calculate_loss(outputs, targets, class_weights, criterion):

    task_losses = []

    for task_index, output in enumerate(outputs):

        loss = criterion(
            output,
            targets[:, task_index],
        )

        task_losses.append(loss)

    return torch.stack(task_losses).mean(), task_losses


# ============================================================
# METRICS
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.inference_mode():

        for features, labels in loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features)

            predictions = torch.stack(
                [output.argmax(dim=1) for output in outputs],
                dim=1,
            )

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_targets)

    metrics = {}
    task_f1_scores = []

    for index, label_name in enumerate(LABEL_COLUMNS):

        true = y_true[:, index]
        pred = y_pred[:, index]

        macro_f1 = f1_score(
            true, pred,
            average="macro",
            zero_division=0,
        )

        task_f1_scores.append(macro_f1)

        metrics[label_name] = {
            "accuracy": accuracy_score(true, pred),
            "precision_macro": precision_score(
                true, pred,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                true, pred,
                average="macro",
                zero_division=0,
            ),
            "f1_macro": macro_f1,
            "f1_weighted": f1_score(
                true, pred,
                average="weighted",
                zero_division=0,
            ),
        }

    metrics["mean_macro_f1"] = float(np.mean(task_f1_scores))

    return metrics, y_true, y_pred


# ============================================================
# TRAINING
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSYNAPSENSE MULTI-TASK LSTM TRAINING")
    print("=" * 65)

    print("Device:", DEVICE)
    print("Loading cached CNN features...")

    train_x, train_y = load_split("train")
    val_x, val_y = load_split("validation")
    test_x, test_y = load_split("test")

    print("Train:", tuple(train_x.shape))
    print("Validation:", tuple(val_x.shape))
    print("Test:", tuple(test_x.shape))

    # Normalize using training statistics only.
    train_x, val_x, test_x, mean, std = normalize_features(
        train_x,
        val_x,
        test_x,
    )

    np.savez(
        OUTPUT_DIR / "feature_normalization.npz",
        mean=mean.numpy(),
        std=std.numpy(),
    )

    # Data loaders.
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    class_weights = calculate_class_weights(train_y)

    model = SynapSenseLSTM().to(DEVICE)

    criterion = nn.ModuleList([
        FocalLoss(
            alpha=weight.to(DEVICE),
            gamma=2.0
        )
        for weight in class_weights
    ])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_score = -1.0
    patience_counter = 0

    history = []

    best_model_path = OUTPUT_DIR / "best_visual_lstm.pth"

    start_time = time.time()

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0

        for features, labels in train_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(features)

            task_losses = []

            for task_index, output in enumerate(outputs):

                loss = criterion[task_index](
                    output,
                    labels[:, task_index],
                )

                task_losses.append(loss)

            loss = torch.stack(task_losses).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()

        average_train_loss = total_loss / len(train_loader)

        val_metrics, _, _ = evaluate(model, val_loader)

        validation_score = val_metrics["mean_macro_f1"]

        scheduler.step(validation_score)

        epoch_result = {
            "epoch": epoch,
            "train_loss": average_train_loss,
            "validation_mean_macro_f1": validation_score,
            "validation_metrics": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        history.append(epoch_result)

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Validation Macro-F1: {validation_score:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        for label_name in LABEL_COLUMNS:

            task = val_metrics[label_name]

            print(
                f"  {label_name}: "
                f"Accuracy={task['accuracy']:.4f}, "
                f"Macro-F1={task['f1_macro']:.4f}, "
                f"Weighted-F1={task['f1_weighted']:.4f}"
            )

        # Save the best validation checkpoint.
        if validation_score > best_score:

            best_score = validation_score
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_size": INPUT_SIZE,
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers": NUM_LAYERS,
                        "num_classes": NUM_CLASSES,
                        "bidirectional": True,
                    },
                    "label_columns": LABEL_COLUMNS,
                    "best_validation_macro_f1": best_score,
                    "epoch": epoch,
                },
                best_model_path,
            )

            print("  Best model checkpoint saved.")

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")
            break

    # Save training history.
    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(history, file, indent=4)

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 65)
    print("FINAL TEST EVALUATION")
    print("=" * 65)

    checkpoint = torch.load(
        best_model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, y_true, y_pred = evaluate(
        model,
        test_loader,
    )

    print(
        "Best validation mean Macro-F1:",
        round(best_score, 4),
    )

    print(
        "Test mean Macro-F1:",
        round(test_metrics["mean_macro_f1"], 4),
    )

    for index, label_name in enumerate(LABEL_COLUMNS):

        print(f"\n{label_name.upper()}")

        print(
            classification_report(
                y_true[:, index],
                y_pred[:, index],
                labels=[0, 1, 2, 3],
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            y_true[:, index],
            y_pred[:, index],
            labels=[0, 1, 2, 3],
        )

        np.savetxt(
            OUTPUT_DIR / f"{label_name.lower()}_confusion_matrix.csv",
            matrix,
            delimiter=",",
            fmt="%d",
        )

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(test_metrics, file, indent=4)

    elapsed = (time.time() - start_time) / 60

    print("\nTraining time:", round(elapsed, 2), "minutes")
    print("Best model:", best_model_path)
    print("Output directory:", OUTPUT_DIR)

    print("\nSYNAPSENSE LSTM TRAINING COMPLETED")


if __name__ == "__main__":
    main()
