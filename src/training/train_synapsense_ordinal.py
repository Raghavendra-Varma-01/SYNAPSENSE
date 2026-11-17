
from pathlib import Path
import json
import random
import time

import numpy as np
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

BASELINE_DIR = ROOT / "models" / "visual_lstm"

OUTPUT_DIR = ROOT / "models" / "ordinal_lstm"

LABEL_COLUMNS = [
    "Boredom",
    "Engagement",
    "Confusion",
    "Frustration",
]

INPUT_SIZE = 576
HIDDEN_SIZE = 128
NUM_CLASSES = 4
NUM_THRESHOLDS = 3

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 6

DEVICE = torch.device("cpu")

torch.set_num_threads(4)

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
# NORMALIZATION
# ============================================================

def normalize_features(train_x, val_x):

    mean = train_x.mean(dim=(0, 1))
    std = train_x.std(dim=(0, 1))

    std = torch.where(std < 1e-6, 1.0, std)

    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    return train_x, val_x, mean, std


# ============================================================
# ORDINAL TARGETS
# ============================================================

def create_ordinal_targets(labels):

    thresholds = torch.tensor(
        [0, 1, 2],
        dtype=labels.dtype,
    )

    # y > 0, y > 1, y > 2
    targets = (
        labels.unsqueeze(-1) > thresholds
    ).float()

    return targets


# ============================================================
# CLASS IMBALANCE WEIGHTS
# ============================================================

def calculate_positive_weights(labels):

    ordinal_targets = create_ordinal_targets(labels)

    # Shape: (4 tasks, 3 thresholds)
    positive_counts = ordinal_targets.sum(dim=0)

    negative_counts = (
        ordinal_targets.shape[0] - positive_counts
    )

    # Square-root weighting to avoid excessively large weights.
    weights = torch.sqrt(
        negative_counts / positive_counts.clamp(min=1)
    )

    weights = torch.clamp(weights, min=1.0, max=5.0)

    print("\nORDINAL THRESHOLD WEIGHTS")

    for task_index, label_name in enumerate(LABEL_COLUMNS):

        print(
            f"{label_name}:",
            weights[task_index].round(decimals=3).tolist(),
        )

    return weights


# ============================================================
# ORDINAL BiLSTM MODEL
# ============================================================

class SynapSenseOrdinalLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.shared = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # One continuous severity score per task.
        self.task_heads = nn.ModuleList([
            nn.Linear(128, 1)
            for _ in LABEL_COLUMNS
        ])

        # First threshold is unconstrained.
        self.base_thresholds = nn.Parameter(
            torch.zeros(len(LABEL_COLUMNS))
        )

        # Positive increments guarantee ordered thresholds.
        self.threshold_increments = nn.Parameter(
            torch.zeros(len(LABEL_COLUMNS), 2)
        )

    def get_thresholds(self):

        increments = nn.functional.softplus(
            self.threshold_increments
        )

        first = self.base_thresholds.unsqueeze(1)

        second = first + increments[:, 0:1]

        third = second + increments[:, 1:2]

        return torch.cat(
            [first, second, third],
            dim=1,
        )

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1,
        )

        representation = self.shared(combined)

        scores = torch.cat(
            [head(representation) for head in self.task_heads],
            dim=1,
        )

        thresholds = self.get_thresholds()

        # Shape: (batch, 4 tasks, 3 thresholds)
        logits = (
            scores.unsqueeze(-1) -
            thresholds.unsqueeze(0)
        )

        return logits


# ============================================================
# ORDINAL LOSS
# ============================================================

def ordinal_loss(logits, targets, positive_weights):

    # BCE per threshold.
    raw_loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=positive_weights.unsqueeze(0),
    )

    # Average thresholds separately for each task,
    # then give the four tasks equal importance.
    task_losses = raw_loss.mean(dim=2)

    return task_losses.mean()


# ============================================================
# CONVERT ORDINAL OUTPUTS TO CLASSES
# ============================================================

def decode_predictions(logits):

    probabilities = torch.sigmoid(logits)

    # Each threshold crossed contributes one class level.
    predictions = (
        probabilities >= 0.5
    ).sum(dim=2)

    return predictions.long()


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    predictions_list = []
    targets_list = []

    with torch.inference_mode():

        for features, labels in loader:

            logits = model(features)

            predictions = decode_predictions(logits)

            predictions_list.append(
                predictions.cpu().numpy()
            )

            targets_list.append(
                labels.numpy()
            )

    y_pred = np.concatenate(predictions_list)
    y_true = np.concatenate(targets_list)

    metrics = {}
    scores = []

    for index, label_name in enumerate(LABEL_COLUMNS):

        true = y_true[:, index]
        pred = y_pred[:, index]

        task_metrics = {
            "accuracy": float(
                accuracy_score(true, pred)
            ),
            "precision_macro": float(
                precision_score(
                    true, pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "recall_macro": float(
                recall_score(
                    true, pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "f1_macro": float(
                f1_score(
                    true, pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "f1_weighted": float(
                f1_score(
                    true, pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

        metrics[label_name] = task_metrics
        scores.append(task_metrics["f1_macro"])

    metrics["mean_macro_f1"] = float(np.mean(scores))

    return metrics, y_true, y_pred


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSYNAPSENSE ORDINAL BiLSTM TRAINING")
    print("=" * 65)

    print("Device:", DEVICE)

    train_x, train_y = load_split("train")
    val_x, val_y = load_split("validation")

    print("Train:", tuple(train_x.shape))
    print("Validation:", tuple(val_x.shape))

    train_x, val_x, mean, std = normalize_features(
        train_x,
        val_x,
    )

    np.savez(
        OUTPUT_DIR / "feature_normalization.npz",
        mean=mean.numpy(),
        std=std.numpy(),
    )

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    positive_weights = calculate_positive_weights(
        train_y
    )

    model = SynapSenseOrdinalLSTM().to(DEVICE)

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

    checkpoint_path = (
        OUTPUT_DIR / "best_ordinal_lstm.pth"
    )

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0.0

        for features, labels in train_loader:

            optimizer.zero_grad()

            logits = model(features)

            targets = create_ordinal_targets(labels)

            loss = ordinal_loss(
                logits,
                targets,
                positive_weights,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
        )

        validation_score = val_metrics["mean_macro_f1"]

        scheduler.step(validation_score)

        result = {
            "epoch": epoch,
            "train_loss": average_loss,
            "validation_mean_macro_f1": validation_score,
            "validation_metrics": val_metrics,
        }

        history.append(result)

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation Macro-F1: {validation_score:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        for label_name in LABEL_COLUMNS:

            task = val_metrics[label_name]

            print(
                f"  {label_name}: "
                f"Accuracy={task['accuracy']:.4f} | "
                f"Macro-F1={task['f1_macro']:.4f} | "
                f"Weighted-F1={task['f1_weighted']:.4f}"
            )

        if validation_score > best_score:

            best_score = validation_score
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_size": INPUT_SIZE,
                        "hidden_size": HIDDEN_SIZE,
                        "num_classes": NUM_CLASSES,
                        "num_thresholds": NUM_THRESHOLDS,
                        "bidirectional": True,
                    },
                    "label_columns": LABEL_COLUMNS,
                    "best_validation_macro_f1": best_score,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

            print("  Best ordinal checkpoint saved.")

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")
            break

    with open(
        OUTPUT_DIR / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(history, file, indent=4)

    print("\n" + "=" * 65)
    print("ORDINAL TRAINING COMPLETED")
    print("=" * 65)

    print("Best validation Macro-F1:", round(best_score, 4))
    print("Checkpoint:", checkpoint_path)
    print("Elapsed minutes:", round(
        (time.time() - start_time) / 60, 2
    ))

    print("\nIndependent test set has NOT been evaluated.")
    print("We will evaluate it only after model selection.")


if __name__ == "__main__":
    main()

