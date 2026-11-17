
from pathlib import Path
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = ROOT / "datasets" / "processed" / "features" / "mobilenet_v3_small"
OUTPUT_DIR = ROOT / "models" / "visual_attention"

LABEL_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]

INPUT_SIZE = 576
HIDDEN_SIZE = 128
NUM_CLASSES = 4
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 6
NUM_WORKERS = 0

DEVICE = torch.device("cpu")
torch.set_num_threads(4)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATA
# ============================================================

def load_split(split):
    folder = FEATURE_ROOT / split
    features = np.load(folder / "features.npy")
    labels = np.load(folder / "labels.npy")

    return (
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )


def normalize_features(train_x, val_x):
    mean = train_x.mean(dim=(0, 1))
    std = train_x.std(dim=(0, 1))
    std = torch.where(std < 1e-6, 1.0, std)

    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        mean,
        std,
    )


def calculate_class_weights(labels):
    weights = []

    for task_index in range(labels.shape[1]):
        counts = torch.bincount(
            labels[:, task_index], minlength=NUM_CLASSES
        ).float()

        task_weights = torch.sqrt(
            counts.sum() / (NUM_CLASSES * counts.clamp(min=1))
        )
        task_weights = task_weights / task_weights.mean()
        task_weights = torch.clamp(task_weights, max=3.0)

        weights.append(task_weights.to(DEVICE))

    return weights


# ============================================================
# TEMPORAL ATTENTION BiLSTM
# ============================================================

class SynapSenseAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Learn an importance score for each of the 16 frames.
        self.attention = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
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

    def forward(self, x, return_attention=False):

        sequence_output, _ = self.lstm(x)

        attention_scores = self.attention(sequence_output)
        attention_weights = torch.softmax(attention_scores, dim=1)

        context = torch.sum(
            sequence_output * attention_weights,
            dim=1,
        )

        shared_features = self.shared(context)

        outputs = [
            head(shared_features)
            for head in self.heads
        ]

        if return_attention:
            return outputs, attention_weights.squeeze(-1)

        return outputs


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()
    predictions = []
    targets = []

    with torch.inference_mode():
        for features, labels in loader:
            outputs = model(features.to(DEVICE))
            batch_predictions = torch.stack(
                [output.argmax(dim=1) for output in outputs],
                dim=1,
            )

            predictions.append(batch_predictions.cpu().numpy())
            targets.append(labels.numpy())

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)

    task_scores = [
        f1_score(
            y_true[:, i],
            y_pred[:, i],
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0,
        )
        for i in range(len(LABEL_COLUMNS))
    ]

    return float(np.mean(task_scores)), y_true, y_pred, task_scores


# ============================================================
# TRAINING
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSYNAPSENSE TEMPORAL ATTENTION BiLSTM")
    print("=" * 65)
    print("Device:", DEVICE)

    train_x, train_y = load_split("train")
    val_x, val_y = load_split("validation")

    print("Train:", tuple(train_x.shape))
    print("Validation:", tuple(val_x.shape))

    train_x, val_x, mean, std = normalize_features(train_x, val_x)

    np.savez(
        OUTPUT_DIR / "feature_normalization.npz",
        mean=mean.numpy(),
        std=std.numpy(),
    )

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

    class_weights = calculate_class_weights(train_y)

    model = SynapSenseAttention().to(DEVICE)

    criterion = nn.ModuleList([
        nn.CrossEntropyLoss(weight=weight)
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

    best_model_path = OUTPUT_DIR / "best_attention_lstm.pth"

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        model.train()
        running_loss = 0.0

        for features, labels in train_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(features)

            task_losses = [
                criterion[i](outputs[i], labels[:, i])
                for i in range(len(LABEL_COLUMNS))
            ]

            loss = torch.stack(task_losses).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        val_score, _, _, task_scores = evaluate(model, val_loader)

        scheduler.step(val_score)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Validation Macro-F1: {val_score:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        for name, score in zip(LABEL_COLUMNS, task_scores):
            print(f"  {name}: Macro-F1={score:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_macro_f1": val_score,
            "task_macro_f1": dict(zip(LABEL_COLUMNS, task_scores)),
            "learning_rate": current_lr,
        })

        if val_score > best_score:
            best_score = val_score
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_size": INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_classes": NUM_CLASSES,
                },
                "label_columns": LABEL_COLUMNS,
                "best_validation_macro_f1": best_score,
                "epoch": epoch,
            }, best_model_path)

            print("  Best attention checkpoint saved.")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\n" + "=" * 65)
    print("ATTENTION TRAINING COMPLETED")
    print("=" * 65)
    print("Best validation Macro-F1:", round(best_score, 4))
    print("Best checkpoint:", best_model_path)
    print("Training time:", round((time.time() - start_time) / 60, 2), "minutes")
    print("Test set has NOT been evaluated.")


if __name__ == "__main__":
    main()
