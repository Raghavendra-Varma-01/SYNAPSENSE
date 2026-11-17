
from pathlib import Path
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = ROOT / "datasets/processed/features/mobilenet_v3_small"
OUTPUT_DIR = ROOT / "models/visual_attention_focal"

LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]

INPUT_SIZE = 576
HIDDEN_SIZE = 128
NUM_CLASSES = 4

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
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):

    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)

        pt = log_pt.exp()
        focal_factor = (1.0 - pt).pow(self.gamma)

        alpha_t = self.alpha[targets]

        return (-alpha_t * focal_factor * log_pt).mean()


# ============================================================
# MODEL
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
            for _ in LABELS
        ])

    def forward(self, x):
        sequence, _ = self.lstm(x)

        weights = torch.softmax(
            self.attention(sequence), dim=1
        )

        context = torch.sum(sequence * weights, dim=1)
        shared = self.shared(context)

        return [head(shared) for head in self.heads]


# ============================================================
# DATA
# ============================================================

def load_split(split):
    folder = FEATURE_ROOT / split

    x = np.load(folder / "features.npy")
    y = np.load(folder / "labels.npy")

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
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


def calculate_alpha(labels):
    alphas = []

    for i, name in enumerate(LABELS):
        counts = torch.bincount(
            labels[:, i], minlength=NUM_CLASSES
        ).float()

        alpha = torch.sqrt(
            counts.sum() / (NUM_CLASSES * counts.clamp(min=1))
        )

        alpha = alpha / alpha.mean()
        alpha = torch.clamp(alpha, max=3.0)

        alphas.append(alpha.to(DEVICE))

        print(
            f"{name} class weights:",
            alpha.round(decimals=3).tolist()
        )

    return alphas


# ============================================================
# VALIDATION
# ============================================================

def evaluate(model, loader):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.inference_mode():
        for features, labels in loader:
            outputs = model(features.to(DEVICE))

            predictions = torch.stack(
                [output.argmax(dim=1) for output in outputs],
                dim=1,
            )

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(labels.numpy())

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_targets)

    scores = [
        f1_score(
            y_true[:, i],
            y_pred[:, i],
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0,
        )
        for i in range(len(LABELS))
    ]

    return float(np.mean(scores)), scores


# ============================================================
# TRAINING
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSYNAPSENSE ATTENTION + FOCAL LOSS")
    print("=" * 65)
    print("Device:", DEVICE)

    train_x, train_y = load_split("train")
    val_x, val_y = load_split("validation")

    print("Train:", tuple(train_x.shape))
    print("Validation:", tuple(val_x.shape))

    train_x, val_x, mean, std = normalize_features(
        train_x, val_x
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

    alphas = calculate_alpha(train_y)

    criteria = [
        FocalLoss(alpha, gamma=2.0)
        for alpha in alphas
    ]

    model = SynapSenseAttention().to(DEVICE)

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

    checkpoint_path = OUTPUT_DIR / "best_attention_focal.pth"
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        model.train()
        total_loss = 0.0

        for features, labels in train_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(features)

            losses = [
                criteria[i](outputs[i], labels[:, i])
                for i in range(len(LABELS))
            ]

            loss = torch.stack(losses).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        val_score, task_scores = evaluate(model, val_loader)
        scheduler.step(val_score)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Validation Macro-F1: {val_score:.4f} | "
            f"LR: {lr:.6f}"
        )

        for name, score in zip(LABELS, task_scores):
            print(f"  {name}: {score:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_macro_f1": val_score,
            "task_macro_f1": dict(zip(LABELS, task_scores)),
            "learning_rate": lr,
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
                    "gamma": 2.0,
                },
                "label_columns": LABELS,
                "best_validation_macro_f1": best_score,
                "epoch": epoch,
            }, checkpoint_path)

            print("  Best checkpoint saved.")

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\n" + "=" * 65)
    print("ATTENTION + FOCAL TRAINING COMPLETED")
    print("=" * 65)
    print("Best validation Macro-F1:", round(best_score, 4))
    print("Checkpoint:", checkpoint_path)
    print("Time:", round((time.time() - start_time) / 60, 2), "minutes")
    print("Test set NOT evaluated.")


if __name__ == "__main__":
    main()
