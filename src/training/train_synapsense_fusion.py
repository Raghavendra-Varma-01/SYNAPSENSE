
from pathlib import Path
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/processed/features/fusion"
OUT = ROOT / "models/visual_fusion"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.0005
HIDDEN = 128
DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class FusionAttentionBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=585,
            hidden_size=HIDDEN,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(HIDDEN * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.shared = nn.Sequential(
            nn.Linear(HIDDEN * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.heads = nn.ModuleList([
            nn.Linear(128, 4) for _ in range(4)
        ])

    def forward(self, x):
        sequence, _ = self.lstm(x)
        weights = torch.softmax(
            self.attention(sequence).squeeze(-1), dim=1
        )
        context = torch.sum(
            sequence * weights.unsqueeze(-1), dim=1
        )
        shared = self.shared(context)
        return [head(shared) for head in self.heads]


def load_split(split):
    folder = DATA / split
    x = torch.tensor(
        np.load(folder / "features.npy"), dtype=torch.float32
    )
    y = torch.tensor(
        np.load(folder / "labels.npy"), dtype=torch.long
    )
    return x, y


def main():
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("validation")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Class weights calculated from training labels only.
    weights = []
    for task in range(4):
        counts = np.bincount(
            y_train[:, task].numpy(), minlength=4
        )
        w = len(y_train) / (4 * np.maximum(counts, 1))
        weights.append(torch.tensor(w, dtype=torch.float32))

    model = FusionAttentionBiLSTM().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    criterion = nn.ModuleList([
        nn.CrossEntropyLoss(weight=w) for w in weights
    ])

    best_score = -1
    best_state = None
    patience = 7
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(xb)

            loss = sum(
                criterion[t](outputs[t], yb[:, t])
                for t in range(4)
            ) / 4

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        predictions = [[], [], [], []]
        targets = [[], [], [], []]

        with torch.no_grad():
            for xb, yb in val_loader:
                outputs = model(xb.to(DEVICE))
                for t in range(4):
                    pred = outputs[t].argmax(1).cpu().numpy()
                    predictions[t].extend(pred)
                    targets[t].extend(yb[:, t].numpy())

        scores = [
            f1_score(
                targets[t], predictions[t],
                average="macro", zero_division=0
            )
            for t in range(4)
        ]
        val_score = float(np.mean(scores))

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Loss {train_loss/len(train_loader):.4f} | "
            f"Val Macro-F1 {val_score:.4f} | "
            f"Tasks {[round(s, 4) for s in scores]}"
        )

        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            wait = 0

            torch.save({
                "model_state_dict": best_state,
                "input_size": 585,
                "hidden_size": HIDDEN,
                "best_val_macro_f1": best_score,
                "epoch": epoch + 1,
                "seed": SEED
            }, OUT / "best_fusion_attention.pth")
        else:
            wait += 1

        if wait >= patience:
            print("Early stopping.")
            break

    print(f"\nBest validation Macro-F1: {best_score:.4f}")
    print(f"Checkpoint: {OUT / 'best_fusion_attention.pth'}")


if __name__ == "__main__":
    main()
