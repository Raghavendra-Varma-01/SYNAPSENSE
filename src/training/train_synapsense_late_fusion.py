
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
VISUAL_CKPT = ROOT / "models/visual_attention/best_attention_lstm.pth"
OUT = ROOT / "models/visual_late_fusion"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.0005
DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Must match the pretrained visual model architecture.
class VisualAttentionBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=576,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.shared = nn.Sequential(
            nn.Linear(256, 128),
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
        context = (sequence * weights.unsqueeze(-1)).sum(dim=1)
        shared = self.shared(context)
        return [head(shared) for head in self.heads]


class LateFusionModel(nn.Module):
    def __init__(self, visual_checkpoint):
        super().__init__()

        self.visual = VisualAttentionBiLSTM()
        checkpoint = torch.load(
            visual_checkpoint,
            map_location="cpu",
            weights_only=False
        )
        self.visual.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )

        # Freeze the pretrained visual stream.
        for parameter in self.visual.parameters():
            parameter.requires_grad = False

        self.face_lstm = nn.LSTM(
            input_size=9,
            hidden_size=32,
            batch_first=True,
            bidirectional=True
        )

        self.face_attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.face_projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.heads = nn.ModuleList([
            nn.Linear(128, 4) for _ in range(4)
        ])

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen visual dropout and other layers in eval mode.
        self.visual.eval()
        return self

    def forward(self, x):
        visual_x = x[:, :, :576]
        face_x = x[:, :, 576:585]

        with torch.no_grad():
            visual_sequence, _ = self.visual.lstm(visual_x)
            visual_weights = torch.softmax(
                self.visual.attention(visual_sequence).squeeze(-1),
                dim=1
            )
            visual_context = (
                visual_sequence * visual_weights.unsqueeze(-1)
            ).sum(dim=1)
            visual_context = self.visual.shared(visual_context)

        face_sequence, _ = self.face_lstm(face_x)
        face_weights = torch.softmax(
            self.face_attention(face_sequence).squeeze(-1),
            dim=1
        )
        face_context = (
            face_sequence * face_weights.unsqueeze(-1)
        ).sum(dim=1)
        face_context = self.face_projection(face_context)

        combined = torch.cat(
            [visual_context, face_context], dim=1
        )
        fused = self.fusion(combined)

        return [head(fused) for head in self.heads]


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

    assert x_train.shape[1:] == (16, 585)
    assert x_val.shape[1:] == (16, 585)

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

    class_weights = []
    for task in range(4):
        counts = np.bincount(
            y_train[:, task].numpy(), minlength=4
        )
        weights = len(y_train) / (4 * np.maximum(counts, 1))
        class_weights.append(
            torch.tensor(weights, dtype=torch.float32)
        )

    model = LateFusionModel(VISUAL_CKPT).to(DEVICE)

    criterion = [
        nn.CrossEntropyLoss(weight=w) for w in class_weights
    ]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=1e-4
    )

    best_score = -1
    best_state = None
    patience = 7
    wait = 0

    print("Pretrained visual checkpoint loaded.")
    print("Visual branch frozen. Training facial + fusion branches.")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

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
                [p for p in model.parameters() if p.requires_grad],
                1.0
            )
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        predictions = [[] for _ in range(4)]
        targets = [[] for _ in range(4)]

        with torch.no_grad():
            for xb, yb in val_loader:
                outputs = model(xb.to(DEVICE))

                for task in range(4):
                    predictions[task].extend(
                        outputs[task].argmax(1).cpu().numpy()
                    )
                    targets[task].extend(yb[:, task].numpy())

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
            f"Loss {total_loss/len(train_loader):.4f} | "
            f"Val Macro-F1 {val_score:.4f} | "
            f"Tasks {[round(s, 4) for s in scores]}"
        )

        if val_score > best_score:
            best_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            wait = 0

            torch.save({
                "model_state_dict": best_state,
                "best_val_macro_f1": best_score,
                "epoch": epoch + 1,
                "seed": SEED,
                "visual_checkpoint": str(VISUAL_CKPT)
            }, OUT / "best_late_fusion.pth")
        else:
            wait += 1

        if wait >= patience:
            print("Early stopping.")
            break

    print(f"\nBest validation Macro-F1: {best_score:.4f}")
    print(f"Saved: {OUT / 'best_late_fusion.pth'}")


if __name__ == "__main__":
    main()
