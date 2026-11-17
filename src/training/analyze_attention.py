
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "datasets/processed/features/mobilenet_v3_small"
MODEL_DIR = ROOT / "models/visual_attention"

LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]
DEVICE = "cpu"
torch.set_num_threads(4)


class SynapSenseAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(576, 128, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.shared = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.heads = nn.ModuleList([nn.Linear(128, 4) for _ in LABELS])

    def forward(self, x):
        sequence, _ = self.lstm(x)
        weights = torch.softmax(self.attention(sequence), dim=1)
        context = torch.sum(sequence * weights, dim=1)
        shared = self.shared(context)
        return [head(shared) for head in self.heads], weights.squeeze(-1)


def main():
    checkpoint = torch.load(
        MODEL_DIR / "best_attention_lstm.pth",
        map_location=DEVICE,
        weights_only=False
    )

    stats = np.load(MODEL_DIR / "feature_normalization.npz")
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)

    folder = FEATURE_ROOT / "validation"
    x = torch.tensor(np.load(folder / "features.npy"), dtype=torch.float32)
    y = np.load(folder / "labels.npy")

    x = (x - mean) / std
    loader = DataLoader(TensorDataset(x), batch_size=64, shuffle=False)

    model = SynapSenseAttention().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    attention_weights = []

    with torch.inference_mode():
        for (batch,) in loader:
            outputs, weights = model(batch)
            predictions.append(
                torch.stack([o.argmax(1) for o in outputs], dim=1).numpy()
            )
            attention_weights.append(weights.numpy())

    pred = np.concatenate(predictions)
    attention = np.concatenate(attention_weights)

    report = {}

    for i, label in enumerate(LABELS):
        cm = confusion_matrix(y[:, i], pred[:, i], labels=[0, 1, 2, 3])
        print(f"\n{'='*55}\n{label}\n{'='*55}")
        print("Confusion matrix:\n", cm)
        print(classification_report(
            y[:, i], pred[:, i],
            labels=[0, 1, 2, 3],
            zero_division=0
        ))
        report[label] = {
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y[:, i], pred[:, i],
                labels=[0, 1, 2, 3],
                output_dict=True,
                zero_division=0
            )
        }

    average_attention = attention.mean(axis=0)
    print("\nAverage attention weight per frame:")
    for i, value in enumerate(average_attention):
        print(f"Frame {i:02d}: {value:.5f}")

    report["average_attention_by_frame"] = average_attention.tolist()

    with open(MODEL_DIR / "attention_error_analysis.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\nSaved:", MODEL_DIR / "attention_error_analysis.json")


if __name__ == "__main__":
    main()
