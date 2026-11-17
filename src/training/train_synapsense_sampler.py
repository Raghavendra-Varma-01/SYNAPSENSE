
from pathlib import Path
import json, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "datasets/processed/features/mobilenet_v3_small"
OUTPUT_DIR = ROOT / "models/visual_attention_sampler"
LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]

torch.set_num_threads(4)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(576, 128, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.shared = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.heads = nn.ModuleList([nn.Linear(128, 4) for _ in LABELS])

    def forward(self, x):
        seq, _ = self.lstm(x)
        weights = torch.softmax(self.attention(seq), dim=1)
        context = torch.sum(seq * weights, dim=1)
        shared = self.shared(context)
        return [head(shared) for head in self.heads]

def load(split):
    folder = FEATURE_ROOT / split
    return (
        torch.tensor(np.load(folder / "features.npy"), dtype=torch.float32),
        torch.tensor(np.load(folder / "labels.npy"), dtype=torch.long)
    )

def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.inference_mode():
        for x, y in loader:
            outputs = model(x)
            preds.append(torch.stack([o.argmax(1) for o in outputs], 1).numpy())
            targets.append(y.numpy())
    p, y = np.concatenate(preds), np.concatenate(targets)
    scores = [
        f1_score(y[:, i], p[:, i], labels=[0,1,2,3],
                 average="macro", zero_division=0)
        for i in range(4)
    ]
    return float(np.mean(scores)), scores

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("ATTENTION BiLSTM + WEIGHTED SAMPLER")

    train_x, train_y = load("train")
    val_x, val_y = load("validation")

    mean = train_x.mean((0,1))
    std = train_x.std((0,1)).clamp(min=1e-6)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    np.savez(OUTPUT_DIR / "feature_normalization.npz",
             mean=mean.numpy(), std=std.numpy())

    # Each clip receives a sampling weight based on its rarest
    # observed class across the four prediction tasks.
    sample_weights = torch.ones(len(train_y), dtype=torch.double)
    for task in range(4):
        counts = torch.bincount(train_y[:, task], minlength=4).double()
        class_weights = (counts.sum() / (4 * counts.clamp(min=1))).sqrt()
        sample_weights = torch.maximum(
            sample_weights, class_weights[train_y[:, task]]
        )

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=64, sampler=sampler
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y), batch_size=64, shuffle=False
    )

    class_weights = []
    for task in range(4):
        counts = torch.bincount(train_y[:, task], minlength=4).float()
        weights = (counts.sum() / (4 * counts.clamp(min=1))).sqrt()
        weights = (weights / weights.mean()).clamp(max=3.0)
        class_weights.append(weights)

    model = Model()
    criteria = [nn.CrossEntropyLoss(weight=w) for w in class_weights]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best, patience, history = -1, 0, []
    start = time.time()
    checkpoint = OUTPUT_DIR / "best_attention_sampler.pth"

    for epoch in range(1, 31):
        model.train()
        total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = torch.stack([
                criteria[i](outputs[i], y[:, i]) for i in range(4)
            ]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

        score, task_scores = evaluate(model, val_loader)
        scheduler.step(score)
        print(f"\nEpoch {epoch:02d} | Loss {total/len(train_loader):.4f} | Val Macro-F1 {score:.4f}")
        for name, value in zip(LABELS, task_scores):
            print(f"  {name}: {value:.4f}")

        history.append({
            "epoch": epoch, "validation_macro_f1": score,
            "task_macro_f1": dict(zip(LABELS, task_scores))
        })

        if score > best:
            best, patience = score, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {"input_size":576, "hidden_size":128, "num_classes":4},
                "label_columns": LABELS,
                "best_validation_macro_f1": best,
                "epoch": epoch
            }, checkpoint)
            print("  Best checkpoint saved.")
        else:
            patience += 1

        if patience >= 6:
            print("Early stopping.")
            break

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\nCompleted. Best validation Macro-F1:", round(best, 4))
    print("Time:", round((time.time()-start)/60, 2), "minutes")
    print("Test set NOT evaluated.")

if __name__ == "__main__":
    main()
