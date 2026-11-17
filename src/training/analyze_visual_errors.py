
from pathlib import Path
import json

import numpy as np
import torch
import pandas as pd

from train_synapsense_lstm import SynapSenseLSTM
from train_synapsense_ordinal import SynapSenseOrdinalLSTM

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    ROOT / "datasets" / "processed" /
    "features" / "mobilenet_v3_small"
)

OUTPUT = ROOT / "models" / "error_analysis"

LABELS = [
    "Boredom",
    "Engagement",
    "Confusion",
    "Frustration",
]

DEVICE = "cpu"


def load_validation():

    folder = FEATURE_ROOT / "validation"

    x = np.load(folder / "features.npy").astype(np.float32)
    y = np.load(folder / "labels.npy").astype(np.int64)

    return x, y


def load_weighted_model():

    folder = ROOT / "models" / "visual_lstm"

    norm = np.load(folder / "feature_normalization.npz")

    x_mean = norm["mean"]
    x_std = norm["std"]

    model = SynapSenseLSTM()

    checkpoint = torch.load(
        folder / "best_visual_lstm.pth",
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, x_mean, x_std


def load_ordinal_model():

    folder = ROOT / "models" / "ordinal_lstm"

    norm = np.load(folder / "feature_normalization.npz")

    model = SynapSenseOrdinalLSTM()

    checkpoint = torch.load(
        folder / "best_ordinal_lstm.pth",
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, norm["mean"], norm["std"]


def predict(model, x, mean, std, ordinal=False):

    x = (x - mean) / std

    predictions = []

    with torch.inference_mode():

        for start in range(0, len(x), 64):

            batch = torch.from_numpy(x[start:start + 64])

            output = model(batch)

            if ordinal:
                pred = (
                    torch.sigmoid(output) >= 0.5
                ).sum(dim=2)

            else:
                pred = torch.stack(
                    [task.argmax(dim=1) for task in output],
                    dim=1,
                )

            predictions.append(pred.numpy())

    return np.concatenate(predictions, axis=0)


def analyze_model(name, y_true, y_pred):

    print("\n" + "=" * 65)
    print(name.upper())
    print("=" * 65)

    results = {}

    for task, label in enumerate(LABELS):

        true = y_true[:, task]
        pred = y_pred[:, task]

        matrix = confusion_matrix(
            true,
            pred,
            labels=[0, 1, 2, 3],
        )

        report = classification_report(
            true,
            pred,
            labels=[0, 1, 2, 3],
            output_dict=True,
            zero_division=0,
        )

        print(f"\n{label.upper()}")
        print("Confusion matrix:")
        print(matrix)

        print("Actual distribution:", np.bincount(true, minlength=4))
        print("Predicted distribution:", np.bincount(pred, minlength=4))

        results[label] = {
            "confusion_matrix": matrix.tolist(),
            "actual_distribution": np.bincount(
                true, minlength=4
            ).tolist(),
            "predicted_distribution": np.bincount(
                pred, minlength=4
            ).tolist(),
            "macro_f1": float(
                f1_score(true, pred, average="macro", zero_division=0)
            ),
            "classification_report": report,
        }

    return results


def main():

    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("\nSYNAPSENSE VALIDATION ERROR ANALYSIS")
    print("=" * 65)

    x, y = load_validation()

    print("Validation samples:", len(y))

    weighted_model, mean, std = load_weighted_model()

    weighted_predictions = predict(
        weighted_model, x, mean, std
    )

    ordinal_model, mean, std = load_ordinal_model()

    ordinal_predictions = predict(
        ordinal_model, x, mean, std, ordinal=True
    )

    results = {
        "Weighted_BiLSTM": analyze_model(
            "Weighted BiLSTM",
            y,
            weighted_predictions,
        ),
        "Ordinal_BiLSTM": analyze_model(
            "Ordinal BiLSTM",
            y,
            ordinal_predictions,
        ),
    }

    output_file = OUTPUT / "validation_error_analysis.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nSaved:", output_file)
    print("\nERROR ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
