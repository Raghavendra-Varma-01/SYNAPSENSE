
from pathlib import Path
import json

import numpy as np
import torch

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    ROOT / "datasets" / "processed" /
    "features" / "mobilenet_v3_small"
)

MODEL_PATH = ROOT / "models" / "visual_lstm" / "best_visual_lstm.pth"
NORMALIZATION_PATH = ROOT / "models" / "visual_lstm" / "feature_normalization.npz"

OUTPUT_DIR = ROOT / "models" / "baseline_comparison"

LABELS = [
    "Boredom",
    "Engagement",
    "Confusion",
    "Frustration",
]


def load_data(split):

    folder = FEATURE_ROOT / split

    x = np.load(folder / "features.npy").astype(np.float32)
    y = np.load(folder / "labels.npy").astype(np.int64)

    return x, y


def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def load_bilstm_predictions(val_x):

    from train_synapsense_lstm import SynapSenseLSTM

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    normalization = np.load(NORMALIZATION_PATH)

    mean = normalization["mean"]
    std = normalization["std"]

    val_x = (val_x - mean) / std

    model = SynapSenseLSTM()

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []

    with torch.inference_mode():

        for start in range(0, len(val_x), 64):

            batch = torch.from_numpy(
                val_x[start:start + 64]
            )

            outputs = model(batch)

            batch_predictions = np.stack(
                [
                    output.argmax(dim=1).numpy()
                    for output in outputs
                ],
                axis=1,
            )

            predictions.append(batch_predictions)

    return np.concatenate(predictions, axis=0)


def main():

    print("\nSYNAPSENSE VISUAL BASELINE COMPARISON")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading training and validation features...")

    train_x, train_y = load_data("train")
    val_x, val_y = load_data("validation")

    print("Train:", train_x.shape)
    print("Validation:", val_x.shape)

    # --------------------------------------------------------
    # 1. MAJORITY CLASS BASELINE
    # --------------------------------------------------------

    print("\n1. MAJORITY CLASS BASELINE")

    majority_predictions = np.zeros_like(val_y)

    for task in range(len(LABELS)):

        majority_class = np.bincount(
            train_y[:, task],
            minlength=4,
        ).argmax()

        majority_predictions[:, task] = majority_class

    # --------------------------------------------------------
    # 2. LOGISTIC REGRESSION
    # --------------------------------------------------------

    print("\n2. LOGISTIC REGRESSION")

    # Mean and standard deviation summarize each video.
    train_mean = train_x.mean(axis=1)
    train_std = train_x.std(axis=1)

    val_mean = val_x.mean(axis=1)
    val_std = val_x.std(axis=1)

    train_pooled = np.concatenate(
        [train_mean, train_std],
        axis=1,
    )

    val_pooled = np.concatenate(
        [val_mean, val_std],
        axis=1,
    )

    logistic_predictions = np.zeros_like(val_y)

    for task, label in enumerate(LABELS):

        print(f"Training {label} classifier...")

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
            ),
        )

        classifier.fit(
            train_pooled,
            train_y[:, task],
        )

        logistic_predictions[:, task] = classifier.predict(
            val_pooled
        )

    # --------------------------------------------------------
    # 3. EXISTING BILSTM
    # --------------------------------------------------------

    print("\n3. EXISTING BIDIRECTIONAL LSTM")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}"
        )

    bilstm_predictions = load_bilstm_predictions(val_x)

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    predictions_by_model = {
        "Majority baseline": majority_predictions,
        "Logistic regression": logistic_predictions,
        "BiLSTM": bilstm_predictions,
    }

    results = {}

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS")
    print("=" * 65)

    for model_name, predictions in predictions_by_model.items():

        results[model_name] = {}

        print(f"\n{model_name.upper()}")

        task_scores = []

        for task, label in enumerate(LABELS):

            metrics = calculate_metrics(
                val_y[:, task],
                predictions[:, task],
            )

            results[model_name][label] = metrics

            task_scores.append(metrics["f1_macro"])

            print(
                f"{label}: "
                f"Accuracy={metrics['accuracy']:.4f} | "
                f"Macro-F1={metrics['f1_macro']:.4f} | "
                f"Weighted-F1={metrics['f1_weighted']:.4f}"
            )

        mean_macro_f1 = float(np.mean(task_scores))

        results[model_name]["mean_macro_f1"] = mean_macro_f1

        print(f"Mean Macro-F1: {mean_macro_f1:.4f}")

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / "validation_baseline_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    ranking = sorted(
        [
            (name, values["mean_macro_f1"])
            for name, values in results.items()
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    print("\n" + "=" * 65)
    print("MODEL RANKING — VALIDATION MACRO-F1")
    print("=" * 65)

    for rank, (name, score) in enumerate(ranking, start=1):
        print(f"{rank}. {name}: {score:.4f}")

    print("\nResults saved to:", output_path)
    print("\nBASELINE COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
