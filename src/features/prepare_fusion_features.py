
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CNN = ROOT / "datasets/processed/features/mobilenet_v3_small"
FACIAL = ROOT / "datasets/processed/facial_features"
OUT = ROOT / "datasets/processed/features/fusion"
OUT.mkdir(parents=True, exist_ok=True)

COLS = [
    "left_ear", "right_ear", "mean_ear", "mar",
    "face_width", "face_height",
    "head_yaw_proxy", "head_pitch_proxy"
]
SPLITS = ["train", "validation", "test"]

# Fit normalization ONLY on training data.
train_cnn = np.load(CNN / "train/features.npy")
cnn_mean = train_cnn.mean(axis=(0, 1))
cnn_std = train_cnn.std(axis=(0, 1))
cnn_std = np.maximum(cnn_std, 1e-6)

train_face = pd.read_csv(
    FACIAL / "train/facial_features.csv"
)
valid_face = train_face.loc[
    train_face["face_detected"] == 1, COLS
]

face_mean = valid_face.mean().to_numpy(dtype=np.float32)
face_std = valid_face.std(ddof=0).to_numpy(dtype=np.float32)
face_std = np.maximum(face_std, 1e-6)

np.savez(
    OUT / "normalization.npz",
    cnn_mean=cnn_mean,
    cnn_std=cnn_std,
    face_mean=face_mean,
    face_std=face_std
)

for split in SPLITS:
    base = CNN / split

    visual = np.load(base / "features.npy")
    labels = np.load(base / "labels.npy")
    ids = np.load(base / "clip_ids.npy").astype(str)

    assert visual.shape == (len(ids), 16, 576)
    assert len(labels) == len(ids)
    assert len(set(ids)) == len(ids)

    df = pd.read_csv(
        FACIAL / split / "facial_features.csv",
        dtype={"clip_id": str, "frame": str}
    )

    df = df.set_index(["clip_id", "frame"])
    assert df.index.is_unique, "Duplicate facial frame records"

    expected = pd.MultiIndex.from_product(
        [ids, [f"frame_{i:02d}" for i in range(16)]],
        names=["clip_id", "frame"]
    )

    assert len(df) == len(expected)
    assert df.index.difference(expected).empty
    df = df.reindex(expected)

    mask = df["face_detected"].fillna(0).to_numpy(
        dtype=np.float32
    ).reshape(len(ids), 16, 1)

    face = df[COLS].to_numpy(
        dtype=np.float32
    ).reshape(len(ids), 16, 8)

    # Missing detections become training-set mean (normalized zero).
    face = (face - face_mean) / face_std
    face = np.nan_to_num(face, nan=0.0, posinf=0.0, neginf=0.0)
    face = np.clip(face, -5, 5)

    visual = (visual - cnn_mean) / cnn_std
    visual = np.clip(visual, -5, 5)

    fused = np.concatenate(
        [visual, face, mask], axis=2
    ).astype(np.float32)

    assert fused.shape == (len(ids), 16, 585)
    assert np.isfinite(fused).all()

    target = OUT / split
    target.mkdir(parents=True, exist_ok=True)

    np.save(target / "features.npy", fused)
    np.save(target / "labels.npy", labels)
    np.save(target / "clip_ids.npy", ids)

    print(
        f"{split.upper()}: {fused.shape} | "
        f"missing faces: {int((mask == 0).sum())} | SAVED"
    )

print("\nFUSION FEATURE PREPARATION COMPLETE")
