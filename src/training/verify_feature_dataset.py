
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    ROOT / "datasets" / "processed" /
    "features" / "mobilenet_v3_small"
)

MANIFEST_ROOT = ROOT / "datasets" / "processed"

SPLITS = {
    "train": 5357,
    "validation": 1429,
    "test": 1784,
}

LABEL_COLUMNS = [
    "Boredom",
    "Engagement",
    "Confusion",
    "Frustration",
]


def verify_split(split, expected_count):

    folder = FEATURE_ROOT / split

    features = np.load(folder / "features.npy", mmap_mode="r")
    labels = np.load(folder / "labels.npy")
    clip_ids = np.load(folder / "clip_ids.npy")
    completed = np.load(folder / "completed.npy")
    failed = np.load(folder / "failed.npy")

    manifest = pd.read_csv(
        MANIFEST_ROOT / f"final_{split}_manifest.csv",
        dtype={"clip_id": str},
    )

    expected_shape = (expected_count, 16, 576)

    checks = {
        "Feature shape": features.shape == expected_shape,
        "Label shape": labels.shape == (expected_count, 4),
        "Clip ID count": len(clip_ids) == expected_count,
        "Completed count": bool(completed.all()),
        "No failed videos": not bool(failed.any()),
        "Finite features": bool(np.isfinite(features).all()),
        "Valid label range": bool(
            np.isin(labels, [0, 1, 2, 3]).all()
        ),
        "Clip IDs match manifest": np.array_equal(
            clip_ids.astype(str),
            manifest["clip_id"].astype(str).to_numpy(),
        ),
        "Labels match manifest": np.array_equal(
            labels,
            manifest[LABEL_COLUMNS].to_numpy(dtype=np.int64),
        ),
    }

    print(f"\n{split.upper()} SPLIT")
    print("-" * 50)

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    return all(checks.values())


def main():

    print("\nSYNAPSENSE FEATURE DATASET VERIFICATION")
    print("=" * 55)

    results = {}

    for split, count in SPLITS.items():
        results[split] = verify_split(split, count)

    print("\n" + "=" * 55)

    if all(results.values()):
        print("ALL FEATURE DATASETS VERIFIED SUCCESSFULLY!")
    else:
        print("VERIFICATION FAILED. CHECK THE FAILED ITEMS.")

    print("=" * 55)


if __name__ == "__main__":
    main()
