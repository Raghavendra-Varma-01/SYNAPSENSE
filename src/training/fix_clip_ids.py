
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    ROOT / "datasets" / "processed" /
    "features" / "mobilenet_v3_small"
)

SPLITS = ["train", "validation", "test"]

for split in SPLITS:

    path = FEATURE_ROOT / split / "clip_ids.npy"

    # Load the existing object array
    clip_ids = np.load(path, allow_pickle=True)

    # Convert to a safe Unicode string array
    clip_ids = np.asarray(clip_ids, dtype=str)

    # Save in standard NumPy string format
    np.save(path, clip_ids, allow_pickle=False)

    print(
        f"{split.upper()}: "
        f"{len(clip_ids)} clip IDs converted successfully"
    )

print("\nALL CLIP ID FILES FIXED!")

