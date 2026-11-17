
from pathlib import Path
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
FRAME_ROOT = PROCESSED_DIR / "frames"

OUTPUT_DIR = PROCESSED_DIR / "features" / "mobilenet_v3_small"

MANIFESTS = {
    "train": PROCESSED_DIR / "final_train_manifest.csv",
    "validation": PROCESSED_DIR / "final_validation_manifest.csv",
    "test": PROCESSED_DIR / "final_test_manifest.csv",
}

LABEL_COLUMNS = [
    "Boredom",
    "Engagement",
    "Confusion",
    "Frustration",
]

FRAMES_PER_VIDEO = 16
FEATURE_DIM = 576

# Process 16 frames per video as one batch.
BATCH_SIZE = 16

# Save progress after this many videos.
SAVE_INTERVAL = 25


# ============================================================
# DEVICE AND CPU SETTINGS
# ============================================================

torch.set_num_threads(4)

DEVICE = torch.device("cpu")


# ============================================================
# LOAD PRETRAINED MOBILENETV3
# ============================================================

def load_model():

    print("\nLoading pretrained MobileNetV3-Small...")

    weights = MobileNet_V3_Small_Weights.DEFAULT

    try:
        model = mobilenet_v3_small(weights=weights)

    except Exception as error:
        raise RuntimeError(
            "\nCould not load pretrained ImageNet weights.\n"
            "Check your internet connection and run again.\n"
            f"Original error: {error}"
        )

    model.classifier = nn.Identity()

    model = model.to(DEVICE)
    model.eval()

    transform = weights.transforms()

    print("Model loaded successfully.")
    print("Feature dimension:", FEATURE_DIM)
    print("Device:", DEVICE)

    return model, transform


# ============================================================
# EXTRACT FEATURES FROM ONE VIDEO
# ============================================================

def extract_video_features(model, transform, clip_id, split):

    clip_dir = FRAME_ROOT / split / clip_id

    tensors = []

    for frame_number in range(FRAMES_PER_VIDEO):

        frame_path = clip_dir / f"frame_{frame_number:02d}.jpg"

        if not frame_path.exists():
            raise FileNotFoundError(
                f"Missing frame: {frame_path}"
            )

        with Image.open(frame_path) as image:
            image = image.convert("RGB")
            tensors.append(transform(image))

    batch = torch.stack(tensors).to(DEVICE)

    with torch.inference_mode():

        features = model(batch)

    features = features.cpu().numpy().astype(np.float32)

    if features.shape != (FRAMES_PER_VIDEO, FEATURE_DIM):
        raise ValueError(
            f"Unexpected feature shape: {features.shape}"
        )

    return features


# ============================================================
# PROCESS ONE DATASET SPLIT
# ============================================================

def process_split(split, manifest_path, model, transform):

    print("\n" + "=" * 65)
    print(f"PROCESSING {split.upper()} FEATURES")
    print("=" * 65)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    df = pd.read_csv(manifest_path, dtype={"clip_id": str})

    total_videos = len(df)

    split_output = OUTPUT_DIR / split
    split_output.mkdir(parents=True, exist_ok=True)

    features_path = split_output / "features.npy"
    labels_path = split_output / "labels.npy"
    clip_ids_path = split_output / "clip_ids.npy"
    completed_path = split_output / "completed.npy"
    failed_path = split_output / "failed.npy"

    expected_shape = (
        total_videos,
        FRAMES_PER_VIDEO,
        FEATURE_DIM,
    )

    # Create or resume feature storage.
    if features_path.exists() and completed_path.exists():

        features = np.load(features_path, mmap_mode="r+")

        completed = np.load(completed_path)
        failed = np.load(failed_path)

        if features.shape != expected_shape:
            raise ValueError(
                f"Existing feature shape mismatch: {features.shape}"
            )

        if len(completed) != total_videos:
            raise ValueError("Existing progress size mismatch.")

        print("Resuming previous feature extraction.")

    else:

        features = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=np.float32,
            shape=expected_shape,
        )

        completed = np.zeros(total_videos, dtype=bool)
        failed = np.zeros(total_videos, dtype=bool)

        labels = df[LABEL_COLUMNS].to_numpy(dtype=np.int64)

        np.save(labels_path, labels)

        clip_ids = df["clip_id"].astype(str).to_numpy(dtype=str)

        np.save(clip_ids_path, clip_ids)

        np.save(completed_path, completed)
        np.save(failed_path, failed)

    successful = int(completed.sum())
    failures = int(failed.sum())

    start_time = time.time()

    for index, row in df.iterrows():

        if completed[index]:
            continue

        clip_id = str(row["clip_id"]).strip()

        if clip_id.endswith(".0"):
            clip_id = clip_id[:-2]

        try:

            video_features = extract_video_features(
                model,
                transform,
                clip_id,
                split,
            )

            features[index] = video_features

            completed[index] = True
            failed[index] = False

            successful += 1

        except Exception as error:

            failed[index] = True
            failures += 1

            print(
                f"\nERROR: {clip_id} | "
                f"{type(error).__name__}: {error}"
            )

        processed = index + 1

        if processed % SAVE_INTERVAL == 0 or processed == total_videos:

            features.flush()

            np.save(completed_path, completed)
            np.save(failed_path, failed)

            elapsed = time.time() - start_time

            print(
                f"\rProgress: {processed}/{total_videos} | "
                f"Successful: {successful} | "
                f"Failed: {failures} | "
                f"Time: {elapsed / 60:.1f} min",
                end="",
                flush=True,
            )

    features.flush()

    np.save(completed_path, completed)
    np.save(failed_path, failed)

    print("\n\nSPLIT COMPLETED")
    print("Total videos:", total_videos)
    print("Successful:", int(completed.sum()))
    print("Failed:", int(failed.sum()))

    return {
        "split": split,
        "total_videos": total_videos,
        "successful": int(completed.sum()),
        "failed": int(failed.sum()),
        "feature_shape": list(expected_shape),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nSYNAPSENSE CNN FEATURE EXTRACTION")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, transform = load_model()

    summaries = []

    total_start = time.time()

    for split, manifest_path in MANIFESTS.items():

        result = process_split(
            split,
            manifest_path,
            model,
            transform,
        )

        summaries.append(result)

    summary = {
        "model": "MobileNetV3-Small",
        "pretrained_weights": "ImageNet1K_V1",
        "feature_dimension": FEATURE_DIM,
        "frames_per_video": FRAMES_PER_VIDEO,
        "device": str(DEVICE),
        "splits": summaries,
        "total_elapsed_minutes": round(
            (time.time() - total_start) / 60,
            2,
        ),
    }

    summary_path = OUTPUT_DIR / "feature_extraction_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("FINAL FEATURE EXTRACTION SUMMARY")
    print("=" * 65)

    print(json.dumps(summary, indent=4))

    print("\nSaved to:", OUTPUT_DIR)
    print("\nCNN FEATURE EXTRACTION FINISHED")


if __name__ == "__main__":
    main()

