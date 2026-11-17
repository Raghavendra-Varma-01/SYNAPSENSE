
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]

FRAME_ROOT = ROOT / "datasets/processed/frames"
OUTPUT_ROOT = ROOT / "datasets/processed/facial_features"
MODEL_PATH = ROOT / "models/mediapipe/face_landmarker.task"

SPLITS = ["train", "validation", "test"]

# MediaPipe eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmark indices
MOUTH = [61, 291, 13, 14, 78, 308]


def distance(a, b):
    return np.linalg.norm(a - b)


def ear(points, indices):
    p = points[indices]
    vertical = distance(p[1], p[5]) + distance(p[2], p[4])
    horizontal = 2 * distance(p[0], p[3])
    return float(vertical / horizontal) if horizontal else 0.0


def mar(points):
    p = points
    vertical = distance(p[13], p[14])
    horizontal = distance(p[61], p[291])
    return float(vertical / horizontal) if horizontal else 0.0


def extract_features(landmarks):
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    left_ear = ear(points, LEFT_EYE)
    right_ear = ear(points, RIGHT_EYE)

    return {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "mean_ear": (left_ear + right_ear) / 2,
        "mar": mar(points),
        "face_width": distance(points[234], points[454]),
        "face_height": distance(points[10], points[152]),
        "head_yaw_proxy": float(points[454][2] - points[234][2]),
        "head_pitch_proxy": float(points[10][2] - points[152][2]),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        num_faces=1
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:

        for split in SPLITS:
            split_dir = FRAME_ROOT / split
            output_dir = OUTPUT_ROOT / split
            output_dir.mkdir(parents=True, exist_ok=True)

            rows = []

            clip_dirs = sorted(
                p for p in split_dir.iterdir() if p.is_dir()
            )

            for clip_dir in tqdm(
                clip_dirs, desc=f"Processing {split}"
            ):
                frame_files = sorted(clip_dir.glob("frame_*.jpg"))

                for frame_path in frame_files:
                    image = cv2.imread(str(frame_path))

                    if image is None:
                        continue

                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )

                    result = landmarker.detect(mp_image)

                    row = {
                        "clip_id": clip_dir.name,
                        "frame": frame_path.stem,
                        "face_detected": 0
                    }

                    if result.face_landmarks:
                        row.update(
                            extract_features(result.face_landmarks[0])
                        )
                        row["face_detected"] = 1

                    rows.append(row)

            df = pd.DataFrame(rows)
            output_path = output_dir / "facial_features.csv"
            df.to_csv(output_path, index=False)

            print(
                f"{split}: {len(df)} frames | "
                f"{df['face_detected'].sum()} faces detected"
            )
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
