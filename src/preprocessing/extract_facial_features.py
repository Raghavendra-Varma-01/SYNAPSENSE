
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[2]
FRAME_ROOT = ROOT / "datasets/processed/frames"
OUTPUT_ROOT = ROOT / "datasets/processed/facial_features"

SPLITS = ["train", "validation", "test"]

# MediaPipe Face Mesh eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

mp_face_mesh = mp.solutions.face_mesh


def eye_aspect_ratio(landmarks, indices):
    points = np.array([
        [landmarks[i].x, landmarks[i].y] for i in indices
    ])

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-8)


def extract_features(image, face_mesh):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark

    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)

    # Normalized face geometry
    points = np.array([[p.x, p.y, p.z] for p in landmarks],
                      dtype=np.float32)

    return {
        "left_ear": float(left_ear),
        "right_ear": float(right_ear),
        "mean_ear": float((left_ear + right_ear) / 2),
        "face_center_x": float(points[:, 0].mean()),
        "face_center_y": float(points[:, 1].mean()),
        "face_center_z": float(points[:, 2].mean()),
        "face_width": float(points[:, 0].max() - points[:, 0].min()),
        "face_height": float(points[:, 1].max() - points[:, 1].min()),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f"\nProcessing {split}...")

        split_root = FRAME_ROOT / split
        records = []

        clip_folders = sorted(
            p for p in split_root.iterdir() if p.is_dir()
        )

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:

            for clip_index, clip_folder in enumerate(clip_folders, 1):
                frame_files = sorted(clip_folder.glob("frame_*.jpg"))

                for frame_file in frame_files:
                    image = cv2.imread(str(frame_file))

                    if image is None:
                        continue

                    features = extract_features(image, face_mesh)

                    if features is None:
                        continue

                    features["clip_id"] = clip_folder.name
                    features["frame"] = frame_file.stem
                    records.append(features)

                if clip_index % 250 == 0:
                    print(f"Processed {clip_index}/{len(clip_folders)} clips")

        output_file = OUTPUT_ROOT / f"{split}_facial_features.csv"
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)

        summary = {
            "split": split,
            "clips": len(clip_folders),
            "detected_frames": len(df),
            "unique_clips_detected": int(df["clip_id"].nunique()) if not df.empty else 0,
            "frames_without_detected_face": (
                len(clip_folders) * 16 - len(df)
            ),
        }

        with open(
            OUTPUT_ROOT / f"{split}_summary.json", "w"
        ) as f:
            json.dump(summary, f, indent=4)

        print(summary)

    print("\nFACIAL FEATURE EXTRACTION COMPLETED")


if __name__ == "__main__":
    main()
