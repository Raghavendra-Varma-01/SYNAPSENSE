
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

MANIFESTS = {
    "train": PROCESSED_DIR / "final_train_manifest.csv",
    "validation": PROCESSED_DIR / "final_validation_manifest.csv",
    "test": PROCESSED_DIR / "final_test_manifest.csv",
}

OUTPUT = PROCESSED_DIR / "frame_pilot"

FRAMES_PER_VIDEO = 16
IMAGE_SIZE = 224
VIDEOS_PER_SPLIT = 3


# --------------------------------------------------
# FRAME EXTRACTION FUNCTION
# --------------------------------------------------

def extract_frames(video_path, output_dir):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return 0, "Could not open video"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return 0, "Invalid frame count"

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        FRAMES_PER_VIDEO,
        dtype=int
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_frames = 0

    for frame_number, frame_index in enumerate(frame_indices):

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        frame_path = output_dir / f"frame_{frame_number:02d}.jpg"

        saved = cv2.imwrite(
            str(frame_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if saved:
            saved_frames += 1

    cap.release()

    return saved_frames, "Success"


# --------------------------------------------------
# MAIN PILOT
# --------------------------------------------------

def main():

    print("\nSYNAPSENSE FRAME EXTRACTION PILOT")
    print("=" * 55)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    report = []

    for split, manifest_path in MANIFESTS.items():

        print(f"\n{split.upper()} SPLIT")
        print("-" * 55)

        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {manifest_path}")
            continue

        # Important: preserve clip IDs as strings
        df = pd.read_csv(
            manifest_path,
            dtype={"clip_id": str}
        )

        if df.empty:
            print("WARNING: Manifest contains no records.")
            continue

        sample = df.head(VIDEOS_PER_SPLIT)

        for _, row in sample.iterrows():

            clip_id = str(row["clip_id"]).strip()

            # Handle IDs accidentally read with a .0 suffix
            if clip_id.endswith(".0"):
                clip_id = clip_id[:-2]

            video_path = Path(row["video_path"])

            clip_output = OUTPUT / split / clip_id

            print(f"\nProcessing: {clip_id}")

            if not video_path.exists():
                print("ERROR: Video file not found")

                report.append({
                    "split": split,
                    "clip_id": clip_id,
                    "video_path": str(video_path),
                    "frames_saved": 0,
                    "status": "Video not found",
                })
                continue

            frames_saved, status = extract_frames(
                video_path,
                clip_output
            )

            print(f"Frames saved: {frames_saved}/{FRAMES_PER_VIDEO}")
            print(f"Status: {status}")

            report.append({
                "split": split,
                "clip_id": clip_id,
                "video_path": str(video_path),
                "frames_saved": frames_saved,
                "status": status,
            })

    # --------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------

    report_df = pd.DataFrame(report)

    csv_path = OUTPUT / "pilot_report.csv"
    json_path = OUTPUT / "pilot_report.json"

    report_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4
        )

    total_videos = len(report)
    total_frames = int(report_df["frames_saved"].sum()) if total_videos else 0

    print("\n" + "=" * 55)
    print("PILOT EXTRACTION SUMMARY")
    print("=" * 55)

    print(f"Videos attempted: {total_videos}")
    print(f"Frames successfully saved: {total_frames}")
    print(f"Expected maximum frames: {total_videos * FRAMES_PER_VIDEO}")

    print(f"\nCSV report: {csv_path}")
    print(f"JSON report: {json_path}")

    if total_videos == 9 and total_frames == 144:
        print("\nPILOT COMPLETED SUCCESSFULLY!")
    else:
        print("\nPILOT COMPLETED WITH WARNINGS. Review the reports.")

    print("=" * 55)


if __name__ == "__main__":
    main()
