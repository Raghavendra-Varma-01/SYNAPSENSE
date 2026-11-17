
from pathlib import Path
import json
import time

import cv2
import numpy as np
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

MANIFESTS = {
    "train": PROCESSED_DIR / "final_train_manifest.csv",
    "validation": PROCESSED_DIR / "final_validation_manifest.csv",
    "test": PROCESSED_DIR / "final_test_manifest.csv",
}

OUTPUT_DIR = PROCESSED_DIR / "frames"

FRAMES_PER_VIDEO = 16
IMAGE_SIZE = 224
JPEG_QUALITY = 90

# Set to None to process every usable video.
MAX_VIDEOS_PER_SPLIT = None


# ==================================================
# VIDEO FRAME EXTRACTION
# ==================================================

def extract_frames(video_path, output_dir):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return False, 0, "Could not open video"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return False, 0, "Invalid frame count"

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
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        if saved:
            saved_frames += 1

    cap.release()

    if saved_frames == FRAMES_PER_VIDEO:
        return True, saved_frames, "Success"

    return False, saved_frames, "Incomplete frame extraction"


# ==================================================
# CHECK EXISTING OUTPUT
# ==================================================

def is_already_processed(output_dir):

    if not output_dir.exists():
        return False

    frames = list(output_dir.glob("frame_*.jpg"))

    if len(frames) != FRAMES_PER_VIDEO:
        return False

    return all(
        (output_dir / f"frame_{i:02d}.jpg").exists()
        for i in range(FRAMES_PER_VIDEO)
    )


# ==================================================
# PROCESS ONE SPLIT
# ==================================================

def process_split(split, manifest_path, all_results):

    print("\n" + "=" * 65)
    print(f"PROCESSING {split.upper()} SPLIT")
    print("=" * 65)

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return

    df = pd.read_csv(manifest_path, dtype={"clip_id": str})

    if MAX_VIDEOS_PER_SPLIT is not None:
        df = df.head(MAX_VIDEOS_PER_SPLIT)

    total_videos = len(df)

    print(f"Videos in manifest: {total_videos}")

    split_output = OUTPUT_DIR / split
    split_output.mkdir(parents=True, exist_ok=True)

    successful = 0
    skipped = 0
    failed = 0

    start_time = time.time()

    for index, row in df.iterrows():

        clip_id = str(row["clip_id"]).strip()

        if clip_id.endswith(".0"):
            clip_id = clip_id[:-2]

        video_path = Path(row["video_path"])
        clip_output = split_output / clip_id

        if is_already_processed(clip_output):
            skipped += 1
            successful += 1
            continue

        if not video_path.exists():
            status = "Video file not found"
            frames_saved = 0
            failed += 1

        else:
            ok, frames_saved, status = extract_frames(
                video_path,
                clip_output
            )

            if ok:
                successful += 1
            else:
                failed += 1

        result = {
            "split": split,
            "clip_id": clip_id,
            "video_path": str(video_path),
            "frames_saved": frames_saved,
            "status": status,
        }

        all_results.append(result)

        # Save progress after every video
        if (successful + failed) % 25 == 0:
            pd.DataFrame(all_results).to_csv(
                OUTPUT_DIR / "extraction_progress.csv",
                index=False
            )

        completed = index + 1

        if completed % 50 == 0 or completed == total_videos:

            elapsed = time.time() - start_time

            print(
                f"\rProgress: {completed}/{total_videos} | "
                f"Success: {successful} | "
                f"Failed: {failed} | "
                f"Skipped: {skipped} | "
                f"Time: {elapsed / 60:.1f} min",
                end="",
                flush=True
            )

    print("\n\nSplit completed.")
    print(f"Successful: {successful}")
    print(f"Skipped (already processed): {skipped}")
    print(f"Failed: {failed}")


# ==================================================
# MAIN
# ==================================================

def main():

    print("\nSYNAPSENSE FULL FRAME EXTRACTION")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    progress_file = OUTPUT_DIR / "extraction_progress.csv"

    # Load existing progress if available
    if progress_file.exists():
        previous = pd.read_csv(
            progress_file,
            dtype={"clip_id": str}
        )
        all_results = previous.to_dict("records")
        print(f"Previous logged records: {len(all_results)}")

    start = time.time()

    for split, manifest_path in MANIFESTS.items():
        process_split(split, manifest_path, all_results)

        pd.DataFrame(all_results).to_csv(
            progress_file,
            index=False
        )

    # Final report
    report_df = pd.DataFrame(all_results)

    report_df.to_csv(
        OUTPUT_DIR / "final_extraction_report.csv",
        index=False
    )

    summary = {
        "frames_per_video": FRAMES_PER_VIDEO,
        "image_size": IMAGE_SIZE,
        "total_logged_videos": len(report_df),
        "successful": int(
            (report_df["status"] == "Success").sum()
        ) if not report_df.empty else 0,
        "failed": int(
            (report_df["status"] != "Success").sum()
        ) if not report_df.empty else 0,
        "elapsed_minutes": round(
            (time.time() - start) / 60, 2
        ),
    }

    with open(
        OUTPUT_DIR / "extraction_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("FULL EXTRACTION SUMMARY")
    print("=" * 65)

    print(json.dumps(summary, indent=4))

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nEXTRACTION PROCESS FINISHED")


if __name__ == "__main__":
    main()

