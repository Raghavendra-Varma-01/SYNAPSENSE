
from pathlib import Path
import json

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
FRAME_ROOT = PROCESSED_DIR / "frames"

MANIFESTS = {
    "train": PROCESSED_DIR / "final_train_manifest.csv",
    "validation": PROCESSED_DIR / "final_validation_manifest.csv",
    "test": PROCESSED_DIR / "final_test_manifest.csv",
}

EXPECTED_FRAMES = 16
EXPECTED_SIZE = (224, 224)


def verify_video_frames(split, clip_id):

    clip_dir = FRAME_ROOT / split / clip_id

    if not clip_dir.exists():
        return False, 0, "Missing folder"

    frame_paths = [
        clip_dir / f"frame_{i:02d}.jpg"
        for i in range(EXPECTED_FRAMES)
    ]

    missing = [p.name for p in frame_paths if not p.exists()]

    if missing:
        return False, len(frame_paths) - len(missing), (
            "Missing frames: " + ", ".join(missing)
        )

    for frame_path in frame_paths:

        image = cv2.imread(str(frame_path))

        if image is None:
            return False, EXPECTED_FRAMES, (
                f"Unreadable image: {frame_path.name}"
            )

        height, width = image.shape[:2]

        if (width, height) != EXPECTED_SIZE:
            return False, EXPECTED_FRAMES, (
                f"Wrong dimensions: {width}x{height}"
            )

    return True, EXPECTED_FRAMES, "Verified"


def main():

    print("\nSYNAPSENSE FRAME INTEGRITY VERIFICATION")
    print("=" * 65)

    all_results = []

    for split, manifest_path in MANIFESTS.items():

        print(f"\nVERIFYING {split.upper()}")
        print("-" * 65)

        df = pd.read_csv(manifest_path, dtype={"clip_id": str})

        verified = 0
        failed = 0

        for index, row in df.iterrows():

            clip_id = str(row["clip_id"]).strip()

            if clip_id.endswith(".0"):
                clip_id = clip_id[:-2]

            ok, frame_count, status = verify_video_frames(
                split, clip_id
            )

            if ok:
                verified += 1
            else:
                failed += 1

            all_results.append({
                "split": split,
                "clip_id": clip_id,
                "frames_found": frame_count,
                "status": status,
            })

            if (index + 1) % 250 == 0 or index + 1 == len(df):
                print(
                    f"\rChecked: {index + 1}/{len(df)} | "
                    f"Verified: {verified} | Failed: {failed}",
                    end="",
                    flush=True
                )

        print(f"\n{split.upper()} verified: {verified}")
        print(f"{split.upper()} failed: {failed}")

    report = pd.DataFrame(all_results)

    report_path = FRAME_ROOT / "frame_integrity_report.csv"
    report.to_csv(report_path, index=False)

    summary = {
        "total_videos": len(report),
        "verified_videos": int(
            (report["status"] == "Verified").sum()
        ),
        "failed_videos": int(
            (report["status"] != "Verified").sum()
        ),
        "expected_frames_per_video": EXPECTED_FRAMES,
        "expected_dimensions": "224x224",
        "total_expected_frames": len(report) * EXPECTED_FRAMES,
    }

    summary_path = FRAME_ROOT / "frame_integrity_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("FINAL INTEGRITY SUMMARY")
    print("=" * 65)
    print(json.dumps(summary, indent=4))

    if summary["failed_videos"] == 0:
        print("\nALL FRAMES VERIFIED SUCCESSFULLY!")
    else:
        print("\nWARNING: Some videos require investigation.")

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
