
from pathlib import Path
import csv
from collections import Counter

# ==========================================
# SYNAPSENSE DATASET CONFIGURATION
# ==========================================

DAISEE_ROOT = Path(
    r"C:\Users\yarra\Downloads\DAiSEE\DAiSEE"
)

DATASET_ROOT = DAISEE_ROOT / "DataSet"
OUTPUT_DIR = Path("datasets/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".avi", ".mp4"}

SPLITS = ["Train", "Validation", "Test"]


# ==========================================
# INDEX ALL AVAILABLE VIDEOS
# ==========================================

def index_videos():

    video_index = {}

    print("\nIndexing available video files...")

    for path in DATASET_ROOT.rglob("*"):

        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:

            key = path.name.lower()

            video_index.setdefault(key, []).append(path)

    print("Unique video filenames:", len(video_index))

    return video_index


# ==========================================
# PROCESS OFFICIAL SPLIT LISTS
# ==========================================

def process_split(split, video_index):

    split_file = DATASET_ROOT / f"{split}.txt"

    if not split_file.exists():

        print(f"\n{split}: Split list not found. Skipping.")
        return

    with open(split_file, "r", encoding="utf-8-sig") as file:

        filenames = [
            line.strip()
            for line in file
            if line.strip()
        ]

    records = []

    for filename in filenames:

        matches = video_index.get(filename.lower(), [])

        video_path = str(matches[0]) if len(matches) == 1 else ""

        if len(matches) > 1:
            status = "duplicate_match"
        elif len(matches) == 1:
            status = "available"
        else:
            status = "missing"

        records.append({
            "split": split,
            "filename": filename,
            "subject_id": filename[:6],
            "video_path": video_path,
            "status": status
        })

    output_file = OUTPUT_DIR / f"{split.lower()}_manifest.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "filename",
                "subject_id",
                "video_path",
                "status"
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    counts = Counter(row["status"] for row in records)

    print(f"\n{'=' * 45}")
    print(f"{split.upper()} DATASET SUMMARY")
    print(f"{'=' * 45}")

    print("Listed clips:", len(records))
    print("Available:", counts["available"])
    print("Missing:", counts["missing"])
    print("Duplicate matches:", counts["duplicate_match"])
    print("Unique participants:",
          len(set(row["subject_id"] for row in records)))

    print("Manifest saved:", output_file)


# ==========================================
# INSPECT ANNOTATION FILES
# ==========================================

def inspect_labels():

    labels_dir = DAISEE_ROOT / "Labels"

    for split in SPLITS:

        label_file = labels_dir / f"{split}Labels.csv"

        if not label_file.exists():
            continue

        with open(
            label_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            print(f"\n{split.upper()} ANNOTATIONS")
            print("Columns:", reader.fieldnames)

            first_row = next(reader, None)

            if first_row:
                print("Sample annotation:", first_row)


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 55)
    print("SYNAPSENSE - DATASET MANIFEST GENERATOR")
    print("=" * 55)

    video_index = index_videos()

    for split in SPLITS:
        process_split(split, video_index)

    inspect_labels()

    print("\nDataset manifest generation completed.")


if __name__ == "__main__":
    main()
