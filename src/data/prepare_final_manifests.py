
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\yarra\Downloads\DAiSEE\DAiSEE")
DATASET = ROOT / "DataSet"
LABELS = ROOT / "Labels"

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "datasets" / "processed"
OUTPUT.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".avi", ".mp4"}

def normalize_id(value):
    value = str(value).strip().replace("\\", "/")
    return Path(value).stem.lower()

print("\nSYNAPSENSE — FINAL MANIFEST PREPARATION")
print("=" * 55)

# Index every recovered video, preserving its official folder.
video_index = {}

for path in DATASET.rglob("*"):
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        key = path.stem.lower()

        if key in video_index:
            raise RuntimeError(
                f"Duplicate video ID detected: {key}\n"
                f"{video_index[key]}\n{path}"
            )

        video_index[key] = path

print(f"Indexed videos: {len(video_index)}")

summary = []

for split in ["Train", "Validation", "Test"]:
    print(f"\nProcessing {split}...")

    split_file = DATASET / f"{split}.txt"
    label_file = LABELS / f"{split}Labels.csv"

    if not split_file.exists() or not label_file.exists():
        raise FileNotFoundError(
            f"Missing split or annotation file for {split}"
        )

    split_ids = {
        normalize_id(line)
        for line in split_file.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    }

    labels = pd.read_csv(label_file)
    labels.columns = labels.columns.str.strip()

    required = [
        "ClipID",
        "Boredom",
        "Engagement",
        "Confusion",
        "Frustration"
    ]

    missing_columns = set(required) - set(labels.columns)

    if missing_columns:
        raise ValueError(
            f"{split}: Missing annotation columns: {missing_columns}"
        )

    labels["clip_id"] = labels["ClipID"].map(normalize_id)

    if labels["clip_id"].duplicated().any():
        raise RuntimeError(f"Duplicate annotation IDs in {split}")

    label_map = labels.set_index("clip_id")[required[1:]]

    rows = []

    for clip_id in sorted(split_ids):
        video_path = video_index.get(clip_id)

        if video_path is None or clip_id not in label_map.index:
            continue

        row = {
            "clip_id": clip_id,
            "video_path": str(video_path.resolve()),
            "split": split.lower()
        }

        row.update(label_map.loc[clip_id].to_dict())
        rows.append(row)

    manifest = pd.DataFrame(rows)

    output_file = OUTPUT / f"final_{split.lower()}_manifest.csv"
    manifest.to_csv(output_file, index=False)

    count = len(manifest)

    summary.append({
        "Split": split,
        "Usable pairs": count,
        "Output": str(output_file)
    })

    print(f"Usable video-label pairs: {count}")
    print(f"Saved: {output_file}")

print("\n" + "=" * 55)
print("FINAL MANIFEST SUMMARY")
print("=" * 55)

for item in summary:
    print(f"{item['Split']}: {item['Usable pairs']}")

total = sum(item["Usable pairs"] for item in summary)

print("-" * 55)
print(f"TOTAL USABLE PAIRS: {total}")

report = pd.DataFrame(summary)
report.to_csv(OUTPUT / "final_manifest_summary.csv", index=False)

print("\nMANIFEST PREPARATION COMPLETED")
