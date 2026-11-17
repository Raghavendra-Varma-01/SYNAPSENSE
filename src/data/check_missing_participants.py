
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\Users\yarra\Downloads\DAiSEE\DAiSEE")
TRAIN_DIR = ROOT / "DataSet" / "Train"
TRAIN_LIST = ROOT / "DataSet" / "Train.txt"

listed = [
    line.strip()
    for line in TRAIN_LIST.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

expected_participants = {
    name[:6] for name in listed
}

actual_videos = [
    p for p in TRAIN_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in {".avi", ".mp4"}
]

actual_participants = {
    p.parent.parent.name for p in actual_videos
}

print("=" * 55)
print("SYNAPSENSE - PARTICIPANT FOLDER AUDIT")
print("=" * 55)

print("Expected participants:", len(expected_participants))
print("Participants with extracted videos:", len(actual_participants))

missing = sorted(expected_participants - actual_participants)

print("Participants with no extracted videos:", len(missing))

print("\nMissing participant IDs:")
for participant in missing:
    print(participant)

print("\nActual video counts by extension:")
print(Counter(p.suffix.lower() for p in actual_videos))

print("=" * 55)

