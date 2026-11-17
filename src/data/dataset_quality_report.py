
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\yarra\Downloads\DAiSEE\DAiSEE")
DATASET = ROOT / "DataSet"
LABELS = ROOT / "Labels"

for split in ["Train", "Validation", "Test"]:

    print("\n" + "=" * 55)
    print(split.upper(), "DATA QUALITY REPORT")
    print("=" * 55)

    split_file = DATASET / f"{split}.txt"
    label_file = LABELS / f"{split}Labels.csv"

    split_ids = {
        x.strip().lower()
        for x in split_file.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if x.strip()
    }

    labels = pd.read_csv(label_file)
    labels.columns = labels.columns.str.strip()

    labels["ClipID"] = (
        labels["ClipID"].astype(str).str.strip().str.lower()
    )

    label_ids = set(labels["ClipID"])

    available_ids = {
        p.name.lower()
        for p in DATASET.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".avi", ".mp4"}
    }

    # Only videos belonging to this split
    available_ids &= split_ids

    print("Split entries:", len(split_ids))
    print("Annotation records:", len(label_ids))
    print("Available videos:", len(available_ids))

    print("Split IDs without labels:",
          len(split_ids - label_ids))

    print("Annotations outside split list:",
          len(label_ids - split_ids))

    print("Annotations without videos:",
          len(label_ids - available_ids))

    print("Videos without annotations:",
          len(available_ids - label_ids))

    print("Usable video-label pairs:",
          len(split_ids & label_ids & available_ids))

print("\nQUALITY REPORT COMPLETED")
