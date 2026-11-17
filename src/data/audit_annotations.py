
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DAISEE_ROOT = Path(
    r"C:\Users\yarra\Downloads\DAiSEE\DAiSEE"
)

MANIFEST_DIR = PROJECT_ROOT / "datasets" / "processed"
LABEL_DIR = DAISEE_ROOT / "Labels"

SPLITS = ["train", "validation", "test"]


def audit_split(split):

    print("\n" + "=" * 55)
    print(f"{split.upper()} ANNOTATION AUDIT")
    print("=" * 55)

    manifest_path = MANIFEST_DIR / f"{split}_manifest.csv"
    label_path = LABEL_DIR / f"{split.title()}Labels.csv"

    manifest = pd.read_csv(manifest_path)
    labels = pd.read_csv(label_path)

    # Normalize whitespace in annotation column names
    labels.columns = labels.columns.str.strip()

    # Normalize video identifiers
    manifest["filename"] = (
        manifest["filename"].astype(str).str.strip().str.lower()
    )

    labels["ClipID"] = (
        labels["ClipID"].astype(str).str.strip().str.lower()
    )

    label_columns = [
        "Boredom",
        "Engagement",
        "Confusion",
        "Frustration"
    ]

    # Validate annotation values
    for column in label_columns:
        labels[column] = pd.to_numeric(
            labels[column], errors="coerce"
        )

    # Detect duplicate annotation IDs
    duplicates = labels["ClipID"].duplicated().sum()

    if duplicates:
        raise ValueError(
            f"{duplicates} duplicate annotation IDs in {split}"
        )

    # Match annotations to official split entries
    merged = manifest.merge(
        labels,
        left_on="filename",
        right_on="ClipID",
        how="left",
        validate="one_to_one",
        indicator=True
    )

    matched = merged["_merge"] == "both"
    available = merged["status"] == "available"

    # A video is usable only when both video and labels exist
    usable = matched & available

    print("Official split entries:", len(manifest))
    print("Annotation records:", len(labels))
    print("Matched annotations:", int(matched.sum()))
    print("Unmatched annotations:", int((~matched).sum()))
    print("Available videos:", int(available.sum()))
    print("Usable video + annotation pairs:", int(usable.sum()))

    print("\nLabel value distribution:")

    for column in label_columns:
        print(f"\n{column}:")
        print(labels[column].value_counts(dropna=False).sort_index())

    merged["annotation_status"] = merged["_merge"].map({
        "both": "matched",
        "left_only": "missing_annotation"
    })

    output = merged.drop(
        columns=["ClipID", "_merge"]
    )

    output["usable"] = usable

    output_path = (
        MANIFEST_DIR / f"{split}_annotated_manifest.csv"
    )

    output.to_csv(output_path, index=False)

    print("\nSaved:", output_path)


def main():

    for split in SPLITS:
        audit_split(split)

    print("\nANNOTATION AUDIT COMPLETED")


if __name__ == "__main__":
    main()

