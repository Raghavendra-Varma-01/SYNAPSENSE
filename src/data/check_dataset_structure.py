
from pathlib import Path

ROOT = Path(r"C:\Users\yarra\Downloads\DAiSEE\DAiSEE")

print("=" * 55)
print("SYNAPSENSE - DATASET STRUCTURE DIAGNOSTIC")
print("=" * 55)

print("\nDAiSEE root exists:", ROOT.exists())

dataset = ROOT / "DataSet"
labels = ROOT / "Labels"

print("\nDataset folder:", dataset)
print("Dataset exists:", dataset.exists())

if dataset.exists():
    print("\nFolders inside DataSet:")

    for folder in sorted(dataset.iterdir()):
        if folder.is_dir():
            videos = list(folder.rglob("*.avi"))
            print(f"{folder.name}: {len(videos)} AVI files")

print("\nLabels folder:", labels)
print("Labels exists:", labels.exists())

if labels.exists():
    print("\nFiles inside Labels:")

    for file in sorted(labels.iterdir()):
        if file.is_file():
            print(file.name)

print("\n" + "=" * 55)
