
import sys
import torch
import pandas as pd
import cv2

print("\nSYNAPSENSE MODEL ENVIRONMENT CHECK")
print("=" * 55)

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Pandas:", pd.__version__)
print("OpenCV:", cv2.__version__)

print("\nChecking Torchvision...")

try:
    import torchvision
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    print("Torchvision:", torchvision.__version__)

    print("\nChecking MobileNetV3-Small...")

    model = mobilenet_v3_small(weights=None)

    print("Model initialization: SUCCESS")
    print("Classifier input features:", model.classifier[0].in_features)

    print("\nChecking pretrained weights availability...")

    weights = MobileNet_V3_Small_Weights.DEFAULT
    print("Weights identifier:", weights)

except Exception as e:
    print("Torchvision/model check failed:")
    print(type(e).__name__, str(e))

print("\nChecking dataset manifests...")

from pathlib import Path

root = Path(__file__).resolve().parents[2]
processed = root / "datasets" / "processed"

for filename in [
    "final_train_manifest.csv",
    "final_validation_manifest.csv",
    "final_test_manifest.csv",
]:
    path = processed / filename

    if path.exists():
        df = pd.read_csv(path)
        print(f"{filename}: {len(df)} rows")
        print("Columns:", list(df.columns))
    else:
        print(f"MISSING: {filename}")

print("\nENVIRONMENT CHECK COMPLETED")
