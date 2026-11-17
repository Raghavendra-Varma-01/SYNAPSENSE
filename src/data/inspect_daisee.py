
from pathlib import Path
import cv2

# Update this path if your DAiSEE folder is elsewhere
DATASET_ROOT = Path(
    r"C:\Users\yarra\Downloads\DAiSEE\DAiSEE\DataSet"
)

video_path = (
    DATASET_ROOT
    / "Train"
    / "110010"
    / "1100102003"
    / "1100102003.avi"
)

print("=" * 50)
print("SYNAPSENSE - DAiSEE VIDEO INSPECTION")
print("=" * 50)

if not video_path.exists():
    print("ERROR: Video not found!")
    print("Expected path:", video_path)
else:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0

        print("Video:", video_path.name)
        print("FPS:", fps)
        print("Total frames:", frame_count)
        print("Resolution:", f"{width} x {height}")
        print("Duration:", round(duration, 2), "seconds")

        success, frame = cap.read()

        if success:
            print("First frame successfully decoded!")
            cv2.imshow("SYNAPSENSE - DAiSEE Frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Could not read the first frame.")

        cap.release()

print("=" * 50)