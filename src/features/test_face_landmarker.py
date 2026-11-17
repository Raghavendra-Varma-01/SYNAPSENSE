import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

IMAGE_PATH = (
    "datasets/processed/frames/train/"
    "1100011002/frame_00.jpg"
)

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
)

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="models/mediapipe/face_landmarker.task"
    ),
    num_faces=1
)

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        print("Face detected!")
        print("Landmark count:", len(result.face_landmarks[0]))
    else:
        print("No face detected in this frame.")