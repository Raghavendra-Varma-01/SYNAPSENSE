# SYNAPSENSE implementation audit

## Current architecture (audited before implementation)

- The workspace contains DAiSEE data preparation, frame extraction, MobileNetV3-Small feature extraction, facial landmark feature extraction, training, and evaluation code under `src/`.
- Prepared manifests and frame/features are under `datasets/processed/`.
- The champion temporal-attention BiLSTM checkpoint and normalization statistics are under `models/visual_attention/`; the MediaPipe Face Landmarker asset is under `models/mediapipe/`.
- There is no existing browser application, HTTP API, session manager, or deployment configuration. `app/components/` is empty.
- There is no standalone logo asset in the project. The user supplied a logo reference image in the conversation.

## Existing working features

- DAiSEE datasets are prepared into train, validation, and test manifests; sampled frames and MobileNetV3-Small feature arrays are present.
- `src/preprocessing/extract_cnn_features.py` extracts 16 RGB frames per clip with the torchvision MobileNetV3-Small ImageNet transforms, producing 576-dimensional features.
- `src/training/train_synapsense_attention.py` defines the four-head, 4-class BiLSTM with temporal attention (576 input dimensions, hidden size 128) and stores the best checkpoint and feature mean/std.
- `src/features/extract_facial_features.py` defines the existing 478-landmark-derived EAR, MAR, face geometry, and head-pose proxy calculations.
- `models/visual_attention/training_history.json` records a best validation Macro-F1 of 0.2982 (epoch 6).
- The MediaPipe task asset, model checkpoint, normalization data, and pretrained MobileNetV3 weights are present locally.

## Gaps found in the initial audit

- Responsive dashboard, navigation, brand asset integration, and neural background.
- Local HTTP inference service, session lifecycle, health status, and browser camera integration.
- On-device browser landmark overlay and live facial metrics.
- Prediction history, charts, replay controls, and a conceptual brain visualization.
- Deployment and run instructions.

## Files planned for modification

- Add `app/index.html`, `app/styles.css`, and `app/app.js` for the interface and browser camera/session controls.
- Add `src/inference/live_server.py` for localhost health/session/frame inference using the existing champion pipeline.
- Add `README.md` for local startup and deployment limitations.
- Leave datasets, extracted features, trained checkpoints, and training code untouched.

## Dependencies

- Reuse the installed Python environment: PyTorch, torchvision, NumPy, Pillow, OpenCV, and MediaPipe are present.
- Use Python's standard-library HTTP server for localhost transport; no frontend or web framework dependency is required.
- Use browser Canvas and SVG for the background, landmark overlay, and conceptual visualization; no frontend package install is needed.

## Implementation order

1. Record this audit and implementation plan.
2. Build the brand-aligned responsive application shell, navigation, neural background, and honest empty states.
3. Add localhost webcam processing with MediaPipe and the existing 16-frame champion model.
4. Connect actual session history to the NEUROMAP, temporal chart, and timeline.
5. Document local startup, privacy behavior, and deployment constraints; verify the integrated path.

## Scientific and operational constraints

- Model outputs remain ordinal classes 0–3; no confidence scores or fabricated demonstration data are shown.
- The brain graphic is explicitly an educational conceptual mapping, not measured neural activation.
- Webcam frames are posted only to the local backend, held in a bounded in-memory window, and never written to disk.
- The supplied emblem was extracted from the reference image into `app/assets/synapsense-emblem.png`; the wordmark is rendered as text to keep it readable at small sizes.
- No cloud deployment can provide local-only webcam processing. A public frontend can be deployed separately, but live inference requires a reachable backend and an explicit cloud privacy mode.

## Implementation status

- Added a responsive dashboard, the local emblem and favicon, sidebar navigation, visible focus styles, reduced-motion handling, and an animated canvas background.
- Added a browser webcam client and a localhost-only standard-library HTTP server. The server uses existing MediaPipe feature calculations and the champion BiLSTM with saved normalization and torchvision transforms.
- Added empty-state charts, actual session prediction history, playback controls, conceptual network filters and region details, facial metrics, and JSON export.
- Added local startup and deployment/privacy guidance in the root README.

## Remaining limitations

- NEUROMAP is an interactive SVG side-view illustration with conceptual selectable regions. There is no 3D anatomical brain asset in the workspace, so realistic 3D anatomy, cutaway mode, and true 360-degree mesh rotation are not complete.
- Live transport uses same-origin HTTP polling at about one frame per second, not WebSockets. A rolling 16-frame input is supported; its spacing differs from DAiSEE clip sampling and is documented as experimental.
- Browser camera permission and mobile device behavior still need verification on the target devices.
- Public deployment is not prepared for live inference. A cloud backend needs a deliberate privacy/security design and matching disclosure; the current backend is local only.
