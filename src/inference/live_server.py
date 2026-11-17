"""SYNAPSENSE HTTP inference service.

Run locally with ``python src/inference/live_server.py``. Camera frames are
processed synchronously in memory and are not written to disk.
"""
from __future__ import annotations

import io
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.extract_facial_features import extract_features
from src.training.train_synapsense_attention import SynapSenseAttention

APP_ROOT = ROOT / "app"
CHECKPOINT = ROOT / "models" / "visual_attention" / "best_attention_lstm.pth"
NORMALIZATION = ROOT / "models" / "visual_attention" / "feature_normalization.npz"
FACE_TASK = ROOT / "models" / "mediapipe" / "face_landmarker.task"
LABELS = ("Boredom", "Engagement", "Confusion", "Frustration")
DEVICE = torch.device("cpu")
SEQUENCE_LENGTH = 16
MAX_FRAME_BYTES = 4_000_000
MOBILENET_V3_SMALL_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
DEFAULT_SESSION_TTL = 1800
DEFAULT_MAX_SESSIONS = 8
torch.set_num_threads(2)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("synapsense")

visual_model = None
feature_transform = None
face_landmarker = None
model_error = None


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def server_host() -> str:
    host = os.environ.get("HOST", "127.0.0.1").strip()
    if not host:
        raise ValueError("HOST cannot be empty")
    return host


def server_port() -> int:
    return env_int("PORT", 8000, 1, 65535)


def allowed_origins() -> frozenset[str]:
    configured = os.environ.get("CORS_ALLOWED_ORIGINS", "")
    defaults = {"http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5173", "http://127.0.0.1:5173"}
    origins = defaults | {item.strip().rstrip("/") for item in configured.split(",") if item.strip()}
    for origin in origins:
        parsed = urlparse(origin)
        if origin == "*" or parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or "*" in parsed.netloc:
            raise ValueError(f"Invalid CORS origin {origin!r}; configure exact http(s) origins only")
        if parsed.scheme == "http" and (parsed.hostname or "").lower() not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Non-local CORS origins must use HTTPS")
    return frozenset(origins)


@dataclass
class SessionState:
    frames: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=SEQUENCE_LENGTH))
    last_active: float = field(default_factory=time.monotonic)
    last_predictions: dict | None = None


class SessionStore:
    """Bounded, TTL-cleaned session buffers; inference and mutations are serialized."""
    def __init__(self, max_sessions: int = DEFAULT_MAX_SESSIONS, ttl_seconds: int = DEFAULT_SESSION_TTL, clock=time.monotonic):
        if max_sessions < 1 or ttl_seconds < 1:
            raise ValueError("Session limits must be positive")
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.sessions: dict[str, SessionState] = {}
        self.lock = threading.RLock()

    def cleanup(self) -> int:
        now = self.clock()
        expired = [key for key, state in self.sessions.items() if now - state.last_active >= self.ttl_seconds]
        for key in expired:
            del self.sessions[key]
        return len(expired)

    def start(self, session_id: str) -> bool:
        with self.lock:
            self.cleanup()
            if session_id in self.sessions:
                return False
            if len(self.sessions) >= self.max_sessions:
                raise OverflowError("Server is at its active session limit")
            self.sessions[session_id] = SessionState(last_active=self.clock())
            return True

    def stop(self, session_id: str) -> None:
        with self.lock:
            self.cleanup()
            self.sessions.pop(session_id, None)


sessions = SessionStore(
    max_sessions=env_int("MAX_SESSIONS", DEFAULT_MAX_SESSIONS, 1, 256),
    ttl_seconds=env_int("SESSION_TTL_SECONDS", DEFAULT_SESSION_TTL, 30, 86400),
)


def load_runtime() -> None:
    global visual_model, feature_transform, face_landmarker, model_error
    try:
        if not all(path.exists() for path in (CHECKPOINT, NORMALIZATION, FACE_TASK)):
            missing = [str(p.relative_to(ROOT)) for p in (CHECKPOINT, NORMALIZATION, FACE_TASK) if not p.exists()]
            raise FileNotFoundError("Missing required inference assets: " + ", ".join(missing))
        weights = MobileNet_V3_Small_Weights.DEFAULT
        cache_root = Path(torch.hub.get_dir()) / "checkpoints"
        cached_weight = cache_root / Path(urlparse(weights.url).path).name
        log.info("Loading MobileNetV3-Small weights from %s (downloads verified weight to this cache if absent)", cached_weight)
        # TorchVision's DEFAULT weight metadata uses check_hash=True for the official 047dcff4 filename.
        encoder = mobilenet_v3_small(weights=weights)
        digest = hashlib.sha256(cached_weight.read_bytes()).hexdigest()
        if digest != MOBILENET_V3_SMALL_SHA256:
            raise RuntimeError(f"MobileNetV3-Small weight SHA-256 mismatch: {digest}")
        encoder.classifier = nn.Identity()
        encoder.eval().to(DEVICE)
        payload = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        model = SynapSenseAttention().to(DEVICE)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        norm = np.load(NORMALIZATION)
        mean = torch.from_numpy(norm["mean"].astype(np.float32)).view(1, 1, -1)
        std = torch.from_numpy(norm["std"].astype(np.float32)).view(1, 1, -1)
        if mean.shape[-1] != 576 or std.shape != mean.shape:
            raise ValueError(f"Unexpected normalization shape: mean={tuple(mean.shape)}, std={tuple(std.shape)}")
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_TASK)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            output_face_blendshapes=False, output_facial_transformation_matrixes=False,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        visual_model, feature_transform, face_landmarker = (model, (encoder, weights.transforms(), mean, std), landmarker)
        model_error = None
        log.info("Loaded champion checkpoint %s (validation Macro-F1 %.4f)", CHECKPOINT.name, payload.get("best_validation_macro_f1", float("nan")))
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"
        log.exception("Unable to initialize inference runtime")


def process_frame(raw: bytes, state: SessionState) -> dict:
    if visual_model is None or feature_transform is None or face_landmarker is None:
        raise RuntimeError(model_error or "Inference model is not loaded")
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image_np = np.asarray(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    result = face_landmarker.detect(mp_image)
    landmarks = []
    metrics = None
    if result.face_landmarks:
        face = result.face_landmarks[0]
        metrics = {name: float(value) for name, value in extract_features(face).items()}
        landmarks = [[float(point.x), float(point.y)] for point in face]

    encoder, transform, mean, std = feature_transform
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        visual_feature = encoder(tensor).cpu().numpy().astype(np.float32)[0]
    if visual_feature.shape != (576,):
        raise ValueError(f"Expected a 576-value visual feature, received {visual_feature.shape}")

    state.frames.append(visual_feature)
    classes = None
    if len(state.frames) == SEQUENCE_LENGTH:
        sequence = torch.from_numpy(np.stack(tuple(state.frames))).unsqueeze(0)
        sequence = (sequence - mean) / std
        with torch.inference_mode():
            logits = visual_model(sequence)
            classes = {name: int(output.argmax(dim=1).item()) for name, output in zip(LABELS, logits)}
        state.last_predictions = classes
    return {
        "face_detected": bool(result.face_landmarks), "landmarks": landmarks, "metrics": metrics,
        "frames_ready": len(state.frames), "predictions": classes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def valid_session_id(value) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9_-]{16,128}", value))


class Handler(SimpleHTTPRequestHandler):
    server_version = "SYNAPSENSE/2.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_ROOT), **kwargs)

    def _cors(self, origin: str | None) -> bool:
        if origin and origin in allowed_origins():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            return True
        return False

    def _json(self, status: int, value: dict) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors(self.headers.get("Origin"))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        origin = self.headers.get("Origin")
        if not origin or origin not in allowed_origins():
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(204)
        self._cors(origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Session-Id")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            with sessions.lock:
                sessions.cleanup()
                count = len(sessions.sessions)
                frames_ready = max((len(s.frames) for s in sessions.sessions.values()), default=0)
            self._json(200, {"status": "ok", "processing_mode": "local", "model_loaded": visual_model is not None,
                             "model_error": model_error, "frames_ready": frames_ready, "session_active": count > 0,
                             "active_sessions": count})
            return
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"detail": "Invalid Content-Length"})
            return
        if length < 0 or length > MAX_FRAME_BYTES:
            self._json(413, {"detail": "Request exceeds the 4 MB frame limit"})
            return
        body = self.rfile.read(length)
        if path in {"/api/session/start", "/api/session/stop"}:
            try:
                payload = json.loads(body or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json(400, {"detail": "Invalid session request"})
                return
            session_id = payload.get("session_id") if isinstance(payload, dict) else None
            if not valid_session_id(session_id):
                self._json(400, {"detail": "session_id must be a unique 16-128 character identifier"})
                return
            if path.endswith("/start"):
                try:
                    created = sessions.start(session_id)
                except OverflowError as exc:
                    self._json(503, {"detail": str(exc)})
                    return
                if not created:
                    self._json(409, {"detail": "Session identifier is already active"})
                    return
                self._json(200, {"status": "started", "processing_mode": "local"})
                return
            sessions.stop(session_id)
            self._json(200, {"status": "stopped"})
            return
        if path == "/api/frame":
            session_id = self.headers.get("X-Session-Id", "")
            if not valid_session_id(session_id):
                self._json(400, {"detail": "A valid X-Session-Id header is required"})
                return
            if self.headers.get_content_type() != "image/jpeg" or not body.startswith(b"\xff\xd8"):
                self._json(415, {"detail": "Frame body must be a JPEG image"})
                return
            try:
                Image.open(io.BytesIO(body)).verify()
            except (UnidentifiedImageError, OSError, ValueError):
                self._json(400, {"detail": "Invalid JPEG frame"})
                return
            with sessions.lock:
                sessions.cleanup()
                state = sessions.sessions.get(session_id)
                if state is None:
                    self._json(409, {"detail": "Start this session before sending frames"})
                    return
                state.last_active = sessions.clock()
                try:
                    value = process_frame(body, state)
                except Exception as exc:
                    log.exception("Frame processing failed")
                    self._json(503, {"detail": f"{type(exc).__name__}: {exc}"})
                    return
                self._json(200, value)
                return
        self._json(404, {"detail": "Not found"})

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.client_address[0], fmt % args)


def main() -> None:
    load_runtime()
    host, port = server_host(), server_port()
    allowed_origins()  # validate configuration before opening a public listener
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    log.info("SYNAPSENSE HTTP app listening on %s:%s", host, port)
    log.info("Frames are processed in memory; configure CORS_ALLOWED_ORIGINS for the frontend origin")
    if model_error:
        log.error("Model status: %s", model_error)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopping server")
    finally:
        server.server_close()
        if face_landmarker is not None:
            face_landmarker.close()


if __name__ == "__main__":
    main()
