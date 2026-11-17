import io
import json
import socket
import threading
import time
from urllib.parse import urlsplit
from collections import deque
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from src.inference import live_server as server


def jpeg_bytes():
    out = io.BytesIO()
    Image.new("RGB", (8, 8), (25, 80, 140)).save(out, format="JPEG")
    return out.getvalue()


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://synapsense.example")
    original = server.sessions
    server.sessions = server.SessionStore(max_sessions=3, ttl_seconds=30)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)
    server.sessions = original


def request(base, path, *, method="GET", payload=None, headers=None, data=None):
    import urllib.error
    import urllib.request

    body = data if data is not None else (json.dumps(payload).encode() if payload is not None else None)
    req = urllib.request.Request(base + path, data=body, headers=headers or {}, method=method)
    try:
        response = urllib.request.urlopen(req)
    except urllib.error.HTTPError as exc:
        response = exc
    return response.status, response.headers, json.loads(response.read() or b"{}")


def start(base, sid):
    return request(base, "/api/session/start", method="POST", payload={"session_id": sid}, headers={"Content-Type": "application/json"})


def test_health_schema(api, monkeypatch):
    monkeypatch.setattr(server, "visual_model", object())
    status, _, body = request(api, "/health")
    assert status == 200
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["processing_mode"] == "local"
    assert body["active_sessions"] == 0 and body["session_active"] is False


def test_session_creation_duplicate_isolation_and_stop(api, monkeypatch):
    calls = []

    def fake_process(raw, state):
        state.frames.append(np.full((576,), len(raw), dtype=np.float32))
        calls.append((id(state), len(raw), len(state.frames)))
        return {"face_detected": False, "landmarks": [], "metrics": None, "frames_ready": len(state.frames), "predictions": None, "timestamp": "test"}

    monkeypatch.setattr(server, "process_frame", fake_process)
    a, b = "session-client-A-0001", "session-client-B-0002"
    assert start(api, a)[0] == 200
    assert start(api, b)[0] == 200
    assert start(api, a)[0] == 409
    raw = jpeg_bytes()
    for _ in range(2):
        status, _, result = request(api, "/api/frame", method="POST", data=raw, headers={"Content-Type": "image/jpeg", "X-Session-Id": a})
        assert status == 200 and result["frames_ready"] in (1, 2)
    status, _, result = request(api, "/api/frame", method="POST", data=raw, headers={"Content-Type": "image/jpeg", "X-Session-Id": b})
    assert status == 200 and result["frames_ready"] == 1
    assert len({item[0] for item in calls}) == 2
    assert len(server.sessions.sessions[a].frames) == 2
    assert len(server.sessions.sessions[b].frames) == 1
    assert request(api, "/api/session/stop", method="POST", payload={"session_id": a}, headers={"Content-Type": "application/json"})[0] == 200
    assert a not in server.sessions.sessions and b in server.sessions.sessions
    assert request(api, "/api/frame", method="POST", data=raw, headers={"Content-Type": "image/jpeg", "X-Session-Id": a})[0] == 409


@pytest.mark.parametrize("payload", [{}, {"session_id": "x"}, {"session_id": "*" * 20}])
def test_session_creation_requires_valid_unique_id(api, payload):
    status, _, body = request(api, "/api/session/start", method="POST", payload=payload, headers={"Content-Type": "application/json"})
    assert status == 400 and "session_id" in body["detail"]


def test_frame_request_validation_and_unknown_session(api):
    sid = "session-frame-client-001"
    assert start(api, sid)[0] == 200
    status, _, _ = request(api, "/api/frame", method="POST", data=b"not jpeg", headers={"Content-Type": "image/jpeg", "X-Session-Id": sid})
    assert status == 415
    status, _, _ = request(api, "/api/frame", method="POST", data=jpeg_bytes(), headers={"Content-Type": "image/jpeg", "X-Session-Id": "session-other-client-001"})
    assert status == 409
    status, _, _ = request(api, "/api/frame", method="POST", data=jpeg_bytes(), headers={"Content-Type": "text/plain", "X-Session-Id": sid})
    assert status == 415


def test_frame_request_size_limit(api):
    parsed = urlsplit(api)
    with socket.create_connection((parsed.hostname, parsed.port), timeout=3) as client:
        client.sendall((f"POST /api/frame HTTP/1.1\r\nHost: {parsed.hostname}\r\nContent-Type: image/jpeg\r\n"
                        f"Content-Length: {server.MAX_FRAME_BYTES + 1}\r\nConnection: close\r\n\r\n").encode())
        response = client.recv(1024).decode("latin-1")
    assert "413" in response.split("\r\n", 1)[0]


def test_expired_sessions_are_removed(api):
    now = [1.0]
    store = server.SessionStore(max_sessions=2, ttl_seconds=10, clock=lambda: now[0])
    assert store.start("session-expire-me-001")
    now[0] = 11.0
    assert store.cleanup() == 1
    assert store.sessions == {}


def test_session_capacity_is_bounded():
    store = server.SessionStore(max_sessions=1, ttl_seconds=100)
    assert store.start("session-capacity-A-001")
    with pytest.raises(OverflowError):
        store.start("session-capacity-B-002")


def test_cors_allows_exact_configured_and_local_origins(api):
    status, headers, _ = request(api, "/health", headers={"Origin": "https://synapsense.example"})
    assert status == 200 and headers.get("Access-Control-Allow-Origin") == "https://synapsense.example"
    status, headers, _ = request(api, "/health", headers={"Origin": "http://localhost:5173"})
    assert status == 200 and headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"
    status, headers, _ = request(api, "/health", headers={"Origin": "https://evil.example"})
    assert status == 200 and headers.get("Access-Control-Allow-Origin") is None
    status, headers, _ = request(api, "/health", method="OPTIONS", headers={"Origin": "https://synapsense.example"})
    assert status == 204 and headers.get("Access-Control-Allow-Origin") == "https://synapsense.example"
    status, _, _ = request(api, "/health", method="OPTIONS", headers={"Origin": "https://preview.vercel.app"})
    assert status == 403


def test_cors_rejects_wildcard(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    with pytest.raises(ValueError, match="exact"):
        server.allowed_origins()


def test_host_and_port_keep_local_defaults(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    assert server.server_host() == "127.0.0.1"
    assert server.server_port() == 8000
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9001")
    assert server.server_host() == "0.0.0.0" and server.server_port() == 9001


def test_model_loading_failure_is_reported(monkeypatch):
    monkeypatch.setattr(server, "visual_model", None)
    monkeypatch.setattr(server, "mobilenet_v3_small", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("weight download unavailable")))
    server.load_runtime()
    assert server.visual_model is None
    assert "weight download unavailable" in server.model_error
