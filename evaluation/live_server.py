"""FastAPI + WebSocket live dashboard server for streaming inference.

Replaces the old polling-based HTTP server with a push-based WebSocket
approach for more reliable real-time data delivery.

Usage (standalone test):
    python -m evaluation.live_server --port 8765
"""
from __future__ import annotations

import asyncio
import html
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _sanitize_value(v: Any) -> Any:
    """Replace NaN/Inf floats with None (becomes null in JSON)."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, list):
        return [_sanitize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _sanitize_value(val) for k, val in v.items()}
    return v


def _safe_json_dumps(obj: Any) -> str:
    """JSON serialize with NaN/Inf replaced by null."""
    return json.dumps(_sanitize_value(obj), separators=(",", ":"))


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages active WebSocket connections and broadcasts state."""

    def __init__(self) -> None:
        self._connections: List[WebSocket] = []
        self._lock = threading.Lock()
        self._last_state: Optional[Dict[str, Any]] = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        with self._lock:
            self._connections.append(websocket)
        # Send the latest state immediately so new clients see current data
        if self._last_state is not None:
            try:
                await websocket.send_text(json.dumps(self._last_state, separators=(",", ":")))
            except Exception:
                pass

    def disconnect(self, websocket: WebSocket) -> None:
        with self._lock:
            if websocket in self._connections:
                self._connections.remove(websocket)

    async def broadcast(self, state: Dict[str, Any]) -> None:
        self._last_state = state
        with self._lock:
            connections = list(self._connections)
        dead: List[WebSocket] = []
        message = _safe_json_dumps(state)
        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    if ws in self._connections:
                        self._connections.remove(ws)

    def broadcast_sync(self, state: Dict[str, Any]) -> None:
        """Thread-safe synchronous broadcast from the inference loop."""
        self._last_state = state
        with self._lock:
            connections = list(self._connections)
        if not connections:
            return
        message = json.dumps(state, separators=(",", ":"))
        dead: List[WebSocket] = []
        for ws in connections:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(ws.send_text(message), loop)
                else:
                    asyncio.run(ws.send_text(message))
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for ws in dead:
                    if ws in self._connections:
                        self._connections.remove(ws)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._connections)


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def _label_color(label: str) -> str:
    palette = {
        "other": "#94a3b8",
        "concentric": "#E8854C",
        "eccentric": "#4C9BE8",
        "db_bench_press": "#16a34a",
        "db_rdl": "#2563eb",
        "db_weighted_crunch": "#dc2626",
        "one_arm_db_row": "#f97316",
        "db_squat": "#ca8a04",
        "db_biceps_curl": "#db2777",
        "db_shoulder_press": "#0f766e",
        "db_triceps_curl": "#9333ea",
    }
    return palette.get(str(label), "#64748b")


def _build_dashboard_html(stream_id: str, sample_rate_hz: float, window_seconds: float, ws_port: int) -> str:
    """Generate the HTML dashboard that connects via WebSocket."""
    title = html.escape(stream_id)
    label_colors = json.dumps({k: _label_color(k) for k in [
        "other", "concentric", "eccentric",
        "db_bench_press", "db_rdl", "db_weighted_crunch",
        "one_arm_db_row", "db_squat", "db_biceps_curl",
        "db_shoulder_press", "db_triceps_curl",
    ]})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Live streaming - {title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; color: #111827; background: #ffffff; }}
    .top {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
    .pill {{ border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 10px; background: #f8fafc; }}
    .label {{ font-size: 12px; color: #64748b; }}
    .value {{ font-size: 16px; font-weight: 700; margin-left: 4px; }}
    .connected {{ color: #16a34a; }}
    .disconnected {{ color: #dc2626; }}
    canvas {{ width: 1280px; max-width: 100%; height: 680px; border: 1px solid #e2e8f0; display: block; }}
    #log {{ margin-top: 12px; font-size: 12px; color: #64748b; max-height: 80px; overflow-y: auto; }}
  </style>
</head>
<body>
  <h1>Live streaming: {title}</h1>
  <div class="top">
    <div class="pill"><span class="label">WS</span><span id="ws_status" class="value disconnected">connecting</span></div>
    <div class="pill"><span class="label">Status</span><span id="status" class="value">loading</span></div>
    <div class="pill"><span class="label">Time</span><span id="time" class="value">0.00s</span></div>
    <div class="pill"><span class="label">Online micro</span><span id="micro" class="value">-</span></div>
    <div class="pill"><span class="label">Online action</span><span id="macro" class="value">-</span></div>
    <div class="pill"><span class="label">Display reps</span><span id="drep" class="value">0</span></div>
    <div class="pill"><span class="label">Display action</span><span id="daction" class="value">pending</span></div>
    <div class="pill"><span class="label">GT</span><span id="gt" class="value">-</span></div>
  </div>
  <canvas id="plot" width="1280" height="680"></canvas>
  <div id="log"></div>
  <script>
    const sampleRate = {float(sample_rate_hz):.12f};
    const defaultWindowSamples = Math.max(10, Math.round({float(window_seconds):.6f} * sampleRate));
    const colors = {label_colors};
    const canvas = document.getElementById('plot');
    const ctx = canvas.getContext('2d');
    const wsStatusEl = document.getElementById('ws_status');
    const logEl = document.getElementById('log');

    function log(msg) {{
      const line = document.createElement('div');
      line.textContent = new Date().toLocaleTimeString() + ' ' + msg;
      logEl.prepend(line);
      while (logEl.children.length > 20) logEl.removeChild(logEl.lastChild);
    }}

    function color(label) {{ return colors[label] || '#64748b'; }}
    function norm(values, idxs) {{
      let min = Infinity, max = -Infinity;
      for (const i of idxs) {{ const v = values[i]; if (Number.isFinite(v)) {{ min = Math.min(min, v); max = Math.max(max, v); }} }}
      if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return {{min: -1, max: 1}};
      const pad = (max - min) * 0.12;
      return {{min: min - pad, max: max + pad}};
    }}
    function drawSeries(values, idxs, x0, y0, w, h, stroke) {{
      const r = norm(values, idxs);
      ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.beginPath();
      idxs.forEach((idx, j) => {{
        const x = x0 + (j / Math.max(1, idxs.length - 1)) * w;
        const y = y0 + h - ((values[idx] - r.min) / Math.max(1e-9, r.max - r.min)) * h;
        if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
    }}
    function drawBands(labels, idxs, x0, y, w, h) {{
      if (!idxs.length) return;
      let startJ = 0, cur = labels[idxs[0]];
      for (let j = 1; j <= idxs.length; j++) {{
        const lab = j < idxs.length ? labels[idxs[j]] : null;
        if (lab !== cur) {{
          const x = x0 + (startJ / Math.max(1, idxs.length - 1)) * w;
          const ww = Math.max(1, ((j - startJ) / Math.max(1, idxs.length - 1)) * w);
          ctx.fillStyle = color(cur); ctx.globalAlpha = cur === 'other' ? 0.18 : 0.72;
          ctx.fillRect(x, y, ww, h); ctx.globalAlpha = 1;
          startJ = j; cur = lab;
        }}
      }}
    }}
    function text(x, y, value, size = 13, weight = '500') {{
      ctx.font = `${{weight}} ${{size}}px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif`;
      ctx.fillStyle = '#111827'; ctx.fillText(value, x, y);
    }}

    function draw(state) {{
      const n = state.sample_idx.length;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!n) {{ text(40, 60, 'Waiting for samples...', 18, '700'); return; }}
      const end = n - 1;
      const start = Math.max(0, end - (state.window_samples || defaultWindowSamples) + 1);
      const idxs = [];
      for (let i = start; i <= end; i++) idxs.push(i);
      const x0 = 110, w = 1080;
      text(40, 52, `Live window ${{(state.sample_idx[start] / sampleRate).toFixed(1)}}s - ${{(state.sample_idx[end] / sampleRate).toFixed(1)}}s`, 18, '700');
      text(40, 104, 'GT micro', 14, '700'); ctx.strokeStyle = '#cbd5e1'; ctx.strokeRect(x0, 84, w, 24); drawBands(state.gt_micro_label, idxs, x0, 84, w, 24);
      text(40, 142, 'Pred micro', 14, '700'); ctx.strokeRect(x0, 122, w, 24); drawBands(state.online_micro_label, idxs, x0, 122, w, 24);
      text(40, 194, 'GT action', 14, '700'); ctx.strokeRect(x0, 174, w, 28); drawBands(state.gt_macro_label, idxs, x0, 174, w, 28);
      text(40, 236, 'Pred action', 14, '700'); ctx.strokeRect(x0, 216, w, 28); drawBands(state.online_macro_label, idxs, x0, 216, w, 28);
      text(40, 312, 'acc_mag', 14, '700'); ctx.strokeRect(x0, 285, w, 125); drawSeries(state.acc_mag, idxs, x0, 285, w, 125, '#2563eb');
      text(40, 477, 'gyro_mag', 14, '700'); ctx.strokeRect(x0, 450, w, 125); drawSeries(state.gyro_mag, idxs, x0, 450, w, 125, '#7c3aed');
      ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x0 + w, 76); ctx.lineTo(x0 + w, 585); ctx.stroke();
      const row = end;
      document.getElementById('status').textContent = state.done ? 'done' : 'running';
      document.getElementById('time').textContent = (state.sample_idx[row] / sampleRate).toFixed(2) + 's';
      document.getElementById('micro').textContent = state.online_micro_label[row] + ' ' + Number(state.online_micro_confidence[row]).toFixed(2);
      document.getElementById('macro').textContent = state.online_macro_label[row] + ' ' + Number(state.online_macro_confidence[row]).toFixed(2);
      document.getElementById('drep').textContent = String(state.display_rep_count[row]);
      document.getElementById('daction').textContent = state.display_action[row] + ' ' + (Number.isFinite(Number(state.display_action_confidence[row])) ? Number(state.display_action_confidence[row]).toFixed(2) : '-');
      document.getElementById('gt').textContent = state.gt_micro_label[row] + ' / ' + state.gt_macro_label[row];
    }}

    // --- WebSocket connection with auto-reconnect ---
    let ws = null;
    let reconnectTimer = null;
    let messageCount = 0;

    function connectWS() {{
      const wsUrl = `ws://${{window.location.hostname}}:${{window.location.port}}/ws`;
      log('Connecting to ' + wsUrl);
      ws = new WebSocket(wsUrl);
      ws.onopen = () => {{
        wsStatusEl.textContent = 'connected';
        wsStatusEl.className = 'value connected';
        log('WebSocket connected');
      }};
      ws.onmessage = (event) => {{
        messageCount++;
        try {{
          const state = JSON.parse(event.data);
          draw(state);
        }} catch(e) {{
          log('Parse error: ' + e.message);
        }}
      }};
      ws.onclose = (event) => {{
        wsStatusEl.textContent = 'disconnected';
        wsStatusEl.className = 'value disconnected';
        log('WebSocket closed (code=' + event.code + '). Reconnecting in 2s...');
        scheduleReconnect();
      }};
      ws.onerror = (event) => {{
        log('WebSocket error');
        ws.close();
      }};
    }}

    function scheduleReconnect() {{
      if (reconnectTimer) clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connectWS, 2000);
    }}

    connectWS();
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_manager: Optional[ConnectionManager] = None
_app: Optional[FastAPI] = None
_server_thread: Optional[threading.Thread] = None
_uvicorn_server: Optional[uvicorn.Server] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def create_app(stream_id: str, sample_rate_hz: float, window_seconds: float, ws_port: int) -> tuple[FastAPI, ConnectionManager]:
    """Create a FastAPI application with WebSocket endpoint."""
    app = FastAPI(title="Live Streaming Dashboard")
    manager = ConnectionManager()

    dashboard_html = _build_dashboard_html(stream_id, sample_rate_hz, window_seconds, ws_port)

    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard():
        return dashboard_html

    @app.get("/live_dashboard.html", response_class=HTMLResponse)
    async def get_dashboard_alias():
        return dashboard_html

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            while True:
                # Keep connection alive; client doesn't send data
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception:
            manager.disconnect(websocket)

    return app, manager


def _find_free_port() -> int:
    """Bind to port 0 to let the OS pick a free port, then release it."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_server(
    stream_id: str,
    sample_rate_hz: float,
    window_seconds: float,
    port: int = 8765,
) -> tuple[ConnectionManager, int, str]:
    """Start the FastAPI server in a background thread.

    Returns (manager, actual_port, live_url).
    """
    global _manager, _app, _server_thread, _uvicorn_server, _event_loop

    # Resolve port=0 up front so we always know the actual port
    actual_port = port if port != 0 else _find_free_port()

    app, manager = create_app(stream_id, sample_rate_hz, window_seconds, actual_port)
    _app = app
    _manager = manager

    started_event = threading.Event()

    @app.on_event("startup")
    async def _signal_ready():
        started_event.set()

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=actual_port,
        log_level="warning",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
    server = uvicorn.Server(config)
    _uvicorn_server = server

    def _run():
        global _event_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _event_loop = loop
        loop.run_until_complete(server.serve())

    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()

    # Wait for server to be ready
    if not started_event.wait(timeout=10.0):
        raise RuntimeError("FastAPI server failed to start within 10 seconds")

    live_url = f"http://127.0.0.1:{actual_port}/"
    print(f"[OK] Live WebSocket server started on port {actual_port}", flush=True)
    return manager, actual_port, live_url


def broadcast_state(manager: ConnectionManager, state: Dict[str, Any]) -> None:
    """Send state to all connected WebSocket clients from the inference thread.

    Fire-and-forget: does not block the inference loop.
    """
    global _event_loop
    if _event_loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(manager.broadcast(state), _event_loop)
    except Exception:
        pass


def stop_server() -> None:
    """Gracefully stop the FastAPI server."""
    global _uvicorn_server, _server_thread, _event_loop
    if _uvicorn_server is not None:
        _uvicorn_server.should_exit = True
    if _server_thread is not None:
        _server_thread.join(timeout=5.0)
    _uvicorn_server = None
    _server_thread = None
    _event_loop = None


# ---------------------------------------------------------------------------
# CLI test entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    manager, port, url = start_server("test_stream", 50.0, 15.0, args.port)
    print(f"Dashboard URL: {url}")
    print("Press Ctrl+C to stop")

    # Simulate some data
    import numpy as np
    try:
        for i in range(5000):
            state = {
                "stream_id": "test_stream",
                "sample_rate_hz": 50.0,
                "window_samples": 750,
                "done": False,
                "sample_idx": list(range(max(0, i - 749), i + 1)),
                "acc_mag": [float(np.sin(x * 0.1) + np.random.randn() * 0.1) for x in range(max(0, i - 749), i + 1)],
                "gyro_mag": [float(np.cos(x * 0.05) + np.random.randn() * 0.1) for x in range(max(0, i - 749), i + 1)],
                "online_micro_label": ["concentric" if x % 100 < 50 else "eccentric" for x in range(max(0, i - 749), i + 1)],
                "online_micro_confidence": [0.85] * len(list(range(max(0, i - 749), i + 1))),
                "online_macro_label": ["db_rdl"] * len(list(range(max(0, i - 749), i + 1))),
                "online_macro_confidence": [0.9] * len(list(range(max(0, i - 749), i + 1))),
                "display_rep_count": [i // 100] * len(list(range(max(0, i - 749), i + 1))),
                "display_action": ["db_rdl"] * len(list(range(max(0, i - 749), i + 1))),
                "display_action_confidence": [0.88] * len(list(range(max(0, i - 749), i + 1))),
                "display_action_locked": [True] * len(list(range(max(0, i - 749), i + 1))),
                "gt_micro_label": ["concentric" if x % 100 < 50 else "eccentric" for x in range(max(0, i - 749), i + 1)],
                "gt_macro_label": ["db_rdl"] * len(list(range(max(0, i - 749), i + 1))),
            }
            broadcast_state(manager, state)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        stop_server()
