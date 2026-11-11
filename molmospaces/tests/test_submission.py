import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from websockets.sync.client import connect

from batbatbat.probe import fake_observation
from batbatbat.harness.server import Service
from batbatbat.harness.wire import pack, unpack
from batbatbat.submission import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/submission.yaml"
PROMPT_SHA = "f8cf317b38bf94ac786a06a213b051a68e94f50d2199c758ea909e6e5d041229"


def config():
    return load_config(CONFIG, {"BATBATBAT_LLM_BASE_URL": "https://example.invalid/v1", "BATBATBAT_LLM_API_KEY": "test"})[0]


def test_frozen_policy_matches_recorded_submission(monkeypatch):
    monkeypatch.setenv("BATBATBAT_LLM_REASONING_EFFORT", "high")
    service = Service(config(), lambda **row: None, warmup=False)
    assert service.metadata["planner_prompt_sha256"] == PROMPT_SHA
    assert service.metadata["knowledge"]["knowledge_sha256"] == "7bcdff0fd4a21edb"
    assert "M13" in service.metadata["knowledge"]["knowledge_notes"]
    assert "M12" not in service.metadata["knowledge"]["knowledge_notes"]
    assert service.metadata["planner"]["max_calls"] == 200
    assert service.metadata["planner"]["effort"] == "high"
    assert (service.horizon, service.move_speed_multiplier, service.gripper_settle_steps) == (8, 4.0, 8)


@pytest.mark.parametrize("env", [
    {}, {"BATBATBAT_LLM_BASE_URL": "https://example.invalid/v1"},
    {"BATBATBAT_LLM_BASE_URL": "http://example.invalid/v1", "BATBATBAT_LLM_API_KEY": "test"},
    {"BATBATBAT_LLM_BASE_URL": "https://user:secret@example.invalid/v1", "BATBATBAT_LLM_API_KEY": "test"},
    {"BATBATBAT_LLM_BASE_URL": "https://example.invalid/v1?key=secret", "BATBATBAT_LLM_API_KEY": "test"},
])
def test_no_implicit_credentials_or_unsafe_remote_url(env):
    with pytest.raises(ValueError):
        load_config(CONFIG, env)


def test_provider_failure_never_echoes_response_secrets(monkeypatch):
    from batbatbat.harness.responses_api import _ResponsesBackend
    import requests
    class Reply:
        status_code = 401
        text = "sensitive-provider-response"
    monkeypatch.setattr(requests, "post", lambda *a, **kw: Reply())
    backend = _ResponsesBackend("gpt-6.1-sol", "https://example.invalid/v1", "secret")
    with pytest.raises(RuntimeError) as error:
        backend.ping()
    assert "401" in str(error.value)
    assert "sensitive-provider-response" not in str(error.value)


def test_http_to_websocket_and_episode_isolation(tmp_path):
    requests_seen = []
    key = "synthetic-test-key-not-a-credential"
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            assert self.path == "/v1/responses"
            assert self.headers["Authorization"] == "Bearer " + key
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests_seen.append(body)
            reply = {"id": "synthetic", "status": "completed", "output": [
                {"type": "function_call", "call_id": "hold-1", "name": "hold", "arguments": "{}"}
            ]}
            raw = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("BATBATBAT_")}
    key_file = tmp_path / "llm-key"
    key_file.write_text(key + "\n")
    log = tmp_path / "decisions.jsonl"
    with (tmp_path / "server.log").open("w+") as output:
        process = subprocess.Popen([sys.executable, "-m", "batbatbat.submission", "--config", str(CONFIG),
                                    "--port", str(port), "--log", str(log),
                                    "--llm-base-url", f"http://127.0.0.1:{http.server_port}/v1",
                                    "--llm-api-key-file", str(key_file), "--llm-model", "gpt-6.1-sol"], env=env, stdout=output, stderr=output)
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    ws = connect(f"ws://127.0.0.1:{port}", compression=None, max_size=64*1024*1024)
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        output.seek(0)
                        pytest.fail(output.read())
                    time.sleep(.1)
            with ws:
                metadata = unpack(ws.recv(timeout=10))
                assert metadata["planner_prompt_sha256"] == PROMPT_SHA
                obs = fake_observation()
                ws.send(pack(obs))
                first = unpack(ws.recv(timeout=10))
                assert "action" in first, first
                assert first["action"].shape == (8, 8)
                assert np.isfinite(first["action"]).all()
                calls = len(requests_seen)
                ws.send(pack(obs))
                np.testing.assert_array_equal(unpack(ws.recv(timeout=10))["action"], first["action"])
                assert len(requests_seen) == calls  # duplicate request is served from the session cache
                obs2 = fake_observation()
                obs2["prompt"] = "Independent synthetic episode: hold position."
                ws.send(pack(obs2))
                assert unpack(ws.recv(timeout=10))["action"].shape == (8, 8)
                assert len(requests_seen) > calls
            body = requests_seen[-1]
            assert body["reasoning"]["effort"] == "high"
            assert body["store"] is False and body["max_output_tokens"] == 4000
            assert body["model"] == "gpt-6.1-sol"
            assert hashlib.sha256(body["instructions"].encode()).hexdigest() == PROMPT_SHA
            assert any(c.get("type") == "input_image" for item in body["input"] for c in item.get("content", []))
            assert key not in log.read_text()
        finally:
            process.terminate()
            process.wait(timeout=10)
            http.shutdown()
            http.server_close()
