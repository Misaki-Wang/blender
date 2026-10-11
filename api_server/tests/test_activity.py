import sys
import time

from fastapi.testclient import TestClient

from api_server.app import Settings, create_app


def setup_server(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_TIMEOUT_SECONDS", raising=False)
    executable = tmp_path / "fake-codex"
    executable.write_text(f"#!{sys.executable}\n" + r'''
import json, pathlib, subprocess, sys, time
from uuid import uuid4
args = sys.argv[1:]
prompt = sys.stdin.read()
print(json.dumps({"type": "thread.started", "thread_id": str(uuid4())}), flush=True)
if prompt == "crash":
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    sys.exit(7)
release = pathlib.Path(__file__).parent / "release"
while not release.exists():
    print(json.dumps({"type": "item.started", "item": {"type": "command_execution"}}), flush=True)
    time.sleep(0.05)
pathlib.Path(args[args.index("--output-last-message") + 1]).write_text("DONE")
print(json.dumps({"type": "turn.completed", "usage": {}}), flush=True)
''')
    executable.chmod(0o700)
    settings = Settings(runtime=tmp_path / "runtime", codex_bin=str(executable),
                        default_model="fake", token="")
    assert settings.timeout is None
    return create_app(settings)


def test_live_process_activity_and_terminal_status(tmp_path, monkeypatch):
    with TestClient(setup_server(tmp_path, monkeypatch)) as client:
        assert client.get("/v1/capabilities").json()["execution_timeout_seconds"] is None
        job = client.post("/v1/jobs", json={"prompt": "work"}).json()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            current = client.get(f"/v1/jobs/{job['id']}").json()
            if current["activity"]["event_count"] >= 2:
                break
            time.sleep(0.02)
        assert current["status"] == "running"
        activity = current["activity"]
        assert activity["worker_alive"] and activity["process_alive"]
        assert activity["pid"] > 0 and activity["elapsed_seconds"] >= 0
        assert activity["last_event"] == "item.started"
        batch = client.post("/v1/jobs/query", json={"job_ids": [job["id"]]}).json()
        assert batch["jobs"][0]["activity"]["process_alive"] is True
        (tmp_path / "release").touch()
        result = client.get(f"/v1/jobs/{job['id']}", params={"wait_seconds": 5}).json()
        assert result["status"] == "completed"


def test_process_exit_is_detected_even_with_inherited_open_pipes(tmp_path, monkeypatch):
    with TestClient(setup_server(tmp_path, monkeypatch)) as client:
        job = client.post("/v1/jobs", json={"prompt": "crash"}).json()
        result = client.get(f"/v1/jobs/{job['id']}", params={"wait_seconds": 5}).json()
        assert result["status"] == "failed"
        assert result["status_code"] == 502
        assert client.get("/v1/queue").json()["pending"] == 0
