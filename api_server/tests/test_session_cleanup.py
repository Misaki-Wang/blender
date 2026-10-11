import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api_server.app import create_app
from api_server import backend
from api_server.backend import conversation_session_home
from api_server.session_cleanup import delete_session_files
from test_api import settings
from test_queue import finished, submit, wait_for


@pytest.fixture(autouse=True)
def isolated_auth(tmp_path, monkeypatch):
    auth = tmp_path / "fake-auth.json"
    auth.write_text("{}")
    monkeypatch.delenv("API_SERVER_TOKEN", raising=False)
    monkeypatch.delenv("CODEX_GPU_SANDBOX_CONFIG", raising=False)
    monkeypatch.setenv("CODEX_AUTH_FILE", str(auth))


def rollout(home, thread_id, *, archived=False):
    directory = home / ("archived_sessions" if archived else "sessions/2026/09/17")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rollout-2026-09-17T10-00-00-{thread_id}.jsonl"
    path.write_text(json.dumps({"type": "session_meta", "payload": {"id": thread_id}}) + "\n")
    return path


def session_backend(settings):
    executable = settings.codex_bin
    from pathlib import Path
    path = Path(executable)
    source = path.read_text()
    source = source.replace('print(json.dumps({"type": "thread.started", "thread_id": thread}), flush=True)',
        r'''import os
directory = pathlib.Path(os.environ["CODEX_HOME"]) / "sessions/2026/09/17"
directory.mkdir(parents=True, exist_ok=True)
session = directory / ("rollout-2026-09-17T10-00-00-" + thread + ".jsonl")
session.write_text(json.dumps({"type": "session_meta", "payload": {"id": thread}}) + "\n")
print(json.dumps({"type": "thread.started", "thread_id": thread}), flush=True)''')
    path.write_text(source)
    return create_app(settings)


@pytest.mark.parametrize("storage", ["isolated", "legacy", "legacy_relocated"])
def test_completed_cleanup_preserves_api_result_and_other_sessions(settings, storage):
    if storage == "legacy_relocated":
        settings.codex_home.mkdir()
        for name in ("sessions", "archived_sessions"):
            target = settings.codex_home.parent / "local-session-store" / name
            target.mkdir(parents=True)
            (settings.codex_home / name).symlink_to(target, target_is_directory=True)
    other_global = rollout(settings.codex_home, str(uuid4()))
    with TestClient(session_backend(settings)) as client:
        job = finished(client, submit(client, "done")["id"])
        cid = job["conversation_id"]
        thread_id = job["response"]["thread_id"]
        home = conversation_session_home(settings.runtime, cid)
        assert list(home.rglob(f"*{thread_id}.jsonl"))
        if storage.startswith("legacy"):
            # Simulate records created before per-conversation homes existed.
            shutil.copytree(home / "sessions", settings.codex_home / "sessions", dirs_exist_ok=True)
            shutil.rmtree(home)
            home = settings.codex_home
        other = rollout(home, str(uuid4()))
        assert client.get("/v1/capabilities").json()["local_session_cleanup"]
        response = client.post(f"/v1/jobs/{job['id']}/cleanup-session")
        assert response.status_code == 200, response.text
        assert response.json()["thread_id"] == thread_id
        assert len(response.json()["deleted_files"]) == 1
        assert other.exists()
        assert other_global.exists()
        assert not list(home.rglob(f"*{thread_id}.jsonl"))
        assert client.get(f"/v1/jobs/{job['id']}").json()["response"] == job["response"]
        assert len(client.get(f"/v1/conversations/{cid}").json()["turns"]) == 1
        assert client.post(f"/v1/jobs/{job['id']}/cleanup-session").json()["deleted_files"] == []
        # Local history is gone; refuse to pretend the conversation can resume.
        assert client.post("/v1/jobs", json={"prompt": "resume", "conversation_id": cid}).status_code == 409


def test_cancel_stops_process_before_session_can_be_deleted(settings):
    with TestClient(session_backend(settings)) as client:
        job = submit(client, "timeout")
        current = wait_for(lambda: (r if r["activity"]["event_count"] else None)
                           if (r := client.get(f"/v1/jobs/{job['id']}").json()) else None)
        pid = current["activity"]["pid"]
        assert client.post(f"/v1/jobs/{job['id']}/cleanup-session").status_code == 409
        assert list(settings.runtime.glob("codex-home-*/sessions/**/rollout-*.jsonl"))
        client.post(f"/v1/jobs/{job['id']}/cancel")
        result = finished(client, job["id"])
        assert result["status"] == "cancelled"
        assert result["conversation_id"]
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        response = client.post(f"/v1/jobs/{job['id']}/cleanup-session")
        assert response.status_code == 200, response.text
        # The backend already removed the cancelled request's temporary home.
        assert response.json()["deleted_files"] == []
        assert not list(settings.runtime.glob("codex-home-*"))
        assert not list(settings.runtime.rglob("rollout-*.jsonl"))


def test_server_timeout_session_cleanup(settings):
    settings.timeout = 0.5
    with TestClient(session_backend(settings)) as client:
        result = finished(client, submit(client, "timeout")["id"])
        assert result["status_code"] == 504
        response = client.post(f"/v1/jobs/{result['id']}/cleanup-session")
        assert response.status_code == 200, response.text
        assert response.json()["deleted_files"] == []
        assert not list(settings.runtime.glob("codex-home-*"))


@pytest.mark.parametrize("outcome", ["completed", "cancelled", "timeout"])
def test_isolated_cleanup_ignores_dangling_global_session_roots(settings, outcome):
    settings.codex_home.mkdir()
    for name in ("sessions", "archived_sessions"):
        (settings.codex_home / name).symlink_to(
            settings.codex_home.parent / "missing-node-local-store" / name,
            target_is_directory=True,
        )
    if outcome == "timeout":
        settings.timeout = 0.5
    app = session_backend(settings)
    with TestClient(app) as client:
        job = submit(client, "done" if outcome == "completed" else "timeout")
        if outcome == "cancelled":
            wait_for(lambda: client.get(f"/v1/jobs/{job['id']}").json()["activity"]["event_count"])
            client.post(f"/v1/jobs/{job['id']}/cancel")
        result = finished(client, job["id"])
        assert result["status"] == ("failed" if outcome == "timeout" else outcome)
        if outcome == "timeout":
            assert result["status_code"] == 504
        cid = result["conversation_id"]
        home = conversation_session_home(settings.runtime, cid)
        assert home.is_dir()
    # Cleanup must also find the right store after an API restart.
    with TestClient(app) as client:
        response = client.post(f"/v1/jobs/{job['id']}/cleanup-session")
        assert response.status_code == 200, response.text
        assert response.json()["thread_id"]
        assert len(response.json()["deleted_files"]) == (1 if outcome == "completed" else 0)
        assert not list(home.rglob("rollout-*.jsonl"))
        assert not list(settings.runtime.glob("codex-home-*"))
        assert client.post(f"/v1/jobs/{job['id']}/cleanup-session").json()["deleted_files"] == []
        assert client.post("/v1/jobs", json={"prompt": "resume", "conversation_id": cid}).status_code == 409
    for name in ("sessions", "archived_sessions"):
        assert (settings.codex_home / name).is_symlink()
        assert not (settings.codex_home / name).exists()


def test_legacy_cleanup_still_reports_dangling_root(settings):
    with TestClient(session_backend(settings)) as client:
        job = finished(client, submit(client, "done")["id"])
        shutil.rmtree(conversation_session_home(settings.runtime, job["conversation_id"]))
        settings.codex_home.mkdir()
        (settings.codex_home / "sessions").symlink_to(settings.codex_home / "missing")
        response = client.post(f"/v1/jobs/{job['id']}/cleanup-session")
        assert response.status_code == 500


def test_rollout_cleanup_matches_metadata_and_archives(tmp_path):
    thread_id = str(uuid4())
    paths = [rollout(tmp_path, thread_id), rollout(tmp_path, thread_id, archived=True)]
    unrelated = rollout(tmp_path, str(uuid4()))
    assert set(delete_session_files(tmp_path, thread_id)) == {str(p) for p in paths}
    assert unrelated.exists()
    assert delete_session_files(tmp_path, thread_id) == []
    with pytest.raises(ValueError):
        delete_session_files(tmp_path, "../*")
    mismatch = rollout(tmp_path, thread_id)
    mismatch.write_text(json.dumps({"type": "session_meta", "payload": {"id": str(uuid4())}}))
    with pytest.raises(ValueError, match="metadata"):
        delete_session_files(tmp_path, thread_id)
    assert mismatch.exists()


def test_rollout_cleanup_allows_relocated_roots(tmp_path):
    thread_id = str(uuid4())
    relocated = tmp_path / "local-store"
    paths = [rollout(relocated, thread_id), rollout(relocated, thread_id, archived=True)]
    unrelated = rollout(relocated, str(uuid4()))
    home = tmp_path / "home"
    home.mkdir()
    for name in ("sessions", "archived_sessions"):
        (home / name).symlink_to(relocated / name, target_is_directory=True)
    assert set(delete_session_files(home, thread_id)) == {str(p) for p in paths}
    assert all(not p.exists() for p in paths)
    assert unrelated.exists()
    assert all((home / name).is_symlink() for name in ("sessions", "archived_sessions"))
    assert delete_session_files(home, thread_id) == []


def test_rollout_cleanup_rejects_file_symlinks(tmp_path):
    thread_id = str(uuid4())
    outside = rollout(tmp_path / "outside", thread_id)
    home = tmp_path / "home"
    root = home / "sessions"
    root.mkdir(parents=True)
    link = root / outside.name
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="ordinary file"):
        delete_session_files(home, thread_id)
    assert link.is_symlink()
    assert outside.exists()


def test_rollout_cleanup_does_not_follow_nested_directory_symlinks(tmp_path):
    thread_id = str(uuid4())
    outside = rollout(tmp_path / "outside", thread_id)
    home = tmp_path / "home"
    root = home / "sessions"
    root.mkdir(parents=True)
    (root / "nested").symlink_to(tmp_path / "outside/sessions", target_is_directory=True)
    assert delete_session_files(home, thread_id) == []
    assert outside.exists()


def test_rollout_cleanup_rejects_dangling_root(tmp_path):
    (tmp_path / "sessions").symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(FileNotFoundError):
        delete_session_files(tmp_path, str(uuid4()))


def test_cleanup_refuses_conversation_with_running_continuation(settings):
    with TestClient(session_backend(settings)) as client:
        first = finished(client, submit(client, "done")["id"])
        running = submit(client, "timeout", conversation_id=first["conversation_id"])
        response = client.post(f"/v1/jobs/{first['id']}/cleanup-session")
        assert response.status_code == 409
        home = conversation_session_home(settings.runtime, first["conversation_id"])
        assert list(home.rglob("rollout-*.jsonl"))
        client.post(f"/v1/jobs/{running['id']}/cancel")
        finished(client, running["id"])
