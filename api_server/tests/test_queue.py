import asyncio
import base64
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from api_server.app import create_app
from api_server.backend import CodexResult
from test_api import settings, png  # Reuse the isolated CLI fixture.


class ControlledBackend:
    def __init__(self):
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.started = []
        self.history = {}

    async def run(self, *, prompt, images, model, reasoning_effort, thread_id, output_dir, on_thread):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.started.append(prompt)
        try:
            thread_id = thread_id or str(uuid4())
            await on_thread(thread_id)
            if prompt.startswith("block"):
                while not self.release.is_set():
                    await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.01)
            history = self.history.setdefault(thread_id, [])
            history.append(prompt)
            return CodexResult(thread_id, json.dumps({"history":history,"images":[p.read_bytes().hex() for p in images]}), {})
        finally:
            with self.lock:
                self.active -= 1


def wait_for(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("Condition did not become true")


def finished(client, job_id):
    response = client.get(f"/v1/jobs/{job_id}", params={"wait_seconds":10})
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] in {"completed","failed","cancelled","interrupted"}, job
    return job


def submit(client, prompt, **kwargs):
    response = client.post("/v1/jobs", json={"prompt":prompt, **kwargs})
    assert response.status_code == 202, response.text
    return response.json()


def test_nonblocking_admission_bounded_workers_and_bulk_query(settings):
    settings.max_concurrency = 3
    backend = ControlledBackend()
    with TestClient(create_app(settings, backend)) as client:
        response = client.post("/v1/jobs/batch", json={"requests":[{"prompt":f"block-{i}"} for i in range(120)]})
        assert response.status_code == 202, response.text
        jobs = response.json()["jobs"]
        wait_for(lambda: backend.active == 3)
        stats = client.get("/v1/queue").json()
        assert stats["counts"] == {"queued":117,"running":3}
        assert stats["active_workers"] == 3
        assert client.get("/health").status_code == 200
        # There are only three executing tasks, not 120 spawned worker tasks.
        backend.release.set()
        for job in jobs:
            assert finished(client, job["id"])["status"] == "completed"
        missing = str(uuid4())
        query = client.post("/v1/jobs/query", json={"job_ids":[jobs[-1]["id"], missing, jobs[0]["id"]]}).json()
        assert [j["id"] if j else None for j in query["jobs"]] == [jobs[-1]["id"], None, jobs[0]["id"]]
        assert query["missing_ids"] == [missing]
        assert backend.peak == 3


def test_same_conversation_does_not_block_other_conversations(settings):
    settings.max_concurrency = 2
    backend = ControlledBackend()
    with TestClient(create_app(settings, backend)) as client:
        cid = client.post("/v1/chat", json={"prompt":"start"}).json()["conversation_id"]
        first = submit(client, "block-first", conversation_id=cid)
        wait_for(lambda: "block-first" in backend.started)
        second = submit(client, "second", conversation_id=cid)
        third = submit(client, "third", conversation_id=cid)
        independent = submit(client, "independent")
        assert finished(client, independent["id"])["status"] == "completed"
        assert client.get(f"/v1/jobs/{second['id']}").json()["status"] == "queued"
        assert "second" not in backend.started
        backend.release.set()
        finished(client, first["id"])
        finished(client, second["id"])
        final = finished(client, third["id"])
        assert json.loads(final["response"]["answer"])["history"] == ["start","block-first","second","third"]


def test_idempotency_concurrent_submission_and_conflict(settings):
    backend = ControlledBackend()
    with TestClient(create_app(settings, backend)) as client:
        def send(_):
            return client.post("/v1/jobs", json={"prompt":"block"}, headers={"Idempotency-Key":"unique-task"})
        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(send, range(12)))
        assert all(r.status_code == 202 for r in responses)
        ids = {r.json()["id"] for r in responses}
        assert len(ids) == 1
        conflict = client.post("/v1/jobs", json={"prompt":"different"}, headers={"Idempotency-Key":"unique-task"})
        assert conflict.status_code == 409
        backend.release.set()
        finished(client, ids.pop())
        assert backend.started == ["block"]


def test_atomic_capacity_and_byte_limits(settings):
    settings.max_concurrency = 1
    settings.max_pending = 2
    backend = ControlledBackend()
    with TestClient(create_app(settings, backend)) as client:
        job = submit(client, "block")
        response = client.post("/v1/jobs/batch", json={"requests":[{"prompt":"one"},{"prompt":"two"}]})
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "2"
        assert client.get("/v1/queue").json()["pending"] == 1
        bad = client.post("/v1/jobs/batch", json={"requests":[{"prompt":"valid"},{"prompt":"image","images":[{"base64":"bad"}]}]})
        assert bad.status_code == 422
        assert client.get("/v1/queue").json()["pending"] == 1
        assert not list((settings.runtime / "job-inputs").iterdir())
        backend.release.set()
        finished(client, job["id"])
    settings.max_pending_bytes = 10
    with TestClient(create_app(settings, backend)) as client:
        assert client.post("/v1/jobs", json={"prompt":"too large"}).status_code == 413


def test_cancel_queued_and_running_releases_capacity(settings):
    settings.max_concurrency = 1
    backend = ControlledBackend()
    with TestClient(create_app(settings, backend)) as client:
        running = submit(client, "block-running")
        wait_for(lambda: backend.active == 1)
        queued = submit(client, "must-not-run")
        assert client.post(f"/v1/jobs/{queued['id']}/cancel").json()["status"] == "cancelled"
        client.post(f"/v1/jobs/{running['id']}/cancel")
        assert finished(client, running["id"])["status"] == "cancelled"
        next_job = submit(client, "next")
        assert finished(client, next_job["id"])["status"] == "completed"
        assert "must-not-run" not in backend.started
        assert client.post(f"/v1/jobs/{next_job['id']}/cancel").json()["status"] == "completed"


def test_restart_recovers_queued_image_snapshot_not_running_job(settings):
    settings.max_concurrency = 1
    backend = ControlledBackend()
    path = settings.image_roots[0] / "input.png"
    path.write_bytes(png())
    with TestClient(create_app(settings, backend)) as client:
        running = submit(client, "block-interrupted")
        wait_for(lambda: backend.active == 1)
        queued = submit(client, "image", images=[{"path":str(path)}])
        path.unlink()  # Durable input must no longer depend on the source file.
        assert client.get(f"/v1/jobs/{queued['id']}").json()["status"] == "queued"
    with TestClient(create_app(settings, ControlledBackend())) as client:
        assert finished(client, running["id"])["status"] == "interrupted"
        result = finished(client, queued["id"])
        assert result["status"] == "completed"
        assert json.loads(result["response"]["answer"])["images"] == [png().hex()]
        wait_for(lambda: not list((settings.runtime / "job-inputs").iterdir()))


def test_async_cli_failure_timeout_and_long_poll(settings):
    settings.timeout = 0.3
    with TestClient(create_app(settings)) as client:
        timeout = submit(client, "timeout")
        failure = submit(client, "fail")
        assert finished(client, timeout["id"])["status_code"] == 504
        assert finished(client, failure["id"])["status_code"] == 502
        assert client.get(f"/v1/jobs/{uuid4()}").status_code == 404
        assert client.post(f"/v1/jobs/{uuid4()}/cancel").status_code == 404


def test_queue_and_sync_requests_share_global_capacity(settings):
    settings.max_concurrency = 1
    backend = ControlledBackend()
    with TestClient(create_app(settings, backend)) as client:
        running = submit(client, "block")
        wait_for(lambda: backend.active == 1)
        with ThreadPoolExecutor(max_workers=1) as pool:
            sync = pool.submit(client.post, "/v1/chat", json={"prompt":"sync"})
            wait_for(lambda: client.get("/v1/queue").json()["counts"].get("queued") == 1)
            assert "sync" not in backend.started
            backend.release.set()
            assert sync.result(timeout=10).status_code == 200
        finished(client, running["id"])
        assert backend.peak == 1


def test_queue_endpoints_require_auth(settings):
    settings.token = "secret"
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/jobs", json={"prompt":"x"}).status_code == 401
        assert client.get("/v1/queue").status_code == 401
        assert client.post("/v1/jobs/query", json={"job_ids":[str(uuid4())]}).status_code == 401


def test_shutdown_with_unwritable_result_store_recovers_as_interrupted(settings):
    backend = ControlledBackend()
    app = create_app(settings, backend)
    failed_write = threading.Event()

    def disk_full(*args):
        failed_write.set()
        raise sqlite3.OperationalError("simulated disk full")

    with TestClient(app) as client:
        job = submit(client, "block")
        wait_for(lambda: backend.active == 1)
        app.state.queue.store.finish = disk_full
        backend.release.set()
        assert failed_write.wait(5)
    # Shutdown must not loop forever trying to persist to a broken database.
    with TestClient(create_app(settings, ControlledBackend())) as client:
        assert finished(client, job["id"])["status"] == "interrupted"
