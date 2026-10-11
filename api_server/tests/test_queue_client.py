import json

import httpx
import pytest

from api_server.client import CodexClient


def test_iter_results_bounded_window_retries_and_partial_failures(monkeypatch):
    submitted = []
    attempts = []
    pending = {}
    peak = 0
    first = True
    query_count = 0

    def handle(request):
        nonlocal first, peak, query_count
        data = json.loads(request.content)
        if request.url.path == "/v1/jobs/batch":
            attempts.append((request.headers["idempotency-key"], data))
            if first:
                first = False
                return httpx.Response(429, json={"detail":"busy"})
            jobs = []
            for item in data["requests"]:
                index = int(item["prompt"])
                job = {"id":str(index), "status":"queued"}
                submitted.append(index)
                pending[str(index)] = job
                jobs.append(job)
            peak = max(peak, len(pending))
            return httpx.Response(202, json={"jobs":jobs})
        if request.url.path == "/v1/jobs/query":
            query_count += 1
            results = []
            for job_id in data["job_ids"]:
                # Deliberately finish only some on each poll to exercise refill.
                if int(job_id) % 2 == query_count % 2:
                    pending.pop(job_id)
                    results.append({"id":job_id,"status":"failed" if job_id == "5" else "completed"})
                else:
                    results.append(pending[job_id])
            return httpx.Response(200, json={"jobs":results})
        raise AssertionError(request.url)

    clock = [0.0]
    monkeypatch.setattr("api_server.client.time.monotonic", lambda:clock[0])
    monkeypatch.setattr("api_server.client.time.sleep", lambda delay:clock.__setitem__(0, clock[0]+delay))
    with CodexClient() as client:
        client.http.close()
        client.http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle))
        results = list(client.iter_results(({"prompt":str(i)} for i in range(23)),
                                          batch_size=4, max_in_flight=8, run_id="stable-run"))
    assert sorted(r["index"] for r in results) == list(range(23))
    assert [r["index"] for r in results if r["status"] == "failed"] == [5]
    assert attempts[0] == attempts[1]
    assert submitted == list(range(23))
    assert peak <= 8


def test_submit_wait_and_cancel_client():
    paths = []

    def handle(request):
        paths.append(request.url.path)
        if request.url.path == "/v1/jobs":
            assert request.headers["Idempotency-Key"] == "key"
            assert json.loads(request.content)["model"] == "test-model"
            return httpx.Response(202, json={"id":"test", "status":"queued"})
        return httpx.Response(200, json={"id":"test","status":"completed","response":{"answer":"OK"}})

    with CodexClient() as client:
        client.http.close()
        client.http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle))
        job = client.submit("hello", model="test-model", idempotency_key="key")
        assert client.wait(job["id"])["response"]["answer"] == "OK"
        client.cancel(job["id"])
    assert paths == ["/v1/jobs", "/v1/jobs/test", "/v1/jobs/test/cancel"]


def test_iter_results_splits_by_json_bytes():
    sent_sizes = []
    pending = {}

    def handle(request):
        data = json.loads(request.content)
        if request.url.path == "/v1/jobs/batch":
            sent_sizes.append(len(request.content))
            jobs = []
            for item in data["requests"]:
                job = {"id":item["prompt"],"status":"completed"}
                pending[job["id"]] = job
                jobs.append(job)
            return httpx.Response(202, json={"jobs":jobs})
        return httpx.Response(200, json={"jobs":[pending[j] for j in data["job_ids"]]})

    with CodexClient() as client:
        client.http.close()
        client.http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle))
        results = list(client.iter_results([{"prompt":str(i),"images":[{"base64":"x" * 120}]} for i in range(5)],
                                          max_batch_bytes=400))
    assert len(results) == 5
    assert len(sent_sizes) == 3
    assert max(sent_sizes) <= 400


def test_result_collection_recovers_without_resubmitting(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("api_server.client.time.monotonic", lambda:clock[0])
    monkeypatch.setattr("api_server.client.time.sleep", lambda delay:clock.__setitem__(0, clock[0]+delay))
    submissions = reads = 0

    def handle(request):
        nonlocal submissions, reads
        if request.url.path == "/v1/jobs/batch":
            submissions += 1
            return httpx.Response(202, json={"jobs":[{"id":"one","status":"queued"}]})
        reads += 1
        if reads == 1:
            return httpx.Response(503, headers={"Retry-After":"2"})
        if reads == 2:
            raise httpx.ConnectError("temporary disconnect")
        return httpx.Response(200, json={"jobs":[{"id":"one","status":"completed"}]})

    with CodexClient() as client:
        client.http.close()
        client.http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle))
        results = list(client.iter_results([{"prompt":"hello"}]))
    assert submissions == 1 and reads == 3
    assert results[0]["status"] == "completed"
    assert clock[0] >= 3


def test_result_retry_respects_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("api_server.client.time.monotonic", lambda:clock[0])
    monkeypatch.setattr("api_server.client.time.sleep", lambda delay:clock.__setitem__(0, clock[0]+delay))

    def handle(request):
        assert request.extensions["timeout"]["read"] <= 3
        return httpx.Response(503)

    with CodexClient() as client:
        client.http.close()
        client.http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle))
        with pytest.raises(TimeoutError):
            client.wait("one", timeout=3)
    assert clock[0] == 3


def test_wait_tracks_long_running_job_without_default_deadline(monkeypatch):
    clock, updates, retries = [0.0], [], []
    monkeypatch.setattr("api_server.client.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("api_server.client.time.sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    statuses = iter([503, "queued", "running", "running", "completed"])

    def handle(request):
        assert request.method == "GET"
        assert request.extensions["timeout"]["read"] <= 35
        clock[0] += 4000  # Well beyond every previous fixed task deadline.
        status = next(statuses)
        if status == 503:
            return httpx.Response(503)
        return httpx.Response(200, json={"id": "long-job", "status": status})

    with CodexClient() as client:
        client.http.close()
        client.http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle))
        result = client.wait("long-job", on_update=lambda job: updates.append(job["status"]),
                             on_retry=lambda error: retries.append(type(error).__name__))
    assert result["status"] == "completed"
    assert updates == ["queued", "running", "running", "completed"]
    assert retries == ["HTTPStatusError"]
