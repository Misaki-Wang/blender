"""Python client for text, images, resumed conversations and bounded batches."""
import base64
import json
import math
import time
from pathlib import Path
from uuid import uuid4

import httpx


class CodexClient:
    def __init__(self, base_url="http://127.0.0.1:8000", token=None, timeout=1800):
        self.http = httpx.Client(base_url=base_url, timeout=timeout,
                                 headers={"Authorization": f"Bearer {token}"} if token else {})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.http.close()

    @staticmethod
    def image(path):
        """Read an image on the caller's machine and transmit it as base64."""
        return {"base64": base64.b64encode(Path(path).read_bytes()).decode("ascii")}

    def chat(self, prompt, *, model=None, conversation_id=None, images=None,
             reasoning_effort=None, new_conversation=None, workspace_dir=None):
        payload = {"prompt": prompt, "images": images or []}
        for key, value in {"model":model, "conversation_id":conversation_id,
                           "reasoning_effort":reasoning_effort, "new_conversation":new_conversation,
                           "workspace_dir":workspace_dir}.items():
            if value is not None:
                payload[key] = value
        result = self.http.post("/v1/chat", json=payload)
        result.raise_for_status()
        return result.json()

    def batch(self, requests):
        """At most 32 requests per call; inspect each result's ok field."""
        result = self.http.post("/v1/batch", json={"requests": requests})
        result.raise_for_status()
        return result.json()["results"]

    def models(self):
        result = self.http.get("/v1/models")
        result.raise_for_status()
        return result.json()

    def submit(self, prompt, *, model=None, conversation_id=None, images=None,
               reasoning_effort=None, idempotency_key=None, workspace_dir=None):
        """Return a durable job immediately; does not wait for the model."""
        payload = {"prompt":prompt, "images":images or []}
        for name, value in {"model":model, "conversation_id":conversation_id,
                            "reasoning_effort":reasoning_effort, "workspace_dir":workspace_dir}.items():
            if value is not None:
                payload[name] = value
        # A lost POST response does not mean the server rejected the job. Replay
        # the same key so retries recover that job instead of generating twice.
        headers = {"Idempotency-Key": idempotency_key or str(uuid4())}
        return self._request_with_retry("POST", "/v1/jobs", json=payload, headers=headers).json()

    def _request_with_retry(self, method, path, **kwargs):
        """Retry reads and explicitly safe/idempotent writes on transient errors."""
        def request(_remaining):
            response = getattr(self.http, method.lower())(path, **kwargs)
            response.raise_for_status()
            return response
        return self._read_with_retry(request, None)

    def capabilities(self):
        return self._request_with_retry("GET", "/v1/capabilities").json()

    def create_conversation(self):
        """Reserve a conversation ID without executing a model turn."""
        # Reservation does not execute a model turn; a lost response can at most
        # leave an unused empty conversation, never duplicate paid generation.
        return self._request_with_retry("POST", "/v1/conversations").json()

    def submit_batch(self, requests, *, idempotency_key=None):
        """Atomically submit up to 1000 jobs (also subject to HTTP/queue byte limits)."""
        headers = {"Idempotency-Key":idempotency_key} if idempotency_key else {}
        response = self.http.post("/v1/jobs/batch", json={"requests":requests}, headers=headers)
        response.raise_for_status()
        return response.json()["jobs"]

    def get_job(self, job_id, *, wait_seconds=0, request_timeout=None):
        options = {"timeout":request_timeout} if request_timeout is not None else {}
        response = self.http.get(f"/v1/jobs/{job_id}", params={"wait_seconds":wait_seconds}, **options)
        response.raise_for_status()
        return response.json()

    def get_jobs(self, job_ids, *, request_timeout=None):
        """One HTTP query for up to 1000 jobs; unknown IDs have a None entry."""
        options = {"timeout":request_timeout} if request_timeout is not None else {}
        response = self.http.post("/v1/jobs/query", json={"job_ids":job_ids}, **options)
        response.raise_for_status()
        return response.json()["jobs"]

    def _read_with_retry(self, operation, deadline):
        """Retry only result reads; never replay a model turn."""
        delay = 0.5
        while True:
            remaining = deadline - time.monotonic() if deadline is not None else float("inf")
            if remaining <= 0:
                raise TimeoutError("Result collection timed out; accepted jobs continue on the server")
            retry_after = delay
            try:
                return operation(remaining)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 500, 502, 503, 504}:
                    raise
                try:
                    retry_after = max(delay, min(60, float(exc.response.headers.get("Retry-After", delay))))
                except ValueError:
                    pass
            except httpx.RequestError:
                pass
            time.sleep(min(retry_after, max(0, deadline-time.monotonic())) if deadline is not None else retry_after)
            delay = min(10, delay * 2)

    def wait(self, job_id, *, timeout=None, on_update=None, on_retry=None):
        """Wait using server long polling; return terminal job, including failures.

        By default keep checking the job until a terminal state is observed.
        An explicitly supplied timeout only limits this caller's waiting time.
        """
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be None or a positive finite number")
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            remaining = deadline - time.monotonic() if deadline is not None else float("inf")
            if remaining <= 0:
                raise TimeoutError(f"Job {job_id} is still pending; query it later or cancel it explicitly")
            def read(budget):
                try:
                    return self.get_job(job_id, wait_seconds=min(20,budget), request_timeout=min(35,budget))
                except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                    retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code in {429, 500, 502, 503, 504}
                    if on_retry is not None and retryable:
                        on_retry(exc)
                    raise
            job = self._read_with_retry(read, deadline)
            if on_update is not None:
                on_update(job)
            if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return job

    def cancel(self, job_id):
        return self._request_with_retry("POST", f"/v1/jobs/{job_id}/cancel").json()

    def cleanup_session(self, job_id):
        """Delete a stopped job's local rollout files, keeping API results."""
        # Cleanup errors must surface to the caller and remain resumable rather
        # than blocking a worker forever on an unwritable session directory.
        response = self.http.post(f"/v1/jobs/{job_id}/cleanup-session")
        response.raise_for_status()
        return response.json()

    def queue_stats(self):
        response = self.http.get("/v1/queue")
        response.raise_for_status()
        return response.json()

    def iter_results(self, requests, *, batch_size=100, max_in_flight=500, max_batch_bytes=30 * 1024 * 1024,
                     poll_interval=1, timeout=None, run_id=None):
        """Submit an iterable with bounded memory and yield terminal jobs as ready.

        Each result includes its original input index. Failed jobs are yielded too.
        Submission retries use stable idempotency keys. To resume this producer,
        reuse run_id with exactly the same ordered inputs and batching parameters. Caller
        timeouts or abandoning this iterator do not cancel accepted server jobs.
        """
        if not 1 <= batch_size <= min(1000, max_in_flight) or poll_interval <= 0 or (timeout is not None and
                (not math.isfinite(timeout) or timeout <= 0)) or max_batch_bytes <= 32:
            raise ValueError("Require 1 <= batch_size <= min(1000, max_in_flight), positive poll_interval and timeout")
        def batches():
            batch, size = [], 32  # Reserve space for the JSON envelope.
            for request in requests:
                item_size = len(json.dumps(request, ensure_ascii=False, separators=(",",":"), allow_nan=False).encode()) + 1
                if item_size + 32 > max_batch_bytes:
                    raise ValueError("A single request exceeds max_batch_bytes; reduce its image/prompt size")
                if batch and (len(batch) >= batch_size or size + item_size > max_batch_bytes):
                    yield batch
                    batch, size = [], 32
                batch.append(request)
                size += item_size
            if batch:
                yield batch

        source = iter(batches())
        run_id = run_id or str(uuid4())
        pending = {}
        buffer = None
        exhausted = False
        batch_number = 0
        input_index = 0
        retry_at = 0
        deadline = time.monotonic() + timeout if timeout is not None else float("inf")
        while pending or buffer is not None or not exhausted:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Producer {run_id} timed out; accepted jobs continue on the server")
            if buffer is None and not exhausted:
                buffer = next(source, None)
                if buffer is None:
                    exhausted, buffer = True, None
            if buffer and len(pending) + len(buffer) <= max_in_flight and time.monotonic() >= retry_at:
                try:
                    jobs = self.submit_batch(buffer, idempotency_key=f"{run_id}:{batch_number}")
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code not in {429, 502, 503, 504}:
                        raise
                    retry_at = time.monotonic() + max(2, poll_interval)
                except httpx.RequestError:
                    # Retrying the same key resolves an ambiguous lost submission response.
                    retry_at = time.monotonic() + max(2, poll_interval)
                else:
                    for offset, job in enumerate(jobs):
                        pending[job["id"]] = input_index + offset
                    input_index += len(buffer)
                    batch_number += 1
                    buffer = None
                    continue
            ids = list(pending)
            for start in range(0, len(ids), 1000):
                jobs = self._read_with_retry(
                    lambda budget:self.get_jobs(ids[start:start + 1000], request_timeout=min(30,budget)), deadline)
                for job in jobs:
                    if job is None:
                        raise LookupError("An accepted job is missing from the server")
                    if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                        yield {"index":pending.pop(job["id"]), **job}
            if pending or buffer:
                time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
