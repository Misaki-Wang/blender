"""Offline scheduler load test. Uses a simulated backend; makes NO model calls.

api_server/.venv/bin/python -m api_server.benchmark_queue --jobs 3000 --concurrency 8
Measures queue overhead only, not Codex throughput or account rate limits.
"""
import argparse
import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from .app import ROOT, Settings, create_app
from .backend import CodexResult


class SimulatedBackend:
    def __init__(self, delay):
        self.delay = delay
        self.release = threading.Event()
        self.active = self.peak = 0

    async def run(self, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            thread_id = str(uuid4())
            await kwargs["on_thread"](thread_id)
            while not self.release.is_set():
                await asyncio.sleep(0.01)
            await asyncio.sleep(self.delay)
            return CodexResult(thread_id, kwargs["prompt"], {})
        finally:
            self.active -= 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.01)
    args = parser.parse_args()
    if args.jobs < 1 or args.concurrency < 1 or args.delay < 0:
        parser.error("jobs/concurrency must be positive, delay nonnegative")
    (ROOT / "runtime").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="benchmark-", dir=ROOT / "runtime") as tmp:
        settings = Settings(runtime=Path(tmp), max_pending=max(10000,args.jobs),
                            max_concurrency=args.concurrency, timeout=600)
        backend = SimulatedBackend(args.delay)
        headers = {"Authorization":f"Bearer {settings.token}"} if settings.token else {}
        with TestClient(create_app(settings, backend), headers=headers) as client:
            started = time.monotonic()
            ids = []
            for offset in range(0,args.jobs,1000):
                response = client.post("/v1/jobs/batch", json={"requests":[
                    {"prompt":str(i)} for i in range(offset,min(args.jobs,offset+1000))]})
                assert response.status_code == 202, response.text
                ids.extend(j["id"] for j in response.json()["jobs"])
            admitted_seconds = time.monotonic() - started
            stats = client.get("/v1/queue").json()
            assert stats["pending"] == args.jobs, stats
            assert stats["active_workers"] <= args.concurrency, stats
            print(json.dumps({"phase":"admitted", "jobs":len(ids), "seconds":admitted_seconds, "queue":stats}), flush=True)
            backend.release.set()
            execution_start = time.monotonic()
            last_report = execution_start
            while True:
                stats = client.get("/v1/queue").json()
                if stats["pending"] == 0:
                    break
                if time.monotonic()-last_report > 10:
                    print(json.dumps({"phase":"draining", "counts":stats["counts"]}), flush=True)
                    last_report = time.monotonic()
                if time.monotonic()-execution_start > 600:
                    raise TimeoutError("Benchmark did not finish within 600 seconds")
                await_seconds = 0.2
                time.sleep(await_seconds)
            total_seconds = time.monotonic() - started
            assert stats["counts"].get("completed") == args.jobs, stats
            for offset in range(0,args.jobs,1000):
                jobs = client.post("/v1/jobs/query", json={"job_ids":ids[offset:offset+1000]}).json()["jobs"]
                assert all(job["response"]["answer"] == str(offset+i) for i,job in enumerate(jobs))
            print(json.dumps({"simulated_backend":True, "jobs":args.jobs, "concurrency":args.concurrency,
                              "observed_peak":backend.peak, "admission_seconds":round(admitted_seconds,3),
                              "total_seconds":round(total_seconds,3), "completed":stats["counts"]["completed"],
                              "history_database":stats.get("history_database")}), flush=True)


if __name__ == "__main__":
    main()
