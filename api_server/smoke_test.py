"""Opt-in live test. Uses the current Codex login and real model requests.

Run from the project root: api_server/.venv/bin/python -m api_server.smoke_test
"""
import argparse
import base64
import io
import json
import os
import secrets
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from .app import ROOT, Settings, create_app
from .client import CodexClient


def check_http(base_url, model):
    """Two real HTTP requests, including model selection and an image on resume."""
    token = secrets.token_hex(4)
    stream = io.BytesIO()
    Image.new("RGB", (128, 128), "blue").save(stream, format="PNG")
    with CodexClient(base_url, token=os.getenv("API_SERVER_TOKEN")) as client:
        assert client.http.get("/health").json()["status"] == "ok"
        assert client.http.get("/docs").status_code == 200
        assert "/v1/chat" in client.http.get("/openapi.json").json()["paths"]
        print(json.dumps({"check":"models", "response":client.models()}), flush=True)
        first = client.chat(f"Remember the code {token}. Reply only with the code. Do not use tools.",
                            model=model, reasoning_effort="low")
        assert token in first["answer"], first
        if model:
            assert first["model"] == model, first
        print(json.dumps({"check":"http_model_selection", "response":first}), flush=True)
        second = client.chat("What code did I ask you to remember, and what color is this new image? Reply with the code and one English color word. Do not use tools.",
                             conversation_id=first["conversation_id"],
                             images=[{"base64":base64.b64encode(stream.getvalue()).decode()}])
        assert token in second["answer"] and "blue" in second["answer"].lower(), second
        assert second["model"] == first["model"] and second["thread_id"] == first["thread_id"]
        print(json.dumps({"check":"http_resume_with_image", "response":second}), flush=True)
    print("Live HTTP smoke test passed.", flush=True)


def check_queue(base_url, model):
    """Real HTTP: asynchronous admission, duplicate submission, image and resume."""
    token = secrets.token_hex(4)
    stream = io.BytesIO()
    Image.new("RGB", (128, 128), "green").save(stream, format="PNG")
    requests = [
        {"prompt":f"Remember the code {token}. Reply only with the code. Do not use tools."},
        {"prompt":"What color is this image? Reply with one English color word. Do not use tools.",
         "images":[{"base64":base64.b64encode(stream.getvalue()).decode()}]},
        {"prompt":"Reply exactly QUEUE_OK. Do not use tools."},
    ]
    for request in requests:
        request["reasoning_effort"] = "low"
        if model:
            request["model"] = model
    with CodexClient(base_url, token=os.getenv("API_SERVER_TOKEN")) as client:
        key = f"queue-smoke-{token}"
        jobs = client.submit_batch(requests, idempotency_key=key)
        duplicate = client.submit_batch(requests, idempotency_key=key)
        assert [j["id"] for j in jobs] == [j["id"] for j in duplicate]
        print(json.dumps({"check":"async_admission_and_idempotency", "jobs":jobs}), flush=True)
        results = [client.wait(job["id"], timeout=180) for job in jobs]
        assert all(job["status"] == "completed" for job in results), results
        assert token in results[0]["response"]["answer"], results
        assert "green" in results[1]["response"]["answer"].lower(), results
        assert "QUEUE_OK" in results[2]["response"]["answer"], results
        print(json.dumps({"check":"async_text_and_image", "jobs":results}), flush=True)
        resume = client.submit("What code did I ask you to remember? Reply only with the code. Do not use tools.",
                               conversation_id=results[0]["response"]["conversation_id"])
        answer = client.wait(resume["id"], timeout=180)
        assert answer["status"] == "completed" and token in answer["response"]["answer"], answer
        print(json.dumps({"check":"async_resume", "job":answer, "queue":client.queue_stats()}), flush=True)
    print("Live asynchronous queue smoke test passed.", flush=True)


def main():
    token = secrets.token_hex(4)
    stream = io.BytesIO()
    Image.new("RGB", (128, 128), "red").save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode()
    (ROOT / "runtime").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke-", dir=ROOT / "runtime") as tmp:
        settings = Settings(runtime=Path(tmp), timeout=180,
                            default_model=os.getenv("SMOKE_MODEL") or Settings().default_model)
        headers = {"Authorization":f"Bearer {settings.token}"} if settings.token else {}
        with TestClient(create_app(settings), headers=headers) as client:
            first = client.post("/v1/chat", json={
                "prompt":f"Remember this code for our next turn: {token}. What color is the attached image? Reply with the code and one English color word. Do not use tools.",
                "images":[{"base64":encoded}], "reasoning_effort":"low"})
            assert first.status_code == 200, first.text
            data = first.json()
            assert token in data["answer"] and "red" in data["answer"].lower(), data
            print(json.dumps({"check":"image_and_text", "response":data}), flush=True)
        # Recreate the app to verify persistent state and Codex resume together.
        with TestClient(create_app(settings), headers=headers) as client:
            second = client.post("/v1/chat", json={"prompt":"What code did I ask you to remember? Reply only with that code. Do not use tools.",
                "conversation_id":data["conversation_id"], "new_conversation":False})
            assert second.status_code == 200, second.text
            followup = second.json()
            assert token in followup["answer"], followup
            assert followup["thread_id"] == data["thread_id"]
            print(json.dumps({"check":"resume_after_restart", "response":followup}), flush=True)
            batch = client.post("/v1/batch", json={"requests":[
                {"prompt":"Reply exactly BATCH_A. Do not use tools.","reasoning_effort":"low"},
                {"prompt":"Reply exactly BATCH_B. Do not use tools.","reasoning_effort":"low"}]})
            assert batch.status_code == 200, batch.text
            items = batch.json()["results"]
            assert all(item["ok"] for item in items), items
            assert "BATCH_A" in items[0]["response"]["answer"], items
            assert "BATCH_B" in items[1]["response"]["answer"], items
            assert items[0]["response"]["conversation_id"] != items[1]["response"]["conversation_id"]
            print(json.dumps({"check":"batch", "results":items}), flush=True)
    print("Live Codex smoke test passed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Test an already running server over HTTP (2 model calls)")
    parser.add_argument("--model", help="Model ID for the HTTP test")
    parser.add_argument("--queue", action="store_true", help="Test async queue over HTTP (4 model calls)")
    args = parser.parse_args()
    if args.queue:
        if not args.base_url:
            parser.error("--queue requires --base-url")
        check_queue(args.base_url, args.model)
    elif args.base_url:
        check_http(args.base_url, args.model)
    else:
        main()
