import base64
import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api_server.app import Settings, create_app


@pytest.fixture
def settings(tmp_path):
    executable = tmp_path / "fake-codex"
    executable.write_text(f"#!{sys.executable}\n" + r'''
import json, pathlib, sys, time, uuid
args = sys.argv[1:]
assert args[0] == "exec"
assert 'approval_policy="never"' in args
assert 'sandbox_mode="read-only"' in args
prompt = sys.stdin.read()
positional = args[args.index("--") + 1:]
assert positional[-1] == "-"
thread = positional[0] if "resume" in args else str(uuid.uuid4())
print(json.dumps({"type": "thread.started", "thread_id": thread}), flush=True)
if prompt == "timeout":
    time.sleep(10)
if prompt == "fail":
    print(json.dumps({"type":"turn.failed", "error":{"message":"simulated failure"}}))
    sys.exit(1)
if prompt == "incomplete":
    sys.exit(0)
path = pathlib.Path(thread + ".json")
history = json.loads(path.read_text()) if path.exists() else []
history.append(prompt)
path.write_text(json.dumps(history))
image_count = args.count("--image")
for index, value in enumerate(args):
    if value == "--image":
        assert pathlib.Path(args[index + 1]).is_file()
model = args[args.index("--model") + 1] if "--model" in args else None
answer = json.dumps({"history":history,"images":image_count,"model":model})
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":"intermediate commentary"}}))
if prompt != "empty":
    pathlib.Path(args[args.index("--output-last-message") + 1]).write_text(answer)
print(json.dumps({"type":"turn.completed", "usage":{"input_tokens":10,"output_tokens":4}}))
''')
    executable.chmod(0o700)
    return Settings(runtime=tmp_path / "runtime", codex_bin=str(executable),
                    codex_home=tmp_path / "codex-home", default_model="test-model",
                    image_roots=[tmp_path], timeout=5)


def png():
    file = io.BytesIO()
    Image.new("RGB", (24, 24), "red").save(file, format="PNG")
    return file.getvalue()


def test_text_resume_models_and_restart(settings):
    with TestClient(create_app(settings)) as client:
        first = client.post("/v1/chat", json={"prompt":"记住 ABC", "model":"chosen-model",
                                             "reasoning_effort":"low", "new_conversation":True})
        assert first.status_code == 200, first.text
        data = first.json()
        assert json.loads(data["answer"])["history"] == ["记住 ABC"]
        cid = data["conversation_id"]
        second = client.post("/v1/chat", json={"prompt":"继续", "conversation_id":cid}).json()
        assert second["thread_id"] == data["thread_id"]
        assert second["model"] == "chosen-model"
        assert second["reasoning_effort"] == "low"
        assert json.loads(second["answer"])["history"] == ["记住 ABC", "继续"]
        assert second["usage"]["output_tokens"] == 4
        independent = client.post("/v1/chat", json={"prompt":"新的"}).json()
        assert independent["conversation_id"] != cid
    with TestClient(create_app(settings)) as client:
        third = client.post("/v1/chat", json={"prompt":"重启后", "conversation_id":cid}).json()
        assert len(json.loads(third["answer"])["history"]) == 3
        assert len(client.get(f"/v1/conversations/{cid}").json()["turns"]) == 3


def test_images_new_and_resume(settings):
    path = settings.image_roots[0] / "a.png"
    path.write_bytes(png())
    encoded = "data:image/png;base64," + base64.b64encode(png()).decode()
    with TestClient(create_app(settings)) as client:
        data = client.post("/v1/chat", json={"prompt":"看看", "images":[{"path":str(path)}, {"base64":encoded}]}).json()
        assert json.loads(data["answer"])["images"] == 2
        second = client.post("/v1/chat", json={"prompt":"再看", "conversation_id":data["conversation_id"],
                                             "images":[{"base64":encoded}]}).json()
        assert json.loads(second["answer"])["images"] == 1
        assert not list(settings.runtime.glob("request-*"))


@pytest.mark.parametrize("body", [
    {"prompt":" "}, {"prompt":"x","new_conversation":False},
    {"prompt":"x","new_conversation":True,"conversation_id":"00000000-0000-0000-0000-000000000000"},
    {"prompt":"x","model":"--bad"}, {"prompt":"x","conversation_id":"bad"},
    {"prompt":"x","images":[{}]}, {"prompt":"x","images":[{"base64":"invalid"}]},
    {"prompt":"x","images":[{"base64":base64.b64encode(b"not an image").decode()}]},
    {"prompt":"x","images":[{"path":"/etc/passwd"}]}, {"prompt":"x","unknown":1},
])
def test_validation(settings, body):
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/chat", json=body).status_code == 422


def test_missing_session(settings):
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/chat", json={"prompt":"x","conversation_id":"00000000-0000-0000-0000-000000000000"}).status_code == 404


def test_batch_order_and_partial_failure(settings):
    with TestClient(create_app(settings)) as client:
        cid = client.post("/v1/chat", json={"prompt":"start"}).json()["conversation_id"]
        response = client.post("/v1/batch", json={"requests":[
            {"prompt":"one","conversation_id":cid}, {"prompt":"fail"},
            {"prompt":"two","conversation_id":cid}, {"prompt":"independent"}]}).json()
        items = response["results"]
        assert [item["ok"] for item in items] == [True, False, True, True]
        assert items[1]["status_code"] == 502
        assert json.loads(items[2]["response"]["answer"])["history"] == ["start","one","two"]


@pytest.mark.parametrize("prompt", ["fail", "incomplete", "empty"])
def test_failed_cli(settings, prompt):
    with TestClient(create_app(settings)) as client:
        result = client.post("/v1/chat", json={"prompt":prompt})
        assert result.status_code == 502
        cid = result.json()["detail"]["conversation_id"]
        assert client.get(f"/v1/conversations/{cid}").json()["turns"][0]["status"] == "failed"


def test_timeout_and_capacity_recovery(settings):
    settings.timeout = 0.3
    with TestClient(create_app(settings)) as client:
        result = client.post("/v1/chat", json={"prompt":"timeout"})
        assert result.status_code == 504
        cid = result.json()["detail"]["conversation_id"]
        assert client.get(f"/v1/conversations/{cid}").json()["turns"][0]["status"] == "timeout"
        assert client.post("/v1/chat", json={"prompt":"works"}).status_code == 200
        assert not list(settings.runtime.glob("request-*"))


def test_auth_body_limit_and_models(settings):
    settings.token = "secret"
    settings.max_body_bytes = 100
    settings.codex_home.mkdir()
    (settings.codex_home / "models_cache.json").write_text(json.dumps({"models":[{"slug":"test-model"}]}))
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 401
        headers = {"Authorization":"Bearer secret"}
        assert client.get("/v1/models", headers=headers).json()["data"][0]["id"] == "test-model"
        assert client.post("/v1/chat", headers=headers, json={"prompt":"a" * 200}).status_code == 413
        assert client.post("/v1/chat", headers=headers, json={"prompt":"ok"}).status_code == 200


def test_single_worker_guard(settings):
    with TestClient(create_app(settings)):
        with pytest.raises(RuntimeError, match="already in use"):
            with TestClient(create_app(settings)):
                pass
