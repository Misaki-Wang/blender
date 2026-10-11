import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api_server.app import create_app
from api_server.backend import _event_lines
from test_api import settings


def read_fragmented(chunks, limit=128):
    class Stream:
        def __init__(self):
            self.chunks = iter(chunks)

        async def read(self, size):
            chunk = next(self.chunks, b"")
            assert len(chunk) <= size
            return chunk

    async def collect():
        activity = {}
        lines = [bytes(line) async for line in _event_lines(Stream(), activity, limit=limit)]
        assert activity["last_activity_at"] > 0
        return lines

    return asyncio.run(collect())


def test_fragmented_unicode_and_control_records_after_large_event():
    first = json.dumps({"type": "thread.started", "thread_id": "线程"}, ensure_ascii=False).encode()
    last = b'{"type":"turn.completed","usage":{}}'
    oversized = json.dumps({"type": "item.completed", "output": "x" * 512}).encode()
    data = first + b"\n" + oversized + b"\n" + last
    # Includes split UTF-8 characters, newlines, and a final record without LF.
    assert read_fragmented([data[i:i + 7] for i in range(0, len(data), 7)]) == [first, last]


def test_event_limit_boundary_and_oversized_eof():
    exact = b"x" * 128
    assert read_fragmented([exact, b"\n", exact, b"x"]) == [exact]


@pytest.mark.parametrize("prompt,expected_status", [("done", 200), ("fail", 502), ("incomplete", 502)])
def test_large_subprocess_output_preserves_terminal_status(settings, prompt, expected_status, caplog):
    executable = Path(settings.codex_bin)
    source = executable.read_text()
    source = source.replace('if prompt == "timeout":', r'''
sys.stdout.write('{"type":"item.completed","item":{"aggregated_output":"')
for _ in range(160):
    sys.stdout.write("x" * (64 * 1024))
sys.stdout.write('"}}\n')
sys.stdout.flush()
if prompt == "timeout":''')
    executable.write_text(source)
    with TestClient(create_app(settings)) as client:
        result = client.post("/v1/chat", json={"prompt": prompt})
        assert result.status_code == expected_status, result.text
        if expected_status == 200:
            assert json.loads(result.json()["answer"])["history"] == [prompt]
            assert result.json()["usage"]["output_tokens"] == 4
        elif prompt == "fail":
            cid = result.json()["detail"]["conversation_id"]
            error = client.get(f"/v1/conversations/{cid}").json()["turns"][0]["error"]
            assert "simulated failure" in error
    assert "Skipped oversized Codex event" in caplog.text
