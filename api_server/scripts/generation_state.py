"""Durable submission and resume helpers shared by image/code generation."""
import json
import time
from uuid import uuid4


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def start_stage(client, task_dir, run, stage, prompt, **options):
    previous = run.get(stage, {})
    job_id = previous.get("job_id") or previous.get("id")
    if job_id and previous.get("status") not in {"failed", "cancelled", "interrupted"}:
        return {"id": job_id}
    submissions = run.setdefault("submissions", {})
    if previous.get("status") in {"failed", "cancelled", "interrupted"}:
        submissions.pop(stage, None)
    key = submissions.setdefault(stage, str(uuid4()))
    # Persist before POST: a restart can recover even a lost submission response.
    write_json(task_dir / "task.json", run)
    return client.submit(prompt, idempotency_key=key, **options)


def artifact_before(run, stage, path):
    stamps = run.setdefault("artifact_before", {})
    if stage not in stamps:
        # Older tasks did not save the pre-submission timestamp. Their known job
        # must finish before validation; do not compare against its new output.
        stamps[stage] = None if run.get(stage) else path.stat().st_mtime_ns if path.exists() else None
    return stamps[stage]


def reset_stage(run, stage, path):
    run.pop(stage, None)
    run.get("submissions", {}).pop(stage, None)
    run.setdefault("artifact_before", {})[stage] = path.stat().st_mtime_ns if path.exists() else None


def recover_legacy_image(client, task_dir, run, prompt, timeout):
    """Recover pre-idempotency image submissions from their saved conversation.

    If no matching turn is visible, leave the ambiguous task alone: it may still
    be queued. New tasks use durable submission keys and do not need this path.
    """
    if run.get("image") or run.get("submissions", {}).get("image"):
        return
    if not (task_dir / "image_prompt.txt").exists():
        return  # The previous caller stopped before preparing a submission.
    deadline = time.monotonic() + timeout if timeout else None
    cid = run["image_conversation_id"]
    while True:
        response = client._request_with_retry("GET", f"/v1/conversations/{cid}", params={"limit": 1000})
        turns = [t for t in response.json()["turns"] if t["prompt"] == prompt]
        if not turns:
            raise ValueError("旧图片任务缺少 job_id，且提交结果尚无法确认；保留任务，稍后用 --resume 重试")
        turn = max(turns, key=lambda t: t["id"])
        if turn["status"] == "completed":
            result = {"status": "completed", "response": {"conversation_id": cid, "answer": turn["answer"]}}
            run["image"] = result
            run.setdefault("artifact_before", {})["image"] = None
            write_json(task_dir / "image.response.json", result)
            write_json(task_dir / "task.json", run)
            return
        if turn["status"] != "running":
            return  # A definitively failed turn may be retried.
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("旧图片任务仍在服务端运行；请稍后恢复")
        time.sleep(2)
