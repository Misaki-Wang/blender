#!/usr/bin/env python3
"""从 prompt 文字、文件或目录生成图片。

在 data_pipeline 目录运行：
  api_server/.venv/bin/python scripts/image_gen.py --prompt "一把红色椅子"
  api_server/.venv/bin/python scripts/image_gen.py \\
    --prompt output/descriptions/piano --n 3 --output-dir output/images/piano

每个 prompt 生成 n 张图片（默认 1），每张图片提交一个独立任务。
目录输入读取直属的所有普通文件，不递归，按文件名排序。
目录模式输出：output/{完整prompt文件名}/{采样编号}/image.png，编号从 0 开始。
批量模式的 task_id 可指定额外的输出子目录；默认最多并发 4 个任务。
调用端和 API 服务必须能访问相同输出目录，服务端须提供图片生成工具。
重复使用相同输出任务目录会继续原对话并更新图片。
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from uuid import UUID, uuid4
from urllib.parse import quote

import httpx
from PIL import Image

# Allow both direct script execution and importing scripts.image_gen.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_server.client import CodexClient


def api_server_token():
    token = os.environ.get("API_SERVER_TOKEN")
    if token is not None:
        return token.strip() or None
    path = Path(__file__).resolve().parent / "API_SERVER_TOKEN"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def text_or_file(value, name):
    """Only existing files are auto-read; @path forces file interpretation."""
    if value is None:
        return None
    value = str(value)
    forced = value.startswith("@")
    if value.startswith("text:"):
        text = value[5:]
    else:
        path = Path(value[1:] if forced else value).expanduser()
        try:
            exists = path.is_file()
        except OSError as exc:
            if exc.errno not in {errno.ENAMETOOLONG, errno.EINVAL} or forced:
                raise
            exists = False
        if forced or exists:
            text = path.read_text(encoding="utf-8")
        else:
            if "\n" not in value and value.startswith(("/", "./", "../", "~/")):
                raise FileNotFoundError(f"{name} File not exists: {value}")
            text = value
    if not text.strip():
        raise ValueError(f"{name} cannot be empty.")
    return text.strip()


def build_prompt(instruction, image_path):
    return (
        "Generate exactly one image using your image generation tool, following the prompt below.\n"
        "Save the generated image as a single static PNG file at this required absolute path: "
        f"{image_path}\n"
        "If the tool saves the image elsewhere, copy it to the required path "
        "(convert to PNG if necessary). Replace the file if it already exists. "
        "A text response, link, or code alone is insufficient. "
        "If image generation is unavailable or fails, report the failure; do not create a placeholder.\n\n"
        f"Image prompt:\n{instruction}\n"
    )


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def task_directory_name(task_id):
    """Map arbitrary task text to one stable, bounded directory component."""
    name = quote(task_id, safe="._-")
    if not name:
        return "%EMPTY"
    if name in {".", ".."}:
        return name.replace(".", "%2E")
    if len(name) > 200:
        # The reserved '%' prefix cannot collide with a URL-encoded user name.
        return "%sha256-" + hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return name


def model_turn(
    client, 
    prompt, 
    *, 
    model, 
    reasoning_effort, 
    timeout,
    task_dir, 
    stage, 
    run, 
    conversation_id=None, 
    workspace_dir=None
):
    job = client.submit(
        prompt, 
        model=model,
        reasoning_effort=reasoning_effort, conversation_id=conversation_id,
        workspace_dir=str(workspace_dir) if workspace_dir else None
    )
    run[stage] = {"job_id": job["id"], "status": "submitted"}
    write_json(task_dir / "task.json", run)
    print(f"{stage} job_id={job['id']}", file=sys.stderr, flush=True)

    def on_update(current):
        run[stage] = {**current, "job_id": current["id"]}
        write_json(task_dir / "task.json", run)
        activity = current.get("activity") or {}
        print(f"{stage} status={current['status']} "
              f"process_alive={activity.get('process_alive')} "
              f"elapsed={activity.get('elapsed_seconds')}s "
              f"last_activity={activity.get('last_activity_seconds_ago')}s ago",
              file=sys.stderr, flush=True)

    def on_retry(error):
        print(f"{stage}: 暂时无法查询任务状态，将重试查询：{type(error).__name__}",
              file=sys.stderr, flush=True)

    result = client.wait(job["id"], timeout=timeout, on_update=on_update, on_retry=on_retry)
    run[stage] = result
    write_json(task_dir / "task.json", run)
    write_json(task_dir / f"{stage}.response.json", result)
    if result["status"] != "completed":
        raise ValueError(f"{stage} Task {job['id']} Status: {result['status']}：{result.get('error')}")
    return result["response"]


def generate_one(
    prompt, 
    task_id=None, 
    output_dir="output", 
    *, 
    base_url="http://127.0.0.1:8000", 
    model=None, 
    reasoning_effort=None,
    timeout=None
):
    """Submit one prompt-to-image job and return its image path and task metadata."""
    instruction = text_or_file(prompt, "prompt")
    if instruction is None:
        raise ValueError("prompt cannot be empty.")
    model = model.strip() or None if model is not None else None
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout 必须为 None 或正的有限数值")
    root = Path(output_dir).expanduser().resolve()

    conversation_id = None
    if task_id is not None:
        task_id = str(task_id)
        try:
            conversation_id = task_id = str(UUID(task_id))
        except ValueError:
            pass
    with CodexClient(base_url, token=api_server_token(), timeout=60) as client:
        capabilities = client.capabilities()
        if not capabilities.get("reserve_conversation"):
            raise ValueError("Reserve conversation capability is not available on the server")
        allowed = [Path(p).resolve() for p in capabilities["workspace_write_roots"]]

        if not any(root.is_relative_to(p) for p in allowed):
            raise ValueError(f"The output directory {root} is not within any allowed workspace write root: {allowed}")
        if task_id is None:
            conversation_id = task_id = client.create_conversation()["conversation_id"]

        task_dir = root / task_directory_name(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        if task_dir.resolve() != task_dir:
            raise ValueError("任务目录不能是符号链接")
        print(f"task_id={task_id}\noutput_dir={task_dir}", file=sys.stderr, flush=True)

        with (task_dir / ".generation.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("A program for this task is already in progress.") from exc
            # Refuse to duplicate jobs after a previous caller timed out/disconnected.
            state_path = task_dir / "task.json"
            if state_path.exists():
                old = json.loads(state_path.read_text(encoding="utf-8"))
                if old.get("task_id") != task_id or not old.get("image_conversation_id"):
                    raise ValueError(f"任务记录缺少或包含不匹配的对话映射：{state_path}")
                saved_id = str(UUID(old["image_conversation_id"]))
                if conversation_id is not None and conversation_id != saved_id:
                    raise ValueError(f"任务目录中的图片对话 ID 与指定 UUID 不一致：{state_path}")
                conversation_id = saved_id
                for stage in ("image",):
                    stage_state = old.get(stage, {})
                    if stage_state.get("status") in {"submitted", "queued", "running", "cancelling"}:
                        previous_job_id = stage_state.get("job_id") or stage_state["id"]
                        job = client.get_job(previous_job_id)
                        if job["status"] not in {"completed", "failed", "cancelled", "interrupted"}:
                            raise ValueError(f"上一轮任务仍在运行：{previous_job_id}；请先等待完成")
            # Resolve named tasks while holding the directory lock so concurrent
            # callers cannot create two image conversations for the same name.
            if conversation_id is None:
                conversation_id = client.create_conversation()["conversation_id"]
            else:
                previous = client.http.get(f"/v1/conversations/{conversation_id}", params={"limit": 1})
                previous.raise_for_status()
            print(f"image_conversation_id={conversation_id}", file=sys.stderr, flush=True)
            image_path = task_dir / "image.png"
            if image_path.is_symlink():
                raise ValueError("输出图片不能是符号链接")
            run = {"task_id": task_id, "image_conversation_id": conversation_id,
                   "run_id": str(uuid4()), "prompt": instruction, "output_dir": str(task_dir)}
            write_json(state_path, run)
            image_prompt = build_prompt(instruction, image_path)
            (task_dir / "image_prompt.txt").write_text(image_prompt, encoding="utf-8")
            before = image_path.stat().st_mtime_ns if image_path.exists() else None
            response = model_turn(client, image_prompt,
                model=model, reasoning_effort=reasoning_effort, timeout=timeout,
                task_dir=task_dir, stage="image", run=run, conversation_id=conversation_id,
                workspace_dir=task_dir)

            if response["conversation_id"] != conversation_id:
                raise ValueError("服务端返回的图片对话 ID 与记录的对话 ID 不一致")
            if image_path.is_symlink():
                raise ValueError("输出图片不能是符号链接")
            if not image_path.is_file() or image_path.stat().st_mtime_ns == before:
                raise ValueError(f"Codex 未写入或更新目标图片：{image_path}；请查看 image.response.json")
            with Image.open(image_path) as generated:
                if generated.format != "PNG" or getattr(generated, "is_animated", False):
                    raise ValueError("生成结果必须是一张静态 PNG 图片")
                generated.verify()
            # Verify pixel data as well as the PNG container.
            with Image.open(image_path) as generated:
                generated.load()
            run["image_path"] = str(image_path)
            run["image_valid"] = True
            write_json(state_path, run)
            return run


def generate(prompt, task_id=None, output_dir="output", *, n=1, workers=4,
             base_url="http://127.0.0.1:8000", model=None, reasoning_effort=None,
             timeout=None):
    """Generate n images per prompt, preserving the single-image return format.

    Directory inputs always return a batch summary. Failed images are recorded in
    results while remaining jobs continue; a caller timeout does not cancel jobs.
    """
    for name, value in (("n", n), ("workers", workers)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    if prompt is None or not str(prompt).strip():
        raise ValueError("prompt cannot be empty.")
    if str(prompt) == "@":
        raise ValueError("@ 后必须提供 prompt 文件或目录路径")
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout 必须为 None 或正的有限数值")
    directory = None
    if prompt is not None and not str(prompt).startswith("text:"):
        value = str(prompt)
        candidate = Path(value[1:] if value.startswith("@") else value).expanduser()
        try:
            if candidate.is_dir():
                directory = candidate.resolve()
        except OSError as exc:
            if exc.errno not in {errno.ENAMETOOLONG, errno.EINVAL}:
                raise
    options = dict(base_url=base_url, model=model, reasoning_effort=reasoning_effort, timeout=timeout)
    if directory is None and n == 1:
        return generate_one(prompt, task_id=task_id, output_dir=output_dir, **options)

    root = Path(output_dir).expanduser().resolve()
    inputs = []
    if directory is not None:
        # Read and validate the whole input set before any paid generation starts.
        for path in sorted(directory.iterdir()):
            if path.is_file():
                instruction = path.read_text(encoding="utf-8").strip()
                if not instruction:
                    raise ValueError(f"prompt 文件不能为空：{path}")
                inputs.append((path.name, str(path), instruction))
        if not inputs:
            raise ValueError(f"prompt 目录中没有文件：{directory}")
        if task_id is not None:
            root = root / task_directory_name(str(task_id))
    else:
        instruction = text_or_file(prompt, "prompt")
        if instruction is None:
            raise ValueError("prompt cannot be empty.")
        name = str(task_id) if task_id is not None else str(uuid4())
        inputs.append((name, None, instruction))

    total = len(inputs) * n
    print(f"prompts={len(inputs)} n={n} total_images={total} workers={min(workers, total)}",
          file=sys.stderr, flush=True)

    def run_sample(item):
        name, source, instruction, index = item
        sample_root = root / task_directory_name(name)
        entry = {"prompt_file": source, "prompt_name": name, "sample_index": index,
                 "output_dir": str(sample_root / str(index)),
                 "id": name if n == 1 else f"{name}/{index}"}
        try:
            # Force literal text: file contents may themselves look like a path.
            result = generate_one("text:" + instruction, task_id=str(index),
                                  output_dir=sample_root, **options)
        except (httpx.HTTPError, OSError, ValueError, TimeoutError) as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"failed: {name}[{index}]: {exc}", file=sys.stderr, flush=True)
        else:
            entry.update(status="completed", **{key: result[key] for key in
                         ("task_id", "image_conversation_id", "image_path")})
        return entry

    samples = ((name, source, instruction, index)
               for name, source, instruction in inputs for index in range(n))
    with ThreadPoolExecutor(max_workers=min(workers, total)) as executor:
        results = list(executor.map(run_sample, samples))
    completed = sum(item["status"] == "completed" for item in results)
    return {"prompt_count": len(inputs), "images_per_prompt": n, "total": total,
            "completed": completed, "failed": total - completed,
            "failed_ids": [item["id"] for item in results if item["status"] == "failed"],
            "results": results}


def print_summary(completed, failed_ids):
    print(f"\n运行结束：成功 {completed} 张图片，失败 {len(failed_ids)} 张图片", file=sys.stderr, flush=True)
    print("失败 ID：" + json.dumps(failed_ids, ensure_ascii=False), file=sys.stderr, flush=True)


def report_error(args, message):
    print(message, file=sys.stderr, flush=True)
    # Batch job failures already return individual results. This branch handles
    # single-image errors or input validation failures before batch submission.
    value = str(args.prompt)
    candidate = Path(value[1:] if value.startswith("@") else value).expanduser()
    try:
        is_directory = not value.startswith("text:") and candidate.is_dir()
        is_file = not value.startswith("text:") and candidate.is_file()
    except OSError:
        is_directory = is_file = False
    if not is_directory and args.n == 1:
        failure_id = args.task_id if args.task_id is not None else candidate.name if is_file else "single_prompt"
        print_summary(0, [failure_id])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--prompt", required=True, help="prompt 文字、UTF-8 文件路径或包含多个 prompt 文件的目录")
    parser.add_argument("--n", "-n", type=int, default=1, help="每个 prompt 生成图片的数量，默认 1")
    parser.add_argument("--workers", type=int, default=4, help="批量模式最大并发任务数，默认 4")
    parser.add_argument(
        "--task-id", "--task_id", default=None, help="单图模式为图片对话 UUID 或任务名；批量模式为输出子目录名")
    parser.add_argument(
        "--output-dir", "--output_dir", "--output", 
        default="output", type=Path, help="输出根目录，默认 output")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model", default="gpt-6-astra", help="默认 gpt-6-astra；传入空字符串时使用服务端默认模型或续聊原模型")
    parser.add_argument(
        "--reasoning-effort", choices=["minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
        default="max", help="推理强度（默认 %(default)s）")
    parser.add_argument(
        "--timeout", type=float, default=None, help="可选的等待上限（秒）；默认持续查询任务状态，直到结束")
    args = parser.parse_args(argv)
    try:
        result = generate(**vars(args))
    except httpx.HTTPStatusError as exc:
        report_error(args, f"API 错误 {exc.response.status_code}：{exc.response.text}")
        parser.exit(1)
    except (httpx.RequestError, OSError, ValueError, TimeoutError) as exc:
        report_error(args, f"错误：{exc}")
        parser.exit(1)
    if "results" in result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print_summary(result["completed"], result["failed_ids"])
        if result["failed"]:
            parser.exit(1)
    else:
        print(json.dumps({key: result[key] for key in (
            "task_id", "image_conversation_id", "image_path")},
            ensure_ascii=False, indent=2))
        print_summary(1, [])


if __name__ == "__main__":
    main()
