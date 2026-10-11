#!/usr/bin/env python3
"""图片 + prompt -> output/{task_id}/blender_code.py。

在 data_pipeline 目录运行：
  api_server/.venv/bin/python scripts/blender_gen.py --image photo.png \\
    --prompt "重建图片中的场景" --output-dir output
  api_server/.venv/bin/python scripts/blender_gen.py \\
    --image output/images/piano --output-dir output/blender/piano --workers 4

目录输入递归读取 PNG/JPEG/WebP 图片，使用独立进程并行处理，默认 4 个进程。
保留图片所在的相对目录结构，例如：
  输入 piano_0/0/image.png -> 输出 piano_0/0/blender_code.py
每个任务目录同时保存输入图片为 input_image.png（JPEG 为 .jpg，WebP 为 .webp）。
若同一目录有多张图片，则为各图片增加完整文件名子目录，避免覆盖。
批量模式的 task_id 指定额外的输出子目录，prompt 和 scene_description 为全部图片共用。
--description-mode combined 可在一次请求里生成描述和代码；默认保留两阶段模式。

task_id 可为代码对话 UUID 或自定义任务名；省略时自动创建，传入时续聊。
自定义任务名与代码对话 UUID 的对应关系保存在 output/{task_id}/task.json。
scene_description 省略时按 description_mode 生成；提供描述时直接使用。
prompt 和 scene_description 均支持直接文本、UTF-8 文件路径或 @文件路径。
需要 API 服务 2.2+，且调用端和服务端能访问同一个输出目录。
"""

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import errno
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
from uuid import UUID, uuid4
from urllib.parse import quote

import httpx
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_server.client import CodexClient

DEFAULT_TEMPLATE = Path(__file__).resolve().parents[1] / "prompts/blender_code_generation.txt"
OUTPUT_CONTRACT = DEFAULT_TEMPLATE.with_name("blender_code_contract.txt")
CATEGORY_SCALE_HINTS = {
    "banana": "longest dimension about 0.15-0.25 m",
    "book": "longest dimension about 0.20-0.35 m",
    "bowl": "diameter about 0.15-0.30 m",
    "cup": "height about 0.08-0.15 m",
    "mug": "height about 0.08-0.15 m",
    "bottle": "height about 0.20-0.35 m",
    "wine_glass": "height about 0.15-0.25 m",
    "orange": "diameter about 0.07-0.10 m",
    "apple": "diameter about 0.07-0.10 m",
    "camera": "longest dimension about 0.10-0.20 m",
    "fork": "length about 0.18-0.22 m",
    "spoon": "length about 0.15-0.22 m",
    "plate": "diameter about 0.20-0.30 m",
    "candle": "height about 0.10-0.25 m",
    "sandwich": "longest dimension about 0.10-0.20 m",
    "sports_ball": "diameter about 0.20-0.25 m",
}

DESCRIPTION_PROMPT = (
    "Describe the scene in the attached reference image for a Blender 5.0 "
    "procedural reconstruction task. Return only the scene description as plain English text. "
    "Include visible objects and their parts, relative sizes, spatial relationships, "
    "shapes, repeated structures and counts, colors, materials, camera view, background, "
    "lighting and shadows. Distinguish visible details from reasonable guesses about "
    "occluded geometry. Be specific enough that a separate coding agent can reconstruct the scene.\n"
    "Write the description directly to {} using your file tools. "
    "Do not repeat it in chat; reply only with the saved path."
)

def api_server_token():
    token = os.environ.get("API_SERVER_TOKEN")
    if token is not None:
        return token.strip() or None
    for directory in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
        path = directory / "API_SERVER_TOKEN"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip() or None
    return None


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


def build_prompt(
    instruction, description, task_dir, image
):
    instruction = instruction or ""
    # Replace known placeholders in one pass; preserve arbitrary braces in examples.
    template = instruction if "{scene_description}" in instruction else DEFAULT_TEMPLATE.read_text(encoding="utf-8")
    # Apply the same script/artifact requirements to default and custom prompts.
    template += "\n\n" + OUTPUT_CONTRACT.read_text(encoding="utf-8")
    values = {
        "prompt": instruction, 
        "scene_description": description,
        "output_dir": str(task_dir), 
        "code_path": str(task_dir / "blender_code.py"),
        "image": str(image)
    }
    rendered = re.sub(r"\{(prompt|scene_description|output_dir|code_path|image)\}",
                      lambda match: values[match[1]], template)
    if instruction and "{scene_description}" not in instruction and "{prompt}" not in template:
        rendered += f"\n\n# Task instruction\n{instruction}\n"
    category = task_dir.parent.parent.name if task_dir.parent.name == "0" else task_dir.parent.name
    scale_hint = CATEGORY_SCALE_HINTS.get(category)
    if scale_hint:
        rendered += (
            "\n\n# Category scale reference\n"
            f"This is a {category} asset. Use metric units and make the target object's "
            f"largest physical dimension {scale_hint}. Keep all secondary parts proportional. "
            "Do not use camera framing to change this size.\n"
        )
    # Keep the output contract even when a caller supplies a complete custom template.
    return rendered + (
        f"\n\nRequired output file: {task_dir / 'blender_code.py'}\n"
        "Write the complete Blender Python code directly into that file with your file tools. "
        "A code-only chat response is insufficient. You may run Blender in background mode "
        "to validate the code, make test renders, and fix errors. "
        f"Keep temporary validation artifacts inside {task_dir / '_debug'} and follow "
        "the script/artifact contract above. "
        "Do not repeat the script or scene description in chat. "
        "Reply in at most three lines with saved paths and validation results.\n"
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
    image, 
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
        images=[image], model=model,
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
        if (current["status"] == "running"
                and activity.get("process_alive") is True
                and not current.get("error")):
            return
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
    image, 
    task_id=None, 
    output_dir="output", 
    scene_description=None,
    *, 
    base_url="http://127.0.0.1:8000", 
    model=None, 
    reasoning_effort=None,
    timeout=None,
    description_mode="separate",
    _task_dir=None
):
    """
    Generate Blender Python code from a single reference image and a prompt. 
    Returns task_id, code path, description path, and information about the two conversations.
    """
    instruction = text_or_file(prompt, "prompt") or ""
    description = text_or_file(scene_description, "scene_description")
    if description_mode not in {"combined", "separate"}:
        raise ValueError("description_mode must be combined or separate")
    model = model.strip() or None if model is not None else None
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout 必须为 None 或正的有限数值")
    image_path = Path(image).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()

    conversation_id = None
    if task_id is not None:
        task_id = str(task_id)
        try:
            conversation_id = task_id = str(UUID(task_id))
        except ValueError:
            pass
    if image_path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("The reference image must be no larger than 10 MiB")
    with Image.open(image_path) as reference:
        if reference.format not in {"PNG", "JPEG", "WEBP"} or getattr(reference, "is_animated", False):
            raise ValueError("Only a single PNG, JPEG, or WebP image is supported")
        image_suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[reference.format]
        reference.verify()

    with CodexClient(base_url, token=api_server_token(), timeout=60) as client:
        capabilities = client.capabilities()
        if not capabilities.get("reserve_conversation"):
            raise ValueError("需要支持对话预创建的 API 服务 2.2+")
        allowed = [Path(p).resolve() for p in capabilities["workspace_write_roots"]]

        if not any(root.is_relative_to(p) for p in allowed):
            raise ValueError(f"输出目录不在服务端 CODEX_WRITE_ROOTS 中：{root}；允许目录：{allowed}")
        if task_id is None:
            conversation_id = task_id = client.create_conversation()["conversation_id"]

        task_dir = Path(_task_dir) if _task_dir is not None else root / task_directory_name(task_id)
        if not task_dir.is_absolute() or not task_dir.is_relative_to(root):
            raise ValueError("任务目录必须位于输出目录内")
        task_dir.mkdir(parents=True, exist_ok=True)
        if task_dir.resolve() != task_dir:
            raise ValueError("任务目录不能是符号链接")
        print(f"task_id={task_id}\noutput_dir={task_dir}", file=sys.stderr, flush=True)

        with (task_dir / ".generation.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("同一任务已有调用程序正在运行") from exc
            # Refuse to duplicate jobs after a previous caller timed out/disconnected.
            state_path = task_dir / "task.json"
            if state_path.exists():
                old = json.loads(state_path.read_text(encoding="utf-8"))
                if old.get("task_id") != task_id or not old.get("code_conversation_id"):
                    raise ValueError(f"任务记录缺少或包含不匹配的对话映射：{state_path}")
                saved_id = str(UUID(old["code_conversation_id"]))
                if conversation_id is not None and conversation_id != saved_id:
                    raise ValueError(f"任务目录中的代码对话 ID 与指定 UUID 不一致：{state_path}")
                conversation_id = saved_id
                for stage in ("description", "code"):
                    stage_state = old.get(stage, {})
                    if stage_state.get("status") in {"submitted", "queued", "running", "cancelling"}:
                        previous_job_id = stage_state.get("job_id") or stage_state["id"]
                        job = client.get_job(previous_job_id)
                        if job["status"] not in {"completed", "failed", "cancelled", "interrupted"}:
                            raise ValueError(f"上一轮任务仍在运行：{previous_job_id}；请先等待完成")
            # Resolve named tasks while holding the directory lock so concurrent
            # callers cannot create two code conversations for the same name.
            if conversation_id is None:
                conversation_id = client.create_conversation()["conversation_id"]
            else:
                previous = client.http.get(f"/v1/conversations/{conversation_id}", params={"limit": 1})
                previous.raise_for_status()
            print(f"code_conversation_id={conversation_id}", file=sys.stderr, flush=True)
            input_image_path = task_dir / f"input_image{image_suffix}"
            if input_image_path.is_symlink():
                raise ValueError("输入图片副本不能是符号链接")
            if not input_image_path.exists() or not image_path.samefile(input_image_path):
                shutil.copy2(image_path, input_image_path)
            encoded_image = CodexClient.image(input_image_path)
            run = {"task_id": task_id, "code_conversation_id": conversation_id,
                   "run_id": str(uuid4()), "image": str(image_path),
                   "input_image_path": str(input_image_path), "output_dir": str(task_dir),
                   "description_mode": "provided" if description is not None else description_mode}
            write_json(state_path, run)

            scene_description_path = task_dir / "scene_description"
            before_description = scene_description_path.stat().st_mtime_ns if scene_description_path.exists() else None
            combined_description = description is None and description_mode == "combined"
            if description is None and not combined_description:
                response = model_turn(
                    client, 
                    DESCRIPTION_PROMPT.format(scene_description_path), 
                    encoded_image,
                    model=model, 
                    reasoning_effort=reasoning_effort, 
                    timeout=timeout,
                    task_dir=task_dir, 
                    stage="description", 
                    run=run,
                    workspace_dir=task_dir
                )
                if not scene_description_path.is_file() or scene_description_path.stat().st_mtime_ns == before_description:
                    raise ValueError(f"Codex 未写入或更新场景描述文件：{scene_description_path}")
                if scene_description_path.is_symlink() or scene_description_path.stat().st_size > 1024 * 1024:
                    raise ValueError("场景描述必须是任务目录内不超过 1 MiB 的普通文件")
                description = scene_description_path.read_text(encoding="utf-8").strip()
                if not description:
                    raise ValueError("场景描述对话返回了空文本")
                run["description_conversation_id"] = response["conversation_id"]
            elif description is not None:
                run["description_conversation_id"] = None
                scene_description_path.write_text(description + "\n", encoding="utf-8")
            else:
                run["description_conversation_id"] = conversation_id
                description = (
                    "Infer the scene directly from the attached reference image in this request. "
                    "Identify visible objects, parts, proportions, materials and spatial relationships. "
                    "Use those observations to reconstruct the scene; distinguish occluded details from visible ones."
                )
            
            code_prompt = build_prompt(
                instruction, description, task_dir, input_image_path
            )
            if combined_description:
                code_prompt += (
                    f"\nSave your concise scene observations as plain English text to {scene_description_path} "
                    "in this same request, before writing the Blender code. "
                    "Keep only reconstruction-relevant details; do not repeat the description in chat.\n"
                )
            (task_dir / "code_prompt.txt").write_text(code_prompt, encoding="utf-8")
            write_json(state_path, run)
            code_path = task_dir / "blender_code.py"
            before = code_path.stat().st_mtime_ns if code_path.exists() else None
            response = model_turn(client, code_prompt, encoded_image,
                model=model, reasoning_effort=reasoning_effort, timeout=timeout,
                task_dir=task_dir, stage="code", run=run, conversation_id=conversation_id,
                workspace_dir=task_dir)
            
            if response["conversation_id"] != conversation_id:
                raise ValueError("服务端返回的代码对话 ID 与记录的对话 ID 不一致")
            if combined_description:
                if not scene_description_path.is_file() or scene_description_path.stat().st_mtime_ns == before_description:
                    raise ValueError(f"模型未写入或更新场景描述文件：{scene_description_path}")
                if scene_description_path.is_symlink() or scene_description_path.stat().st_size > 1024 * 1024:
                    raise ValueError("场景描述必须是任务目录内不超过 1 MiB 的普通文件")
                if not scene_description_path.read_text(encoding="utf-8").strip():
                    raise ValueError("场景描述为空")
            if not code_path.is_file() or code_path.stat().st_mtime_ns == before:
                raise ValueError(f"Codex 未写入或更新目标代码文件：{code_path}；请查看 code.response.json")
            if code_path.is_symlink() or code_path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("生成代码必须是任务目录内不超过 8 MiB 的普通文件")
            
            code = code_path.read_text(encoding="utf-8")
            tree = ast.parse(code, filename=str(code_path))
            if not any((isinstance(n, ast.Import) and any(a.name == "bpy" for a in n.names))
                       or (isinstance(n, ast.ImportFrom) and n.module == "bpy") for n in ast.walk(tree)):
                raise ValueError("生成代码没有导入 bpy")
            run["code_path"] = str(code_path)
            run["scene_description_path"] = str(task_dir / "scene_description")
            run["syntax_valid"] = True
            write_json(state_path, run)
            return run


def _generate_image(options):
    """Picklable process entry point; create all HTTP clients inside this process."""
    entry = {"image": str(options["image"]), "task_id": options["task_id"],
             "output_dir": str(options.get("_task_dir") or
                               Path(options["output_dir"]) / task_directory_name(options["task_id"])),
             "worker_pid": os.getpid()}
    try:
        result = generate_one(**options)
    except (httpx.HTTPError, OSError, ValueError, SyntaxError, TimeoutError) as exc:
        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"failed: {entry['image']}: {exc}", file=sys.stderr, flush=True)
    else:
        entry.update(status="completed", **{key: result[key] for key in
                     ("code_conversation_id", "code_path", "input_image_path", "scene_description_path", "description_conversation_id")})
    return entry


def generate(prompt, image, task_id=None, output_dir="output", scene_description=None,
             *, workers=4, base_url="http://127.0.0.1:8000", model=None,
             reasoning_effort=None, timeout=None, description_mode="separate"):
    """Process one image directly, or a directory recursively using spawn workers.

    A single image keeps the original return format and conversation semantics.
    Batch results include each image's status and output paths. Reusing the same
    output directory resumes the corresponding conversations and updates code.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers 必须为正整数")
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout 必须为 None 或正的有限数值")
    if image is None or not str(image).strip():
        raise ValueError("image 不能为空")
    source = Path(image).expanduser().resolve()
    options = dict(prompt=prompt, scene_description=scene_description, base_url=base_url,
                   model=model, reasoning_effort=reasoning_effort, timeout=timeout,
                   description_mode=description_mode)
    if not source.is_dir():
        return generate_one(image=source, task_id=task_id, output_dir=output_dir, **options)

    root = Path(output_dir).expanduser().resolve()
    if task_id is not None:
        root = root / task_directory_name(str(task_id))
    images = [path for path in sorted(source.rglob("*"))
              if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
              and not (root != source and path.is_relative_to(root))]
    if not images:
        raise ValueError(f"目录中没有 PNG/JPEG/WebP 图片：{source}")
    # Snapshot common text inputs before starting any child processes.
    for key in ("prompt", "scene_description"):
        value = text_or_file(options[key], key)
        options[key] = "text:" + value if value is not None else None

    tasks = []
    images_per_directory = Counter(path.parent for path in images)
    destinations = set()
    for path in images:
        relative = path.relative_to(source)
        # image_gen emits one image per sample directory. Mirror that directory
        # directly, preserving literal names (including spaces and percent signs).
        destination = root / (relative.parent if images_per_directory[path.parent] == 1 else relative)
        if destination in destinations:
            raise ValueError(f"多张图片对应相同输出目录：{destination}")
        destinations.add(destination)
        tasks.append({**options, "image": str(path), "task_id": relative.as_posix(),
                      "output_dir": str(root), "_task_dir": str(destination)})
    print(f"images={len(tasks)} workers={min(workers, len(tasks))}", file=sys.stderr, flush=True)
    # spawn avoids inheriting HTTP connections, open locks, or server threads.
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        results = list(executor.map(_generate_image, tasks, chunksize=1))
    for entry in results:
        entry["id"] = Path(entry["image"]).relative_to(source).as_posix()
    completed = sum(entry["status"] == "completed" for entry in results)
    return {"total": len(tasks), "completed": completed, "failed": len(tasks) - completed,
            "failed_ids": [entry["id"] for entry in results if entry["status"] == "failed"],
            "results": results}


def print_summary(completed, failed_ids):
    print(f"\n运行结束：成功 {completed} 个任务，失败 {len(failed_ids)} 个任务", file=sys.stderr, flush=True)
    print("失败 ID：" + json.dumps(failed_ids, ensure_ascii=False), file=sys.stderr, flush=True)


def report_error(args, message):
    print(message, file=sys.stderr, flush=True)
    if not args.image.expanduser().is_dir():
        print_summary(0, [args.task_id if args.task_id is not None else args.image.name])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--prompt", default=None, help="任务文字或 UTF-8 prompt 文件路径")
    parser.add_argument(
        "--image", required=True, type=Path, help="一张 PNG/JPEG/WebP 图片或图片目录（递归读取）")
    parser.add_argument("--workers", type=int, default=4, help="目录模式的并行进程数，默认 4")
    parser.add_argument(
        "--task-id", "--task_id", default=None, help="单图模式为代码对话 UUID 或任务名；目录模式为输出子目录名")
    parser.add_argument(
        "--output-dir", "--output_dir", "--output", 
        default="output", type=Path, help="输出根目录，默认 output")
    parser.add_argument(
        "--scene-description", "--scene_description", 
        default=None, help="描述文字或文件路径；省略时根据 description-mode 生成")
    parser.add_argument(
        "--description-mode", choices=["combined", "separate"], default="separate",
        help="combined 一次请求生成描述和代码；separate 先用独立请求描述图片（默认）")
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
    except (httpx.RequestError, OSError, ValueError, SyntaxError, TimeoutError) as exc:
        report_error(args, f"错误：{exc}")
        parser.exit(1)
    if "results" in result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print_summary(result["completed"], result["failed_ids"])
        if result["failed"]:
            parser.exit(1)
    else:
        print(json.dumps({key: result[key] for key in (
            "task_id", "code_conversation_id", "code_path", "input_image_path", "scene_description_path", "description_conversation_id")},
            ensure_ascii=False, indent=2))
        print_summary(1, [])


if __name__ == "__main__":
    main()
