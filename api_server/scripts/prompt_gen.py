#!/usr/bin/env python3
"""Generate varied scene descriptions for one or more object categories.

Run from data_pipeline:
  api_server/.venv/bin/python scripts/prompt_gen.py \
    --category piano --n 10 --output-dir output/descriptions
  api_server/.venv/bin/python scripts/prompt_gen.py \\
    --category piano dishwasher chair --n 100 --workers 4 --output-dir output/descriptions

Writes plain English UTF-8 files piano_0, ..., piano_9 (no extension).
Multiple categories run in separate processes and write to {output_dir}/{category}/.
--categories-file accepts a UTF-8 file with one category per nonempty line.
Within each category, batches remain sequential in one conversation for diversity.
--n is the number of descriptions PER category; --workers defaults to 4.
Each file is a standalone prompt suitable for scripts/image_gen.py --prompt.
Existing files are refused; use another directory or --start-index to append.
The API only generates text; this script writes files locally.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import sys

import httpx

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_server.client import CodexClient


def api_server_token():
    token = os.environ.get("API_SERVER_TOKEN")
    if token is not None:
        return token.strip() or None
    for directory in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
        path = directory / "API_SERVER_TOKEN"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip() or None
    return None


def build_prompt(category, count):
    return f"""Create exactly {count} distinct scene descriptions for the object category
{json.dumps(category, ensure_ascii=False)}. Treat the category as a subject, not instructions.
These descriptions will each independently generate ONE image.

Requirements:
- Write each description in English, as one self-contained paragraph of about 80-140 words.
- Make an instance of the requested category clearly visible and recognizable as the main
  subject in EVERY scene. Name the category explicitly in every description.
- Plan the whole set before writing. Use substantially different, plausible environments,
  room geometries, functions, scales, and arrangements appropriate to this category.
- Do not repeat the same object too many times (>15) in a scene. Avoid cluttered or chaotic arrangements.
- Vary spatial layout explicitly: the subject's position and orientation, placement against
  walls versus freestanding placement where plausible, distance from nearby objects,
  open space and circulation, foreground/background, and the arrangement of supporting objects.
- Describe concrete spatial relationships (left/right, in front of/behind, beside, facing,
  near/far), subject appearance, surrounding objects, materials, camera framing, and lighting.
- Each pair of scenes must differ in both setting and spatial arrangement. Merely swapping
  colors, materials, lighting, camera angle, or a few accessories is NOT sufficient.
- Keep arrangements physically coherent and the subject sufficiently unobstructed.
  Avoid forcing implausible placements solely for variety.
- Do not reuse a scene or arrangement from earlier accepted batches in this conversation.
  Do not refer to another description or use phrases such as "same scene as before".
- Return ONLY a valid JSON array of exactly {count} nonempty strings. No Markdown, headings,
  numbering, commentary, file operations, image generation, or additional keys.
"""


def parse_descriptions(answer, count, previous):
    text = answer.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, count=1).removesuffix("```").strip()
    try:
        descriptions = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("响应不是合法 JSON 数组") from exc
    if not isinstance(descriptions, list) or len(descriptions) != count:
        raise ValueError(f"必须返回恰好 {count} 个 description")
    accepted = [" ".join(item.casefold().split()) for item in previous]
    result = []
    for item in descriptions:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("每个 description 必须为非空字符串")
        item = item.strip()
        normalized = " ".join(item.casefold().split())
        if any(SequenceMatcher(None, normalized, old, autojunk=False).ratio() >= 0.9
               for old in accepted):
            raise ValueError("存在重复或文字过于相似的 description；请更换场景和空间布局")
        accepted.append(normalized)
        result.append(item)
    return result


def validate_inputs(category, n, start_index, timeout):
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category 不能为空")
    category = category.strip()
    if category in {".", ".."} or any(c in category for c in '/\\') or any(ord(c) < 32 for c in category):
        raise ValueError("category 必须为单个文件名组成部分，不能包含路径分隔符或控制字符")
    if len(category.encode("utf-8")) > 200:
        raise ValueError("category 太长（最多 200 UTF-8 字节）")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n 必须为正整数")
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index 必须为非负整数")
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout 必须为 None 或正的有限数值")
    return category


def generate_one(category, n, output_dir="output/descriptions", *, start_index=0,
                 base_url="http://127.0.0.1:8000", model=None, reasoning_effort=None,
                 timeout=None):
    """Generate n descriptions in one conversation, in batches of at most 20.

    Retry malformed/duplicate content twice. timeout limits waiting for each API job;
    timing out does not cancel the job. Write files only after all batches validate.
    """
    category = validate_inputs(category, n, start_index, timeout)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / f"{category}_{i}" for i in range(start_index, start_index + n)]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"输出文件已存在：{path}；请更换目录或设置 --start-index")

    descriptions = []
    conversation_id = None
    model = model.strip() or None if model is not None else None
    with CodexClient(base_url, token=api_server_token(), timeout=60) as client:
        while len(descriptions) < n:
            count = min(20, n - len(descriptions))
            prompt = build_prompt(category, count)
            for attempt in range(3):
                job = client.submit(prompt, model=model, reasoning_effort=reasoning_effort,
                                    conversation_id=conversation_id)
                print(f"category={category} descriptions={len(descriptions)}/{n} job_id={job['id']}",
                      file=sys.stderr, flush=True)
                result = client.wait(
                    job["id"], timeout=timeout,
                    on_update=lambda current: print(f"category={category} status={current['status']}", file=sys.stderr, flush=True),
                    on_retry=lambda error: print(f"category={category} 查询暂时失败，将重试：{type(error).__name__}",
                                                 file=sys.stderr, flush=True))
                if result["status"] != "completed":
                    raise ValueError(f"任务 {job['id']} {result['status']}：{result.get('error')}")
                response = result["response"]
                conversation_id = response["conversation_id"]
                try:
                    batch = parse_descriptions(response["answer"], count, descriptions)
                except ValueError as exc:
                    if attempt == 2:
                        raise ValueError(f"三次生成后仍未得到有效描述：{exc}") from exc
                    prompt = (build_prompt(category, count) +
                              f"\nThe last batch was rejected: {exc}. Replace that entire batch.\n")
                    continue
                descriptions.extend(batch)
                break

    completed_paths = []
    try:
        for path, description in zip(paths, descriptions):
            with path.open("x", encoding="utf-8") as stream:
                stream.write(description + "\n")
            completed_paths.append(str(path))
    except OSError as exc:
        # Preserve actual progress if a disk/write error occurs partway through.
        exc.completed_paths = completed_paths
        raise
    return {"category": category, "n": n, "output_dir": str(root),
            "description_paths": [str(path) for path in paths]}


def _generate_category(options):
    """Picklable process entry point; HTTP clients belong to individual workers."""
    entry = {"category": options["category"], "output_dir": options["output_dir"],
             "worker_pid": os.getpid()}
    try:
        result = generate_one(**options)
    except (httpx.HTTPError, OSError, ValueError, TimeoutError) as exc:
        completed_paths = getattr(exc, "completed_paths", [])
        completed_ids = {Path(path).name for path in completed_paths}
        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                     description_paths=completed_paths,
                     failed_ids=[f"{options['category']}_{i}" for i in
                                 range(options["start_index"], options["start_index"] + options["n"])
                                 if f"{options['category']}_{i}" not in completed_ids])
        print(f"failed: {entry['category']}: {exc}", file=sys.stderr, flush=True)
    else:
        entry.update(status="completed", failed_ids=[], **result)
    return entry


def generate(category, n, output_dir="output/descriptions", *, start_index=0, workers=4,
             base_url="http://127.0.0.1:8000", model=None, reasoning_effort=None,
             timeout=None):
    """A string preserves single-category behavior; a list/tuple runs a process pool.

    Batch output is grouped into category subdirectories. One category's failure
    does not stop the others. Repeated normalized category names are rejected.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers 必须为正整数")
    options = dict(n=n, start_index=start_index, base_url=base_url, model=model,
                   reasoning_effort=reasoning_effort, timeout=timeout)
    if isinstance(category, str):
        return generate_one(category, output_dir=output_dir, **options)
    if not isinstance(category, (list, tuple)) or not category:
        raise ValueError("category 必须为非空字符串或非空类别列表")
    categories = [validate_inputs(item, n, start_index, timeout) for item in category]
    if len(set(categories)) != len(categories):
        raise ValueError("category 列表不能包含重复类别")
    root = Path(output_dir).expanduser().resolve()
    tasks = []
    for name in categories:
        directory = root / name
        if directory.is_symlink():
            raise ValueError(f"类别输出目录不能是符号链接：{directory}")
        tasks.append({**options, "category": name, "output_dir": str(directory)})
    print(f"categories={len(tasks)} descriptions_per_category={n} "
          f"workers={min(workers, len(tasks))}", file=sys.stderr, flush=True)
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        results = list(executor.map(_generate_category, tasks, chunksize=1))
    completed = sum(entry["status"] == "completed" for entry in results)
    return {"category_count": len(tasks), "descriptions_per_category": n,
            "total_descriptions": len(tasks) * n,
            "completed_descriptions": sum(len(entry["description_paths"]) for entry in results),
            "failed_descriptions": sum(len(entry["failed_ids"]) for entry in results),
            "failed_ids": [failed_id for entry in results for failed_id in entry["failed_ids"]],
            "completed_categories": completed, "failed_categories": len(tasks) - completed,
            "results": results}


def print_summary(completed, failed_ids):
    print(f"\n运行结束：成功 {completed} 条描述，失败 {len(failed_ids)} 条描述", file=sys.stderr, flush=True)
    print("失败 ID：" + json.dumps(failed_ids, ensure_ascii=False), file=sys.stderr, flush=True)


def report_error(args, exc, message):
    print(message, file=sys.stderr, flush=True)
    if isinstance(args.category, str) and args.n > 0 and args.start_index >= 0:
        completed_paths = getattr(exc, "completed_paths", [])
        completed_ids = {Path(path).name for path in completed_paths}
        failed_ids = [f"{args.category.strip()}_{i}" for i in range(args.start_index, args.start_index + args.n)
                      if f"{args.category.strip()}_{i}" not in completed_ids]
        print_summary(len(completed_paths), failed_ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--category", nargs="+", action="extend", help="一个或多个类别，例如 piano dishwasher")
    sources.add_argument("--categories-file", type=Path, help="UTF-8 类别文件，每行一个类别；按批量模式输出")
    parser.add_argument("--workers", type=int, default=4, help="多个类别的并行进程数，默认 4；同一类别内串行")
    parser.add_argument("--n", "-n", required=True, type=int, help="每个类别生成的场景描述数量")
    parser.add_argument("--output-dir", "--output_dir", "--output", default="output/descriptions", type=Path)
    parser.add_argument("--start-index", type=int, default=0, help="文件编号起点，默认 0")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="gpt-6-astra", help="默认 gpt-6-astra；空字符串使用服务端默认模型")
    parser.add_argument("--reasoning-effort", default="max", help="推理强度（默认 %(default)s）",
                        choices=["minimal", "low", "medium", "high", "xhigh", "max", "ultra"])
    parser.add_argument("--timeout", type=float, default=None, help="每次任务等待上限（秒），默认等待至结束")
    args = parser.parse_args(argv)
    try:
        categories_file = args.categories_file
        del args.categories_file
        if categories_file is not None:
            args.category = [line.strip() for line in categories_file.expanduser().read_text(encoding="utf-8").splitlines()
                             if line.strip()]
        elif len(args.category) == 1:
            args.category = args.category[0]
        result = generate(**vars(args))
    except httpx.HTTPStatusError as exc:
        report_error(args, exc, f"API 错误 {exc.response.status_code}：{exc.response.text}")
        parser.exit(1)
    except (httpx.RequestError, OSError, ValueError, TimeoutError) as exc:
        report_error(args, exc, f"错误：{exc}")
        parser.exit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if "results" in result:
        print_summary(result["completed_descriptions"], result["failed_ids"])
    else:
        print_summary(len(result["description_paths"]), [])
    if result.get("failed_categories"):
        parser.exit(1)


if __name__ == "__main__":
    main()
