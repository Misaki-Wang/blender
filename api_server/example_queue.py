"""几千任务的提交程序：自动分批提交，限制在途任务数，完成一条保存一条。

在 data_pipeline 下运行（实际调用模型）：
  api_server/.venv/bin/python api_server/example_queue.py --count 20
  api_server/.venv/bin/python api_server/example_queue.py --input tasks.jsonl --output results.jsonl

tasks.jsonl 每行是一个 /v1/chat 请求，例如：
  {"prompt":"请描述图片", "model":"gpt-5.6-luna", "images":[{"path":"/服务器路径/a.png"}]}
  {"prompt":"把 orange 翻译成中文"}

服务端并发由 CODEX_MAX_CONCURRENCY 控制；--batch-size 是 HTTP 提交分块大小。
"""
import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

import httpx

if __package__:
    from .client import CodexClient
else:
    from client import CodexClient


def requests_from_file(path, model):
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict) or not request.get("prompt"):
                    raise ValueError("每行必须是包含 prompt 的 JSON 对象")
            except ValueError as exc:
                raise ValueError(f"{path} 第 {line_number} 行无效：{exc}") from exc
            if model:
                request.setdefault("model", model)
            yield request


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="任务 JSONL 文件，逐行读取")
    parser.add_argument("--count", type=int, default=20, help="无输入文件时生成的演示任务数")
    parser.add_argument("--output", type=Path, default=Path("api_server/runtime/results.jsonl"))
    parser.add_argument("--model", help="模型 ID，省略时使用服务端默认值")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--batch-size", type=int, default=100, help="每次 HTTP 最多提交多少条")
    parser.add_argument("--max-in-flight", type=int, default=500, help="本程序同时在服务器上未完成的任务上限")
    parser.add_argument("--poll-interval", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=86400, help="整个提交程序等待的最长秒数")
    parser.add_argument("--run-id", default=None, help="固定运行 ID，用于中断后重复提交去重")
    args = parser.parse_args()
    if args.count < 1 or not 1 <= args.batch_size <= min(1000, args.max_in_flight):
        parser.error("count 必须为正数，batch-size 必须在 1 到 min(1000,max-in-flight) 之间")
    run_id = args.run_id or str(uuid4())
    print(f"run_id={run_id}（恢复提交时使用相同 ID、输入及 batch-size）", flush=True)
    if args.input:
        requests = requests_from_file(args.input, args.model)
    else:
        requests = ({"prompt":f"只回复 TASK_{i}，不要使用工具。", **({"model":args.model} if args.model else {})}
                    for i in range(args.count))
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Append so restarting the producer never truncates earlier results.
        # A repeated run may append duplicates: use (run_id,index) to deduplicate.
        with CodexClient(args.base_url, token=os.getenv("API_SERVER_TOKEN")) as client, args.output.open("a", encoding="utf-8") as output:
            print("队列状态：", json.dumps(client.queue_stats(), ensure_ascii=False), flush=True)
            completed = failed = 0
            for result in client.iter_results(requests, batch_size=args.batch_size, max_in_flight=args.max_in_flight,
                                              poll_interval=args.poll_interval, timeout=args.timeout, run_id=run_id):
                output.write(json.dumps({"run_id":run_id, **result}, ensure_ascii=False) + "\n")
                output.flush()
                completed += 1
                failed += result["status"] != "completed"
                print(f"已返回 {completed}，失败/取消 {failed}；index={result['index']} status={result['status']}", flush=True)
            print(f"结果已保存：{args.output}")
    except (httpx.HTTPError, OSError, ValueError, TimeoutError) as exc:
        parser.exit(1, f"调用失败：{exc}\n已接受任务仍可在服务端查询；run_id={run_id}\n")


if __name__ == "__main__":
    main()
