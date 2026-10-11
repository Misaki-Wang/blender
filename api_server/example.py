"""Codex API 调用示例（先启动 api_server/start.sh）。

在 data_pipeline 目录运行：
    api_server/.venv/bin/python api_server/example.py
    api_server/.venv/bin/python api_server/example.py --model gpt-5.6-luna --prompt "你好"
    api_server/.venv/bin/python api_server/example.py --example conversation
    api_server/.venv/bin/python api_server/example.py --example image --image /path/to/photo.png
    api_server/.venv/bin/python api_server/example.py --example batch
    api_server/.venv/bin/python api_server/example.py --example models

访问其他服务器时增加 --base-url http://服务器地址:8000。
启用鉴权时设置环境变量 API_SERVER_TOKEN。
"""

import argparse
import json
import os
from pathlib import Path

import httpx

# 同时支持 python api_server/example.py 和 python -m api_server.example。
if __package__:
    from .client import CodexClient
else:
    from client import CodexClient


def show(label, data):
    print(f"\n{label}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def text_example(client, model, prompt):
    # POST /v1/chat：不传 conversation_id 就创建独立新对话。
    # model=None 时使用服务端默认模型；也可以指定模型 ID。
    response = client.chat(
        prompt or "请用一句话介绍你自己。",
        model=model,
        new_conversation=True,
    )
    show("完整响应（answer 字段就是模型回答）：", response)


def conversation_example(client, model):
    # 第一轮：创建对话，保存返回的 conversation_id。
    first = client.chat("请记住：项目代号是 orange。", model=model)
    show("第一轮：", first)

    # 第二轮：传同一个 ID，模型就能看到此前的上下文。
    # 不传 model 时沿用这个会话的模型。
    second = client.chat(
        "刚才的项目代号是什么？",
        conversation_id=first["conversation_id"],
        new_conversation=False,
    )
    show("第二轮：", second)
    # 后续程序也可用这个 ID 续聊，无须再次发送历史消息。
    print(f"后续续聊使用 conversation_id={first['conversation_id']}")


def image_example(client, model, prompt, paths):
    # client.image() 读取调用端的图片并编码为 Base64，远程调用同样适用。
    # 一次可以传多张图片；续聊时也可以同时传 conversation_id 和 images。
    response = client.chat(
        prompt or "请描述这些图片的内容。",
        model=model,
        images=[client.image(path) for path in paths],
    )
    show("图文回答：", response)


def batch_example(client, model):
    # POST /v1/batch：每项是不带 conversation_id 的独立新对话。
    # 单批最多 32 项，服务端控制并发，返回顺序与输入顺序一致。
    requests = [
        {"prompt": "把 apple 翻译成中文。"},
        {"prompt": "把 orange 翻译成中文。"},
    ]
    if model:
        for request in requests:
            request["model"] = model

    for item in client.batch(requests):
        # HTTP 成功不代表批次内每项都成功，需要逐项检查 ok。
        if item["ok"]:
            show(f"第 {item['index'] + 1} 项成功：", item["response"])
        else:
            show(f"第 {item['index'] + 1} 项失败：", item)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example", choices=["text", "conversation", "image", "batch", "models"],
                        default="text", help="选择示例，默认 text")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API 服务地址")
    parser.add_argument("--model", help="模型 ID；省略时使用服务端默认模型")
    parser.add_argument("--prompt", help="自定义文字或图片示例的 prompt")
    parser.add_argument("--image", action="append", type=Path, default=[],
                        help="调用端图片路径，可重复传入以发送多张图片")
    args = parser.parse_args()
    if args.example == "image":
        if not 1 <= len(args.image) <= 8:
            parser.error("image 示例需要通过 --image 指定 1 至 8 张图片")
        for path in args.image:
            if not path.is_file():
                parser.error(f"图片文件不存在：{path}")
    elif args.image:
        parser.error("--image 需要配合 --example image")

    try:
        # 客户端会等待模型完成。较大的批次可按需要调大 timeout（秒）。
        with CodexClient(args.base_url, token=os.getenv("API_SERVER_TOKEN"), timeout=1800) as client:
            if args.example == "text":
                text_example(client, args.model, args.prompt)
            elif args.example == "conversation":
                conversation_example(client, args.model)
            elif args.example == "image":
                image_example(client, args.model, args.prompt, args.image)
            elif args.example == "batch":
                batch_example(client, args.model)
            else:
                # GET /v1/models：读取本机 Codex 缓存，不发起模型推理。
                show("模型列表：", client.models())
    except httpx.HTTPStatusError as exc:
        parser.exit(1, f"API 返回错误 {exc.response.status_code}：{exc.response.text}\n")
    except httpx.RequestError as exc:
        parser.exit(1, f"请求失败，请检查服务地址、服务状态或超时设置：{exc}\n")
    except OSError as exc:
        parser.exit(1, f"读取图片失败：{exc}\n")


if __name__ == "__main__":
    main()
