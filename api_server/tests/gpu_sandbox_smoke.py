#!/usr/bin/env python3
"""Opt-in real Codex sandbox + Blender check, without making model requests."""
import argparse
import errno
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api_server.gpu_sandbox import load_config, prepare_environment


def inner(workspace, outside, render_tool, uuid, host_socket, host_netns):
    root = Path(workspace)
    denied = []
    for path in [Path(outside), *(root / name / "probe" for name in (".git", ".codex", ".agents"))]:
        try:
            with path.open("w") as handle:
                handle.write("must not be written")
        except OSError as exc:
            assert exc.errno in (errno.EPERM, errno.EACCES, errno.EROFS), exc
            denied.append(str(path))
        else:
            raise AssertionError(f"Sandbox unexpectedly allowed writing {path}")
    (root / "allowed.txt").write_text("workspace write works")
    assert os.readlink('/proc/self/ns/net') != host_netns
    for family, kind in ((socket.AF_INET, socket.SOCK_STREAM),
                         (socket.AF_INET6, socket.SOCK_STREAM),
                         (socket.AF_INET, socket.SOCK_DGRAM),
                         (socket.AF_NETLINK, socket.SOCK_RAW),
                         (socket.AF_UNIX, socket.SOCK_DGRAM)):
        try:
            sock = socket.socket(family, kind)
        except OSError as exc:
            assert exc.errno == errno.EPERM, exc
        else:
            sock.close()
            raise AssertionError(f"Unexpected socket permission: {family}, {kind}")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        try:
            sock.connect(host_socket)
        except OSError as exc:
            assert exc.errno == errno.EPERM, exc
        else:
            raise AssertionError("Sandbox connected to a host Unix socket")
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        sock.bind('\0gpu-smoke-' + str(os.getpid()))
        sock.listen(1)
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    with left, right:
        left.sendmsg([b'local IPC works'])
        assert right.recv(128) == b'local IPC works'
        left.send(b'local send works')
        assert right.recv(128) == b'local send works'
    import asyncio
    async def check_thread_wakeup():
        assert await asyncio.to_thread(lambda: 42) == 42
    asyncio.run(check_thread_wakeup())
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True, timeout=15)
    gpus = result.stdout.splitlines()
    assert uuid in gpus
    tool = Path(render_tool)
    command = [sys.executable, "-B", str(tool), str(tool.parent / "examples/demo_scene.py"),
               "--views", str(tool.parent / "examples/views.json"), "--output-dir", str(root / "previews"),
               "--device", "AUTO", "--gpus", uuid, "--resolution", "64", "64", "--samples", "2",
               "--threads", "2", "--retries", "0", "--timeout", "120", "--wait-timeout", "10"]
    render = subprocess.run(command, capture_output=True, text=True, timeout=150)
    (root / "render-stdout.json").write_text(render.stdout)
    (root / "render-stderr.log").write_text(render.stderr)
    assert render.returncode == 0, render.stderr[-8000:]
    output = json.loads(render.stdout)
    assert output["device"]["backend"] in ("OPTIX", "CUDA"), output
    assert len(list((root / "previews").glob("*.png"))) == 4
    print(json.dumps({"status": "passed", "visible_gpus": gpus, "denied_writes": denied,
                      "network": "IP sockets and host Unix connections denied; local CUDA IPC allowed", "render_device": output["device"],
                      "manifest": output["manifest"]}, indent=2))


def main():
    if sys.argv[1:2] == ["--inside"]:
        inner(*sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--render-tool", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, check_devices=True)
    directory = Path(tempfile.mkdtemp(prefix="codex-gpu-smoke-"))
    workspace = directory / "workspace"
    workspace.mkdir()
    for name in (".git", ".codex", ".agents"):
        (workspace / name).mkdir()
    outside = directory / "protected.txt"
    outside.write_text("unchanged")
    environment = prepare_environment(config["codex"], workspace, args.config, os.environ)
    profile = ('permissions.gpu-smoke={extends=":workspace",filesystem={'
               '":slash_tmp"="read",":tmpdir"="read"},network={enabled=true}}')
    command = [config["codex"], "sandbox", "-C", str(workspace), "-P", "gpu-smoke",
               "-c", profile, "-c", "features.network_proxy=false", "--", sys.executable, "-B", str(Path(__file__).resolve()),
               "--inside", str(workspace), str(outside), str(args.render_tool.resolve()), config["gpus"][0]["uuid"],
               str(directory / 'host.sock'), os.readlink('/proc/self/ns/net')]
    print(f"GPU smoke artifacts: {directory}", flush=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as host_socket:
        host_socket.bind(str(directory / 'host.sock'))
        host_socket.listen(1)
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
    (directory / "stdout.log").write_text(result.stdout)
    (directory / "stderr.log").write_text(result.stderr)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    assert result.returncode == 0, f"Sandbox GPU smoke failed: {directory}"
    assert outside.read_text() == "unchanged"


if __name__ == "__main__":
    main()
