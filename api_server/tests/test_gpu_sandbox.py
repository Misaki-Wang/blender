import json
import os
from pathlib import Path
import socket
import struct
import sys

import pytest

from api_server.gpu_sandbox import (
    CONFIG_ENV, WRAPPER_DIR, add_gpu_mounts, add_network_filter, device_record, load_config,
    network_filter_bytes, prepare_environment, select_devices,
)


def test_mounts_preserve_policy_descriptors_and_later_denials():
    args = ["--ro-bind", "/", "/", "--dev", "/dev", "--bind", "/task", "/task",
            "--ro-bind", "/task/.codex", "/task/.codex", "--seccomp", "9",
            "--tmpfs", "/dev/nvidia0", "--unshare-net", "--cap-drop", "ALL",
            "--", "/bin/sh", "-c", "echo --dev /dev"]
    devices = ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"]
    actual = add_gpu_mounts(args, devices)
    added = [v for d in devices for v in ("--dev-bind", d, d)]
    assert actual == args[:5] + added + args[5:]
    assert actual.index("--tmpfs") > actual.index("--dev-bind")
    assert args[5:7] == ["--bind", "/task"]  # caller's argv is not mutated


def test_separator_in_option_value_does_not_confuse_mount_insertion():
    args = ["--setenv", "EXAMPLE", "--", "--dev", "/dev", "--", "/bin/true"]
    assert add_gpu_mounts(args, ["/dev/nvidia3"])[5:8] == [
        "--dev-bind", "/dev/nvidia3", "/dev/nvidia3"]


@pytest.mark.parametrize("args", [
    ["--args", "3", "--", "/bin/true"],
    ["--dev", "/dev"],
    ["--dev", "/dev", "--"],
    ["--ro-bind", "/", "/", "--", "/bin/true"],
    ["--dev", "/dev", "--dev", "/dev", "--", "/bin/true"],
])
def test_unknown_or_unexpected_sandbox_layout_fails_closed(args):
    with pytest.raises(ValueError):
        add_gpu_mounts(args, ["/dev/nvidia0"])


@pytest.mark.parametrize("path", ["/dev", "/dev/mem", "/dev/nvidia-caps", "/dev/nvidia0/../mem"])
def test_no_other_devices_or_directories_can_be_mounted(path):
    with pytest.raises(ValueError):
        add_gpu_mounts(["--dev", "/dev", "--", "/bin/true"], [path])
    with pytest.raises(ValueError):
        device_record(path)


def test_host_allocation_and_both_visibility_lists_are_intersected():
    devices = [{"index": "0", "uuid": "GPU-aaa", "device": "/dev/nvidia3"},
               {"index": "1", "uuid": "GPU-bbb", "device": "/dev/nvidia6"}]
    assert select_devices(devices, {"CUDA_VISIBLE_DEVICES": "1"}) == [devices[1]]
    assert select_devices(devices, {"CUDA_VISIBLE_DEVICES": "1", "NVIDIA_VISIBLE_DEVICES": "GPU-aa"}) == []
    assert select_devices(devices, {"CUDA_VISIBLE_DEVICES": ""}) == []
    with pytest.raises(ValueError):
        select_devices(devices, {"CUDA_VISIBLE_DEVICES": "6"})


def config_file(tmp_path):
    codex = tmp_path / "release/bin/codex"
    codex.parent.mkdir(parents=True)
    codex.write_text("#!/bin/sh\nexit 0\n")
    codex.chmod(0o700)
    config = {"version": 1, "hostname": socket.gethostname(), "codex": str(codex),
              "bwrap": str(codex.parent.parent / "codex-resources/bwrap"),
              "devices": [{"path": p, "major": 195, "minor": i} for i, p in enumerate(
                  ("/dev/nvidia3", "/dev/nvidiactl", "/dev/nvidia-uvm"))]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    path.chmod(0o600)
    return codex, path


def test_environment_is_local_pinned_and_config_cannot_be_task_writable(tmp_path, monkeypatch):
    codex, path = config_file(tmp_path)
    device_map = {d["path"]: d for d in json.loads(path.read_text())["devices"]}
    monkeypatch.setattr("api_server.gpu_sandbox.device_record", device_map.__getitem__)
    env = {"PATH": "/usr/bin", "KEEP": "yes"}
    result = prepare_environment(str(codex), tmp_path / "task", path, env)
    assert result["PATH"] == str(WRAPPER_DIR) + os.pathsep + "/usr/bin"
    assert result[CONFIG_ENV] == str(path)
    assert env == {"PATH": "/usr/bin", "KEEP": "yes"}
    with pytest.raises(ValueError, match="writable task workspace"):
        prepare_environment(str(codex), tmp_path, path, env)
    with pytest.raises(ValueError, match="installation changed"):
        prepare_environment("/usr/bin/true", tmp_path / "task", path, env)


def test_mutable_or_replaced_device_configuration_is_rejected(tmp_path, monkeypatch):
    _, path = config_file(tmp_path)
    path.chmod(0o622)
    with pytest.raises(ValueError, match="not writable by others"):
        load_config(path)
    path.chmod(0o600)
    monkeypatch.setattr("api_server.gpu_sandbox.device_record", lambda p: {"path": p, "major": 0, "minor": 0})
    with pytest.raises(ValueError, match="identity changed"):
        load_config(path, check_devices=True)


def test_network_filter_cannot_be_omitted_or_use_shared_network():
    args = ["--dev", "/dev", "--seccomp", "8", "--", "command", "--"]
    actual = add_network_filter(args, 9)
    assert actual == args[:4] + ["--unshare-net", "--add-seccomp-fd", "9"] + args[4:]
    with pytest.raises(ValueError, match="share the host network"):
        add_network_filter(["--share-net", *args], 9)


def evaluate_filter(number, args=(), arch=0xC000003E):
    # Interpret the Linux classic-BPF subset against arbitrary seccomp_data.
    data = struct.pack("=IIQ6Q", number, arch, 0, *(list(args) + [0] * (6 - len(args))))
    program = list(struct.iter_unpack("=HBBI", network_filter_bytes()))
    pc = accumulator = 0
    for _ in range(200):
        code, yes, no, value = program[pc]
        pc += 1
        if code == 0x20:
            accumulator, = struct.unpack_from("=I", data, value)
        elif code == 0x15:
            pc += yes if accumulator == value else no
        elif code == 0x35:
            pc += yes if accumulator >= value else no
        elif code == 0x54:
            accumulator &= value
        elif code == 0x06:
            return value
        else:
            raise AssertionError(f"Unexpected BPF opcode {code}")
    raise AssertionError("BPF did not terminate")


@pytest.mark.parametrize("number,args", [
    (41, (2, 1, 0)), (41, (10, 1, 0)), (41, (16, 3, 0)),
    (41, (1, 2, 0)), (53, (1, 2, 0)),
    (42, ()), (44, (0, 0, 0, 0, 123, 0)), (44, (0, 0, 0, 0, 0x100000000, 0)),
    (307, ()), (425, ()), (426, ()), (427, ()), (438, ()),
])
def test_network_filter_denies_ip_and_host_ipc_paths(number, args):
    assert evaluate_filter(number, args) == 0x00050001


def test_network_filter_allows_cuda_local_ipc_and_rejects_other_abis():
    for number in (41, 53):
        for kind in (1, 5, 5 | socket.SOCK_CLOEXEC | socket.SOCK_NONBLOCK):
            assert evaluate_filter(number, (1, kind, 0)) == 0x7FFF0000
    for number in (0, 1, 44, 46, 49, 50, 54):  # IO, connected sends, bind/listen, setsockopt
        assert evaluate_filter(number) == 0x7FFF0000
    assert evaluate_filter(0, arch=0x40000003) == 0x80000000
    assert evaluate_filter(0x40000000 | 42) == 0x00050001


@pytest.mark.parametrize("enabled", [False, True])
def test_api_only_enables_local_ipc_with_gpu_wrapper_installed(tmp_path, monkeypatch, enabled):
    from fastapi.testclient import TestClient
    from api_server.app import Settings, create_app
    codex, config = config_file(tmp_path)
    device_map = {d["path"]: d for d in json.loads(config.read_text())["devices"]}
    monkeypatch.setattr("api_server.gpu_sandbox.device_record", device_map.__getitem__)
    codex.write_text(f"#!{sys.executable}\n" + r'''
import json, os, pathlib, sys, uuid
args = sys.argv[1:]
sys.stdin.read()
print(json.dumps({"type":"thread.started", "thread_id":str(uuid.uuid4())}), flush=True)
answer = {"args":args, "path_first":os.environ["PATH"].split(os.pathsep)[0],
          "gpu_config":os.environ.get("CODEX_GPU_SANDBOX_CONFIG"),
          "token_present":"API_SERVER_TOKEN" in os.environ}
pathlib.Path(args[args.index("--output-last-message")+1]).write_text(json.dumps(answer))
print(json.dumps({"type":"turn.completed", "usage":{}}), flush=True)
''')
    workspace = tmp_path / "task"
    workspace.mkdir()
    monkeypatch.setenv("API_SERVER_TOKEN", "must-not-reach-codex")
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    settings = Settings(runtime=tmp_path / "runtime", codex_bin=str(codex), token="",
                        write_roots=[tmp_path], gpu_sandbox_config=config if enabled else None)
    with TestClient(create_app(settings)) as client:
        result = client.post("/v1/chat", json={"prompt":"test", "workspace_dir":str(workspace)})
    assert result.status_code == 200, result.text
    answer = json.loads(result.json()["answer"])
    assert 'sandbox_mode="workspace-write"' in answer["args"]
    assert "sandbox_workspace_write.exclude_slash_tmp=true" in answer["args"]
    assert not answer["token_present"]
    if enabled:
        assert answer["path_first"] == str(WRAPPER_DIR)
        assert answer["gpu_config"] == str(config)
        assert "sandbox_workspace_write.network_access=true" in answer["args"]
        assert "features.network_proxy=false" in answer["args"]
        assert "features.use_legacy_landlock=false" in answer["args"]
    else:
        assert answer["gpu_config"] is None
        assert "sandbox_workspace_write.network_access=false" in answer["args"]
