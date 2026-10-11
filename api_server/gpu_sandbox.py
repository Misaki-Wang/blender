"""GPU-only device passthrough for this API's Codex bubblewrap subprocesses.

Codex still constructs and enforces its filesystem policy. The wrapper adds
validated NVIDIA devices, a private network namespace, and a network filter
which allows CUDA's local Unix IPC while blocking outbound connections.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import platform
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
WRAPPER_DIR = ROOT / "gpu_sandbox_bin"
CONFIG_ENV = "CODEX_GPU_SANDBOX_CONFIG"
DEVICE_PATH = re.compile(r"/dev/nvidia(?:[0-9]+|ctl|-uvm)\Z")

# Fail closed if a future Codex version changes the bubblewrap argument format.
OPTION_ARITY = {
    **dict.fromkeys(("--as-pid-1", "--new-session", "--die-with-parent",
                    "--unshare-all", "--share-net", "--unshare-user",
                    "--unshare-user-try", "--unshare-ipc", "--unshare-pid",
                    "--unshare-net", "--unshare-uts", "--unshare-cgroup",
                    "--unshare-cgroup-try", "--disable-userns",
                    "--assert-userns-disabled", "--clearenv", "--level-prefix"), 0),
    **dict.fromkeys(("--argv0", "--userns", "--userns2", "--pidns", "--uid",
                    "--gid", "--hostname", "--chdir", "--unsetenv", "--lock-file",
                    "--sync-fd", "--remount-ro", "--overlay-src", "--tmp-overlay",
                    "--ro-overlay", "--exec-label", "--file-label", "--proc",
                    "--dev", "--tmpfs", "--mqueue", "--dir", "--seccomp",
                    "--add-seccomp-fd", "--block-fd", "--userns-block-fd",
                    "--info-fd", "--json-status-fd", "--cap-add", "--cap-drop",
                    "--perms", "--size"), 1),
    **dict.fromkeys(("--setenv", "--bind", "--bind-try", "--dev-bind",
                    "--dev-bind-try", "--ro-bind", "--ro-bind-try", "--bind-fd",
                    "--ro-bind-fd", "--file", "--bind-data", "--ro-bind-data",
                    "--symlink", "--chmod"), 2),
    "--overlay": 3,
}


def select_devices(devices, environment):
    """Intersect the host inventory with both inherited GPU visibility lists."""
    allowed = list(devices)
    for key in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        value = environment.get(key)
        if value is None or value.strip().lower() == "all":
            continue
        if value.strip().lower() in ("", "-1", "none", "void", "nodevfiles"):
            allowed = []
            continue
        selected = set()
        for token in value.split(","):
            token = token.strip()
            matches = [d for d in devices if token == d["index"] or
                       (token.startswith("GPU-") and d["uuid"].startswith(token))]
            if len(matches) != 1:
                raise ValueError(f"{key} contains an unknown or ambiguous GPU: {token!r}")
            selected.add(matches[0]["uuid"])
        allowed = [d for d in allowed if d["uuid"] in selected]
    return allowed


def device_record(path):
    if not isinstance(path, str) or not DEVICE_PATH.fullmatch(path):
        raise ValueError(f"Not an allowed NVIDIA device path: {path!r}")
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISCHR(info.st_mode):
        raise ValueError(f"GPU device must be a character device, not a symlink or file: {path}")
    return {"path": path, "major": os.major(info.st_rdev), "minor": os.minor(info.st_rdev)}


def configure(output, codex="codex"):
    executable = shutil.which(codex)
    if not executable:
        raise ValueError(f"Codex executable was not found: {codex}")
    executable = Path(executable).resolve()
    # Pin this installation's real helper; never discover our own wrapper via PATH.
    bwrap = executable.parent.parent / "codex-resources/bwrap"
    if not bwrap.is_file() or not os.access(bwrap, os.X_OK):
        raise ValueError("This configurator requires a standalone Codex installation with bundled bwrap")
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True,
                            text=True, check=True, timeout=15)
    devices = []
    for index, gpu in enumerate(ET.fromstring(result.stdout).findall("gpu")):
        minor = gpu.findtext("minor_number", "")
        uuid = gpu.findtext("uuid", "")
        if not minor.isdigit() or not uuid.startswith("GPU-"):
            raise ValueError("nvidia-smi returned an unsupported GPU identity")
        devices.append({"index": str(index), "uuid": uuid, "device": f"/dev/nvidia{minor}"})
    devices = select_devices(devices, os.environ)
    if not devices:
        raise ValueError("No allocated NVIDIA GPUs are visible; run configuration on the GPU host")
    paths = [d["device"] for d in devices] + ["/dev/nvidiactl", "/dev/nvidia-uvm"]
    config = {"version": 1, "hostname": socket.gethostname(), "codex": str(executable),
              "bwrap": str(bwrap), "gpus": devices,
              "devices": [device_record(p) for p in dict.fromkeys(paths)]}
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as handle:
        temp = Path(handle.name)
        json.dump(config, handle, indent=2)
        handle.write("\n")
    try:
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    return config


def load_config(path, *, check_devices=False):
    path = Path(path)
    metadata = path.stat()
    if metadata.st_uid != os.getuid() or metadata.st_mode & 0o022:
        raise ValueError("GPU sandbox config must be owned by this user and not writable by others")
    config = json.loads(path.read_text())
    if config.get("version") != 1 or config.get("hostname") != socket.gethostname():
        raise ValueError("GPU sandbox configuration version/host differs; configure it again")
    codex = Path(config["codex"])
    bwrap = Path(config["bwrap"])
    if not codex.is_absolute() or bwrap != codex.parent.parent / "codex-resources/bwrap":
        raise ValueError("GPU sandbox must use the pinned Codex installation's bundled bwrap")
    paths = []
    for entry in config["devices"]:
        p = entry["path"]
        if not isinstance(p, str) or not DEVICE_PATH.fullmatch(p):
            raise ValueError(f"Not an allowed NVIDIA device path: {p!r}")
        if check_devices and device_record(p) != entry:
            raise ValueError(f"GPU device identity changed; configure it again: {p}")
        paths.append(p)
    if len(paths) != len(set(paths)) or not any(re.fullmatch(r"/dev/nvidia[0-9]+", p) for p in paths):
        raise ValueError("GPU sandbox needs a unique, nonempty GPU device allowlist")
    if not {"/dev/nvidiactl", "/dev/nvidia-uvm"}.issubset(paths):
        raise ValueError("GPU sandbox requires the NVIDIA control and UVM devices")
    return config


def prepare_environment(executable, workspace, config_path, environment):
    config_path = Path(config_path).resolve()
    workspace = Path(workspace).resolve()
    for protected in (ROOT / "gpu_sandbox.py", WRAPPER_DIR, config_path):
        if protected.resolve().is_relative_to(workspace):
            raise ValueError("GPU sandbox code/config cannot be inside the writable task workspace")
    config = load_config(config_path, check_devices=True)
    resolved = shutil.which(executable, path=environment.get("PATH"))
    if not resolved or Path(resolved).resolve() != Path(config["codex"]):
        raise ValueError("Codex installation changed; reconfigure and verify the GPU sandbox first")
    if not os.access(WRAPPER_DIR / "bwrap", os.X_OK):
        raise ValueError("GPU bubblewrap wrapper is missing or not executable")
    result = environment.copy()
    result[CONFIG_ENV] = str(config_path)
    result["PATH"] = str(WRAPPER_DIR) + os.pathsep + result.get("PATH", "")
    return result


def sandbox_layout(args):
    insertion = None
    i = 0
    while i < len(args) and args[i] != "--":
        option = args[i]
        if option not in OPTION_ARITY:
            raise ValueError(f"Unsupported bubblewrap option {option!r}; review the sandbox wrapper")
        arity = OPTION_ARITY[option]
        if i + arity >= len(args):
            raise ValueError(f"Incomplete bubblewrap option: {option}")
        if option == "--dev" and args[i + 1] == "/dev":
            if insertion is not None:
                raise ValueError("Multiple private /dev mounts are unsupported")
            insertion = i + 2
        i += arity + 1
    if i >= len(args) - 1 or args[i] != "--" or insertion is None:
        raise ValueError("Expected a Codex private /dev mount and explicit command separator")
    return insertion, i


def add_gpu_mounts(arguments, devices):
    args = list(arguments)
    if args in (["--help"], ["--version"]):
        return args
    insertion, _ = sandbox_layout(args)
    extra = []
    for device in devices:
        if not isinstance(device, str) or not DEVICE_PATH.fullmatch(device):
            raise ValueError(f"Not an allowed NVIDIA device path: {device!r}")
        extra.extend(("--dev-bind", device, device))
    args[insertion:insertion] = extra
    return args


def network_filter_bytes():
    """Allow CUDA's local Unix IPC, with no outbound network or host-socket access.

    x86_64 syscall numbers and seccomp_data offsets come from the Linux UAPI.
    Only AF_UNIX STREAM/SEQPACKET sockets can be created. connect/addressed sendto are
    denied, so filesystem Unix sockets cannot reach host daemons. Datagram
    sockets are denied to prevent sendmsg(msg_name=...) reaching host sockets.
    Connected anonymous socketpairs retain sendmsg for normal local FD IPC.
    io_uring is denied so it cannot perform filtered socket operations indirectly.
    """
    if platform.machine() != "x86_64":
        raise ValueError("GPU sandbox network filter currently supports x86_64 Linux only")
    deny = 0x00050000 | errno.EPERM
    allow = 0x7FFF0000
    instructions = [
        (0x20, 0, 0, 4),                 # load seccomp_data.arch
        (0x15, 1, 0, 0xC000003E),        # AUDIT_ARCH_X86_64
        (0x06, 0, 0, 0x80000000),        # kill other ABIs
        (0x20, 0, 0, 0),                 # load syscall number
        (0x35, 0, 1, 0x40000000),        # reject x32 ABI
        (0x06, 0, 0, deny),
    ]
    # send()/asyncio socketpair wakeups use sendto with a NULL destination.
    # This is equivalent to write on an already-connected local descriptor.
    instructions += [
        (0x15, 0, 6, 44),
        (0x20, 0, 0, 48),                # args[4]: destination pointer, low word
        (0x15, 0, 2, 0),
        (0x20, 0, 0, 52),                # destination pointer, high word
        (0x15, 1, 0, 0),
        (0x06, 0, 0, deny),
        (0x06, 0, 0, allow),
    ]
    # connect, sendmmsg, io_uring_{setup,enter,register}, pidfd_getfd
    for number in (42, 307, 425, 426, 427, 438):
        instructions += [(0x15, 0, 1, number), (0x06, 0, 0, deny)]
    instructions += [
        (0x15, 1, 0, 41),                # socket
        (0x15, 0, 8, 53),                # socketpair; otherwise allow
        (0x20, 0, 0, 16),                # args[0]: family
        (0x15, 1, 0, 1),                 # AF_UNIX
        (0x06, 0, 0, deny),
        (0x20, 0, 0, 24),                # args[1]: socket type
        (0x54, 0, 0, 0xF),               # ignore CLOEXEC/NONBLOCK
        (0x15, 2, 0, 1),                 # SOCK_STREAM
        (0x15, 1, 0, 5),                 # SOCK_SEQPACKET
        (0x06, 0, 0, deny),
        (0x06, 0, 0, allow),
    ]
    return b"".join(struct.pack("=HBBI", *row) for row in instructions)


def add_network_filter(arguments, fd):
    args = list(arguments)
    _, separator = sandbox_layout(args)
    if "--share-net" in args[:separator]:
        raise ValueError("GPU sandbox cannot share the host network namespace")
    extra = [] if "--unshare-net" in args[:separator] else ["--unshare-net"]
    extra += ["--add-seccomp-fd", str(fd)]
    args[separator:separator] = extra
    return args


def run_bwrap(arguments):
    try:
        config = load_config(os.environ[CONFIG_ENV], check_devices=True)
        args = add_gpu_mounts(arguments, [d["path"] for d in config["devices"]])
        if args not in (["--help"], ["--version"]):
            fd = os.memfd_create("codex-gpu-network", os.MFD_ALLOW_SEALING)
            os.write(fd, network_filter_bytes())
            os.lseek(fd, 0, os.SEEK_SET)
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
            os.set_inheritable(fd, True)
            args = add_network_filter(args, fd)
        # exec preserves Codex's inherited mount/seccomp/proxy descriptors.
        os.execv(config["bwrap"], [config["bwrap"], *args])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"GPU sandbox: {exc}", file=sys.stderr)
        raise SystemExit(126)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configure", type=Path, required=True, metavar="CONFIG_JSON")
    parser.add_argument("--codex", default="codex")
    args = parser.parse_args()
    config = configure(args.configure, args.codex)
    print(json.dumps({"config": str(args.configure.resolve()), **config}, indent=2))


if __name__ == "__main__":
    main()
