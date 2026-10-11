#!/usr/bin/env python3
"""Launch the local API package or query its health without exposing secrets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


PIPELINE_DIR = Path(__file__).resolve().parent.parent


def server_environment(package_dir: Path, incoming: dict[str, str]) -> dict[str, str]:
    """Explicit caller settings override saved migration settings and defaults."""
    environment = incoming.copy()
    config_path = package_dir / "launch_config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as config_file:
            settings = json.load(config_file)
        if not isinstance(settings, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            or not key or "=" in key or "\0" in key or "\0" in value
            for key, value in settings.items()
        ):
            raise ValueError("launch_config.json must contain environment names and string values")
        for key, value in settings.items():
            environment.setdefault(key, value)

    defaults = {
        "API_SERVER_HOST": "127.0.0.1",
        "API_SERVER_PORT": "8000",
        "API_SERVER_DATA_DIR": str(package_dir / "runtime"),
        "CODEX_MAX_CONCURRENCY": "20",
        "CODEX_IMAGE_ROOTS": str(PIPELINE_DIR),
        "CODEX_WRITE_ROOTS": os.pathsep.join(
            (str(PIPELINE_DIR), str(PIPELINE_DIR / "output"))
        ),
    }
    for key, value in defaults.items():
        environment.setdefault(key, value)
    environment.setdefault("TMPDIR", str(Path(environment["API_SERVER_DATA_DIR"]) / "tmp"))
    return environment


def launch_command(environment: dict[str, str], package_dir: Path) -> list[str]:
    if environment["API_SERVER_HOST"] not in ("127.0.0.1", "::1", "localhost") \
            and not environment.get("API_SERVER_TOKEN"):
        raise ValueError("Set API_SERVER_TOKEN before listening on a non-loopback address.")
    port = int(environment["API_SERVER_PORT"])
    if not 1 <= port <= 65535:
        raise ValueError("API_SERVER_PORT must be between 1 and 65535")
    return [
        sys.executable, "-m", "uvicorn", "api_server.app:create_app", "--factory",
        "--host", environment["API_SERVER_HOST"], "--port", str(port), "--workers", "1",
        "--app-dir", str(package_dir.parent),
    ]


def status(environment: dict[str, str], arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Query API health and queue (read-only).")
    parser.add_argument("--base-url", help="Override the configured server URL")
    parser.add_argument("--timeout", type=float, default=10.0)
    options = parser.parse_args(arguments)
    if options.timeout <= 0:
        parser.error("--timeout must be positive")
    host = environment["API_SERVER_HOST"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    base_url = (options.base_url or f"http://{host}:{environment['API_SERVER_PORT']}").rstrip("/")
    # These are direct requests to the API; never route local traffic through a proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    failed = False
    for route in ("/health", "/v1/queue"):
        headers = {}
        if route != "/health" and environment.get("API_SERVER_TOKEN"):
            headers["Authorization"] = "Bearer " + environment["API_SERVER_TOKEN"]
        request = urllib.request.Request(base_url + route, headers=headers)
        try:
            with opener.open(request, timeout=options.timeout) as response:
                payload = json.load(response)
            print(route)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        except (OSError, ValueError, urllib.error.URLError) as error:
            print(f"{route}: {error}", file=sys.stderr, flush=True)
            failed = True
    return int(failed)


def main() -> int:
    package_dir = Path(__file__).resolve().parent
    environment = server_environment(package_dir, dict(os.environ))
    if sys.argv[1:2] == ["--status"]:
        return status(environment, sys.argv[2:])
    if sys.argv[1:]:
        raise ValueError("start.sh takes no arguments; configure the server with environment variables")
    command = launch_command(environment, package_dir)
    os.umask(0o077)
    Path(environment["API_SERVER_DATA_DIR"]).mkdir(parents=True, exist_ok=True, mode=0o700)
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True, mode=0o700)
    # Import api_server from its physical parent, independently of either symlink entry.
    os.chdir(package_dir.parent)
    os.execve(sys.executable, command, environment)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"API launcher: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
