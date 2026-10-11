"""Codex CLI transport. No shell interpolation; prompts travel through stdin."""
import asyncio
import base64
import binascii
import io
import inspect
import json
import logging
import os
import signal
import shutil
import time
import warnings
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .schemas import ImageInput
from .gpu_sandbox import prepare_environment as prepare_gpu_environment

# The queue owns this per-job state; it is never written into model prompts.
job_activity = ContextVar("codex_job_activity", default=None)
logger = logging.getLogger(__name__)
EVENT_LINE_LIMIT = 4 * 1024 * 1024
EVENT_READ_SIZE = 64 * 1024


async def _event_lines(stream, activity=None, *, limit=EVENT_LINE_LIMIT):
    """Drain JSONL with bounded memory, dropping oversized diagnostic events.

    Thread/turn control records are small; command output events can be huge and
    are not needed for the result. Always drain through their newline so the next
    control record remains readable. An absent completion record still fails the
    job in CodexBackend.run.
    """
    line = bytearray()
    line_size = 0
    while chunk := await stream.read(EVENT_READ_SIZE):
        if activity is not None:
            activity["last_activity_at"] = time.monotonic()
        start = 0
        while start < len(chunk):
            newline = chunk.find(b"\n", start)
            end = newline if newline >= 0 else len(chunk)
            line_size += end - start
            if line_size <= limit:
                line.extend(chunk[start:end])
            else:
                line.clear()
            if newline >= 0:
                if line_size <= limit:
                    yield line
                else:
                    logger.warning("Skipped oversized Codex event: %d bytes (limit %d)", line_size, limit)
                line = bytearray()
                line_size = 0
            start = end + 1
    if line_size:
        if line_size <= limit:
            yield line
        else:
            logger.warning("Skipped oversized Codex event at EOF: %d bytes (limit %d)", line_size, limit)


class BackendError(Exception):
    pass


@dataclass
class CodexResult:
    thread_id: str
    answer: str
    usage: dict


def prepare_images(images: list[ImageInput], directory: Path, roots: list[Path]) -> list[Path]:
    paths = []
    total = 0
    for index, source in enumerate(images):
        if source.path:
            path = Path(source.path).expanduser().resolve()
            if not any(path.is_relative_to(root) for root in roots):
                raise ValueError("Image path is outside CODEX_IMAGE_ROOTS")
            try:
                with path.open("rb") as file:
                    data = file.read(10 * 1024 * 1024 + 1)
            except OSError as exc:
                raise ValueError("Image file is not readable") from exc
        else:
            encoded = source.base64 or ""
            if encoded.startswith("data:"):
                header, separator, encoded = encoded.partition(",")
                if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
                    raise ValueError("Invalid image data URL")
            if len(encoded) > 14 * 1024 * 1024:
                raise ValueError("Image exceeds 10 MiB")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Invalid image base64") from exc
        total += len(data)
        if len(data) > 10 * 1024 * 1024 or total > 20 * 1024 * 1024:
            raise ValueError("Image limits: 10 MiB each, 20 MiB total")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as img:
                    suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(img.format)
                    if suffix is None or getattr(img, "is_animated", False):
                        raise ValueError("Use a non-animated PNG, JPEG, or WebP image")
                    img.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise ValueError("Invalid or oversized image") from exc
        target = directory / f"image-{index}{suffix}"
        target.write_bytes(data)
        paths.append(target)
    return paths


def conversation_session_home(runtime_root, conversation_id):
    """Return the persistent home shared by execution and explicit cleanup."""
    return Path(runtime_root) / "codex-sessions" / str(conversation_id)


class CodexBackend:
    def __init__(self, executable: str, workdir: Path, gpu_sandbox_config: Path | None = None):
        self.executable = executable
        self.workdir = workdir
        self.gpu_sandbox_config = gpu_sandbox_config

    async def run(self, *, prompt, images, model, reasoning_effort, thread_id, output_dir,
                  on_thread, workspace_dir=None, conversation_id=None):
        output_file = output_dir / "answer.txt"
        command = [self.executable, "exec"]
        if workspace_dir is not None:
            command += ["--cd", str(workspace_dir)]
        if thread_id:
            command += ["resume"]
        command += ["--json", "--skip-git-repo-check", "-c", 'approval_policy="never"',
                    "-c", 'sandbox_mode="workspace-write"' if workspace_dir else 'sandbox_mode="read-only"',
                    "--output-last-message", str(output_file)]
        if workspace_dir is not None:
            command += ["-c", "sandbox_workspace_write.writable_roots=[]",
                        "-c", "sandbox_workspace_write.exclude_slash_tmp=true",
                        "-c", "sandbox_workspace_write.exclude_tmpdir_env_var=true",
                        "-c", "sandbox_workspace_write.network_access=" +
                        ("true" if self.gpu_sandbox_config is not None else "false")]
            if self.gpu_sandbox_config is not None:
                # CUDA needs local Unix bind/listen. The GPU bwrap wrapper adds
                # a private network namespace and stricter outbound syscall
                # filter before Codex runs, retaining network isolation.
                command += ["-c", "features.network_proxy=false",
                            "-c", "features.use_legacy_landlock=false"]
        if model:
            command += ["--model", model]
        if reasoning_effort:
            command += ["-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}"]
        for path in images:
            command += ["--image", str(path)]
        # The separator prevents variadic --image from consuming a session ID.
        command += ["--"] + ([thread_id] if thread_id else []) + ["-"]
        environment = os.environ.copy()
        # Every request gets an isolated CODEX_HOME. Only the source auth file is
        # copied into it; model caches and unrelated sessions never persist.
        runtime_root = self.workdir.parent
        session_key = conversation_id or thread_id or "anonymous"
        persistent_session = conversation_session_home(runtime_root, session_key)
        persistent_sessions = persistent_session / "sessions"
        persistent_sessions.mkdir(parents=True, exist_ok=True, mode=0o700)
        request_home = runtime_root / f"codex-home-{os.getpid()}-{time.time_ns()}"
        request_home.mkdir(parents=True, exist_ok=False, mode=0o700)
        request_auth = request_home / "auth.json"
        source_auth = Path(os.environ.get("CODEX_AUTH_FILE") or str(Path.home() / ".codex" / "auth.json")).expanduser()
        try:
            if not source_auth.is_file():
                raise BackendError(f"Codex auth file is unavailable: {source_auth}")
            shutil.copyfile(source_auth, request_auth)
            os.chmod(request_auth, 0o600)
            request_sessions = request_home / "sessions"
            if persistent_sessions.is_dir():
                shutil.copytree(persistent_sessions, request_sessions, dirs_exist_ok=True)
            environment["CODEX_HOME"] = str(request_home)
            # Do not pass the HTTP authentication secret to Codex or its child tools.
            environment.pop("API_SERVER_TOKEN", None)
            if workspace_dir is not None and self.gpu_sandbox_config is not None:
                try:
                    environment = prepare_gpu_environment(
                        self.executable, workspace_dir, self.gpu_sandbox_config, environment)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise BackendError(f"GPU sandbox configuration error: {exc}") from exc
            try:
                process = await asyncio.create_subprocess_exec(
                    *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, cwd=workspace_dir or self.workdir, env=environment,
                    start_new_session=True, limit=EVENT_READ_SIZE,
                )
            except OSError as exc:
                raise BackendError(f"Cannot start Codex: {exc}") from exc
            activity = job_activity.get()
            if activity is not None:
                activity.update(process=process, started_at=time.monotonic(),
                                last_activity_at=time.monotonic(), last_event="process.started", event_count=0)
            found_thread = thread_id
            usage = {}
            completed = False
            failure = None
            stderr_tail = bytearray()

            async def read_events():
                nonlocal found_thread, usage, completed, failure
                async for line in _event_lines(process.stdout, activity):
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if not isinstance(event, dict):
                        continue
                    if activity is not None:
                        activity["event_count"] += 1
                        activity["last_event"] = event.get("type")
                    if event.get("type") == "thread.started":
                        found_thread = event["thread_id"]
                        update = on_thread(found_thread)
                        if inspect.isawaitable(update):
                            await update
                    elif event.get("type") == "turn.completed":
                        completed = True
                        usage = event.get("usage") or {}
                    elif event.get("type") == "turn.failed":
                        failure = str(event.get("error", "Codex turn failed"))

            async def read_stderr():
                while chunk := await process.stderr.read(8192):
                    if activity is not None:
                        activity["last_activity_at"] = time.monotonic()
                    stderr_tail.extend(chunk)
                    del stderr_tail[:-8192]

            async def write_prompt():
                try:
                    process.stdin.write(prompt.encode("utf-8"))
                    await process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    process.stdin.close()

            async def monitor_process():
                # A descendant may keep stdout/stderr open after Codex exits. Check
                # the actual child status so that pipe readers cannot mask its death.
                while process.returncode is None:
                    await asyncio.sleep(0.25)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            tasks = [asyncio.create_task(fn()) for fn in (read_events, read_stderr, write_prompt, monitor_process)]
            try:
                await asyncio.gather(*tasks)
                await process.wait()
            finally:
                # Also clean up tool descendants when a timeout/cancellation interrupts a turn.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            if process.returncode != 0 or failure or not completed or not found_thread:
                detail = failure or stderr_tail.decode("utf-8", errors="replace")[-2000:]
                raise BackendError(f"Codex did not complete successfully (exit {process.returncode}): {detail}")
            if not output_file.exists() or output_file.stat().st_size > 8 * 1024 * 1024:
                raise BackendError("Codex final answer is missing or exceeds 8 MiB")
            answer = output_file.read_text(encoding="utf-8").strip()
            if not answer:
                raise BackendError("Codex returned an empty final answer")
            # Keep only rollout files belonging to this conversation for resume.
            request_sessions = request_home / "sessions"
            if request_sessions.is_dir():
                for candidate in request_sessions.rglob("*"):
                    if not candidate.is_file():
                        continue
                    try:
                        matches = bool(found_thread and found_thread in candidate.name)
                        if not matches and candidate.stat().st_size <= 16 * 1024 * 1024:
                            matches = found_thread in candidate.read_text(encoding="utf-8", errors="ignore")
                    except OSError:
                        matches = False
                    if matches:
                        target = persistent_sessions / candidate.relative_to(request_sessions)
                        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        shutil.copyfile(candidate, target)
            result = CodexResult(found_thread, answer, usage)
            return result
        finally:
            shutil.rmtree(request_home, ignore_errors=True)
