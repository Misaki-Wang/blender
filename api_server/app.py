import asyncio
import fcntl
import hmac
import json
import logging
import math
import os
import shutil
import tempfile
import time
import tomllib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from starlette.responses import JSONResponse

from .backend import BackendError, CodexBackend, prepare_images, job_activity, conversation_session_home
from .history import Store
from .schemas import BatchRequest, ChatRequest, ChatResponse, JobBatchRequest, JobQuery
from .scheduler import JobQueue, TERMINAL
from .session_cleanup import delete_session_files

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("api_server")


@dataclass
class Settings:
    runtime: Path = field(default_factory=lambda: Path(os.getenv("API_SERVER_DATA_DIR", ROOT / "runtime")).resolve())
    codex_bin: str = field(default_factory=lambda: os.getenv("CODEX_BIN", "codex"))
    gpu_sandbox_config: Path | None = field(default_factory=lambda:
        Path(os.environ["CODEX_GPU_SANDBOX_CONFIG"]).expanduser().resolve()
        if os.getenv("CODEX_GPU_SANDBOX_CONFIG") else None)
    codex_home: Path = field(default_factory=lambda: Path(os.getenv("CODEX_HOME", Path.home() / ".codex")).resolve())
    default_model: str | None = field(default_factory=lambda: os.getenv("CODEX_MODEL") or None)
    token: str = field(default_factory=lambda: os.getenv("API_SERVER_TOKEN", ""))
    max_concurrency: int = field(default_factory=lambda: int(os.getenv("CODEX_MAX_CONCURRENCY", "20")))
    max_pending: int = field(default_factory=lambda: int(os.getenv("CODEX_MAX_PENDING", "10000")))
    max_pending_bytes: int = field(default_factory=lambda: int(os.getenv("CODEX_MAX_PENDING_BYTES", str(512 * 1024 * 1024))))
    timeout: float | None = field(default_factory=lambda: float(os.getenv("CODEX_TIMEOUT_SECONDS", "0")) or None)
    image_roots: list[Path] = field(default_factory=lambda: [Path(p).expanduser().resolve() for p in
        os.getenv("CODEX_IMAGE_ROOTS", str(ROOT.parent)).split(os.pathsep) if p])
    write_roots: list[Path] = field(default_factory=lambda: [Path(p).expanduser().resolve() for p in
        os.getenv("CODEX_WRITE_ROOTS", str(ROOT.parent)).split(os.pathsep) if p])
    max_body_bytes: int = 32 * 1024 * 1024

    def __post_init__(self):
        if isinstance(self.max_concurrency, bool) or not isinstance(self.max_concurrency, int) or not 1 <= self.max_concurrency <= 20:
            raise ValueError("CODEX_MAX_CONCURRENCY must be an integer between 1 and 20")
        if self.timeout == 0:
            self.timeout = None
        if self.max_concurrency < 0 or (self.timeout is not None and
                (not math.isfinite(self.timeout) or self.timeout < 0)) or self.max_pending < 1 or self.max_pending_bytes < 1:
            raise ValueError("Concurrency, timeout and queue limits must be non-negative")
        if not self.default_model:
            try:
                self.default_model = tomllib.loads((self.codex_home / "config.toml").read_text()).get("model")
            except (OSError, ValueError):
                pass



class BodyLimit:
    """Bound JSON bodies even when transfer encoding is chunked."""
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.limit:
                return await JSONResponse({"detail": "Request body exceeds 32 MiB"}, 413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None, backend=None):
    settings = settings or Settings()
    session_locks = {}
    semaphore = asyncio.Semaphore(settings.max_concurrency) if settings.max_concurrency else None

    @asynccontextmanager
    async def concurrency_slot():
        if semaphore is None:
            yield
        else:
            async with semaphore:
                yield

    @asynccontextmanager
    async def lifespan(app):
        settings.runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_file = (settings.runtime / "server.lock").open("a")
        try:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("This data directory is already in use; run one server worker") from exc
            app.state.store = await asyncio.to_thread(Store, settings.runtime / "conversations.sqlite3")
            workdir = settings.runtime / "workspace"
            workdir.mkdir(exist_ok=True)
            app.state.backend = backend or CodexBackend(settings.codex_bin, workdir, settings.gpu_sandbox_config)
            app.state.queue = JobQueue(settings, execute_chat, validate_request)
            await app.state.queue.start()
            try:
                yield
            finally:
                await app.state.queue.close()
                await app.state.store.aclose()
        finally:
            lock_file.close()

    app = FastAPI(title="Codex API Server", version="2.3.0", lifespan=lifespan)
    app.add_middleware(BodyLimit, limit=settings.max_body_bytes)

    async def authorize(authorization: str | None = Header(default=None)):
        if settings.token and not hmac.compare_digest((authorization or "").encode(),
                                                      f"Bearer {settings.token}".encode()):
            raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health")
    async def health():
        return {"status": "ok", "codex_available": shutil.which(settings.codex_bin) is not None,
                "max_concurrency": settings.max_concurrency, "workspace_write": bool(settings.write_roots)}

    @app.get("/v1/capabilities", dependencies=[Depends(authorize)])
    async def capabilities():
        return {"reserve_conversation": True, "workspace_write_roots": [str(p) for p in settings.write_roots],
                "execution_timeout_seconds": settings.timeout, "job_activity": True,
                "local_session_cleanup": True}

    @app.post("/v1/conversations", status_code=201, dependencies=[Depends(authorize)])
    async def reserve_conversation():
        cid, now = str(uuid4()), time.time()
        await app.state.store.call(app.state.store.execute,
            "INSERT INTO conversations VALUES (?, NULL, NULL, NULL, ?, ?)", (cid, now, now))
        return {"conversation_id": cid, "thread_id": None}

    def workspace_for(request):
        if request.workspace_dir is None:
            return None
        path = Path(request.workspace_dir).expanduser()
        if not path.is_absolute():
            raise HTTPException(422, "workspace_dir must be an absolute server directory")
        path = path.resolve()
        if not any(path.is_relative_to(root) and path != root for root in settings.write_roots):
            raise HTTPException(422, "workspace_dir must be a subdirectory of CODEX_WRITE_ROOTS")
        if not path.is_dir():
            raise HTTPException(422, "workspace_dir must already exist on the server")
        return path

    async def resumable(previous):
        if await app.state.store.call(app.state.store.local_session_deleted, previous["id"]):
            return False
        if previous["thread_id"]:
            return True
        # A reserved conversation has no turns; a failed first turn is not a reservation.
        return not await app.state.store.call(app.state.store.history, previous["id"], 1, 0)

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        data = []
        try:
            cache = json.loads((settings.codex_home / "models_cache.json").read_text())
            for item in cache.get("models", []):
                if item.get("slug") and item.get("visibility") != "hide":
                    data.append({"id": item["slug"], "name": item.get("display_name", item["slug"])})
        except (OSError, ValueError):
            pass
        return {"default_model": settings.default_model, "data": data, "source": "local_codex_cache",
                "note": "Cache may be stale or empty. Model IDs are passed to Codex; account access is checked there."}

    async def validate_request(request):
        workspace_for(request)
        if request.conversation_id:
            previous = await app.state.store.call(app.state.store.get, str(request.conversation_id))
            if previous is None:
                raise HTTPException(404, "Conversation not found")
            if not await resumable(previous):
                raise HTTPException(409, "Conversation has no Codex thread; start a new conversation")

    async def execute_chat(request: ChatRequest, prepared_images=None):
        started = time.monotonic()
        cid = str(request.conversation_id or uuid4())
        activity = job_activity.get()
        if activity is not None:
            activity["conversation_id"] = cid
        entry = session_locks.setdefault(cid, [asyncio.Lock(), 0])
        entry[1] += 1
        turn_id = None
        store = app.state.store
        try:
            async with asyncio.timeout(settings.timeout), entry[0], concurrency_slot():
                previous = await store.call(store.get, cid) if request.conversation_id else None
                if request.conversation_id and previous is None:
                    raise HTTPException(404, "Conversation not found")
                if previous and not await resumable(previous):
                    raise HTTPException(409, "Conversation has no Codex thread; start a new conversation")
                workspace = workspace_for(request)
                model = request.model or (previous["model"] if previous else None) or settings.default_model
                effort = request.reasoning_effort or (previous["reasoning_effort"] if previous else None)
                with tempfile.TemporaryDirectory(prefix="request-", dir=settings.runtime) as tmp:
                    directory = Path(tmp)
                    try:
                        images = prepared_images if prepared_images is not None else await asyncio.to_thread(
                            prepare_images, request.images, directory, settings.image_roots)
                    except ValueError as exc:
                        raise HTTPException(422, str(exc)) from exc
                    turn_id = await store.call(store.begin_turn, cid, previous is None, model, effort, request.prompt, len(images))

                    async def on_thread(thread_id):
                        await store.call(store.execute, "UPDATE conversations SET thread_id=?,updated_at=? WHERE id=?",
                                      (thread_id, time.time(), cid))

                    result = await app.state.backend.run(
                        prompt=request.prompt, images=images, model=model, reasoning_effort=effort,
                        thread_id=previous["thread_id"] if previous else None,
                        output_dir=directory, on_thread=on_thread,
                        **({"workspace_dir": workspace} if workspace is not None else {}),
                        conversation_id=cid)
                    await store.call(store.complete_turn, turn_id, cid, result, model, effort)
                    return ChatResponse(conversation_id=cid, thread_id=result.thread_id, model=model,
                                        reasoning_effort=effort, answer=result.answer, usage=result.usage,
                                        elapsed_seconds=round(time.monotonic() - started, 3))
        except TimeoutError as exc:
            if turn_id:
                await store.call(store.execute, "UPDATE turns SET status='timeout',error='Request timed out' WHERE id=?", (turn_id,))
            raise HTTPException(504, {"message": "Codex execution timed out",
                                      "conversation_id": cid if turn_id else None}) from exc
        except asyncio.CancelledError:
            if turn_id:
                await store.call(store.execute, "UPDATE turns SET status='interrupted',error='Request cancelled' WHERE id=?", (turn_id,))
            raise
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Codex request failed; conversation_id=%s", cid)
            if turn_id:
                await store.call(store.execute, "UPDATE turns SET status='failed',error=? WHERE id=?", (str(exc)[-4000:], turn_id))
            status = 502 if isinstance(exc, BackendError) else 500
            raise HTTPException(status, {"message": "Codex request failed; inspect server logs and conversation history",
                                         "conversation_id": cid if turn_id else None}) from exc
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                session_locks.pop(cid, None)

    async def chat(request: ChatRequest):
        job = (await app.state.queue.submit([request]))[0]
        result = await app.state.queue.wait(job["id"])
        if result["status"] != "completed":
            raise HTTPException(result["status_code"] or 500, result["error"])
        return ChatResponse.model_validate(result["response"])

    app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(authorize)])(chat)

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(authorize)])
    async def submit_job(request: ChatRequest, idempotency_key: str | None = Header(default=None, max_length=200)):
        return (await app.state.queue.submit([request], idempotency_key))[0]

    @app.post("/v1/jobs/batch", status_code=202, dependencies=[Depends(authorize)])
    async def submit_jobs(request: JobBatchRequest, idempotency_key: str | None = Header(default=None, max_length=200)):
        return {"jobs": await app.state.queue.submit(request.requests, idempotency_key)}

    @app.post("/v1/jobs/query", dependencies=[Depends(authorize)])
    async def query_jobs(request: JobQuery):
        ids = [str(job_id) for job_id in request.job_ids]
        results = await app.state.queue.db(app.state.queue.store.get_many, ids)
        results = [app.state.queue.with_activity(result) for result in results]
        return {"jobs": results, "missing_ids": [job_id for job_id, result in zip(ids, results) if result is None]}

    @app.get("/v1/queue", dependencies=[Depends(authorize)])
    async def queue_stats():
        return {**await app.state.queue.stats(), "history_database":app.state.store.metrics()}

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    async def get_job(job_id: str, wait_seconds: float = Query(default=0, ge=0, le=30)):
        return await app.state.queue.wait(job_id, wait_seconds)

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(authorize)])
    async def cancel_job(job_id: str):
        return await app.state.queue.cancel(job_id)

    @app.post("/v1/jobs/{job_id}/cleanup-session", dependencies=[Depends(authorize)])
    async def cleanup_session(job_id: str):
        queue, store = app.state.queue, app.state.store
        job = await queue.get(job_id)
        if job["status"] not in TERMINAL:
            raise HTTPException(409, "Stop the job and wait for its terminal state before cleanup")
        cid = job.get("conversation_id")
        if not cid:
            # A queued cancellation never started a thread. Legacy interrupted
            # jobs may lack the mapping; do not guess which local file to delete.
            if job.get("started_at") is not None and not (job.get("error") or {}).get("session_not_started"):
                raise HTTPException(409, "Job has no conversation mapping; cannot safely identify its session")
            return {"job_id": job_id, "thread_id": None, "deleted_files": []}
        entry = session_locks.setdefault(cid, [asyncio.Lock(), 0])
        if entry[1]:
            raise HTTPException(409, "Conversation is in use")
        entry[1] += 1
        try:
            async with entry[0], queue.control:
                pending = await queue.db(queue.store.conversation_pending, cid)
                if pending:
                    raise HTTPException(409, "Conversation still has pending jobs")
                previous = await store.call(store.get, cid)
                thread_id = previous["thread_id"] if previous else None
                deleted = []
                if thread_id:
                    # Prevent any new execution from resuming this thread, even
                    # if admission raced cleanup or unlink fails part way through.
                    await store.call(store.execute,
                        "INSERT OR REPLACE INTO deleted_local_sessions VALUES (?, ?, ?)",
                        (cid, thread_id, time.time()))
                    try:
                        # New executions persist under the API runtime, including
                        # an empty home after cancellation. Do not scan unrelated
                        # global sessions or node-local links for these jobs.
                        home = conversation_session_home(settings.runtime, cid)
                        if not home.exists() and not home.is_symlink():
                            # Retain cleanup for records from the shared-home era.
                            home = settings.codex_home
                        deleted = await asyncio.to_thread(delete_session_files, home, thread_id)
                    except (OSError, ValueError) as exc:
                        logger.exception("Local session cleanup failed for %s", thread_id)
                        raise HTTPException(500, "Local session cleanup failed; retry cleanup") from exc
                return {"job_id": job_id, "thread_id": thread_id, "deleted_files": deleted}
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                session_locks.pop(cid, None)

    @app.post("/v1/batch", dependencies=[Depends(authorize)])
    async def batch(request: BatchRequest):
        # Group dependent turns so their order is explicit even if another turn fails.
        groups = {}
        results = [None] * len(request.requests)
        for index, item in enumerate(request.requests):
            key = str(item.conversation_id) if item.conversation_id else f"new-{index}"
            groups.setdefault(key, []).append((index, item))

        async def run_group(items):
            for index, item in items:
                try:
                    result = await chat(item)
                    results[index] = {"index": index, "ok": True, "response": result.model_dump()}
                except HTTPException as exc:
                    results[index] = {"index": index, "ok": False, "status_code": exc.status_code, "error": exc.detail}

        await asyncio.gather(*(run_group(items) for items in groups.values()))
        return {"results": results}

    @app.get("/v1/conversations/{conversation_id}", dependencies=[Depends(authorize)])
    async def conversation(conversation_id: str, limit: int = Query(default=100, ge=1, le=1000),
                           offset: int = Query(default=0, ge=0)):
        result = await app.state.store.call(app.state.store.get, conversation_id)
        if result is None:
            raise HTTPException(404, "Conversation not found")
        return {**result, "turns": await app.state.store.call(app.state.store.history, conversation_id, limit, offset)}

    return app
