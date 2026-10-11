"""Durable FIFO admission and continuously refilled, bounded Codex execution.

Waiting jobs live in SQLite, not one asyncio task per queued request. SQLite work
and image validation run off the HTTP event loop. Use one server process.
"""
import asyncio
import hashlib
import json
import logging
import shutil
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from .backend import job_activity, prepare_images
from .schemas import ChatRequest

logger = logging.getLogger("api_server.queue")
TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})
PUBLIC_COLUMNS = "id,status,conversation_id,created_at,started_at,finished_at,response,error,status_code"


def public(row):
    result = dict(row)
    for name in ("response", "error"):
        result[name] = json.loads(result[name]) if result[name] else None
    result["queue_seconds"] = round((result["started_at"] or result["finished_at"] or time.time()) - result["created_at"], 3)
    return result


class QueueStore:
    def __init__(self, path, max_pending, max_bytes):
        self.path, self.max_pending, self.max_bytes = path, max_pending, max_bytes
        self.connection = None

    def connect(self):
        if self.connection is None:
            self.connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA journal_mode=PERSIST")
        return self.connection

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def initialize(self):
        with self.connect() as db:
            # Keep rollback journaling: this installation can use a network filesystem.
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL, conversation_id TEXT, request TEXT,
                    images TEXT, input_bytes INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL, started_at REAL, finished_at REAL,
                    response TEXT, error TEXT, status_code INTEGER);
                CREATE INDEX IF NOT EXISTS jobs_status_seq ON jobs(status,seq);
                CREATE INDEX IF NOT EXISTS jobs_finished_at ON jobs(finished_at);
                CREATE INDEX IF NOT EXISTS jobs_conversation_seq ON jobs(conversation_id,seq)
                    WHERE status IN ('queued','running','cancelling');
                CREATE TABLE IF NOT EXISTS submissions (
                    key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, job_ids TEXT NOT NULL);
            """)
            db.execute("""UPDATE jobs SET status='interrupted', finished_at=?, status_code=503,
                error=?, request=NULL, images=NULL, input_bytes=0
                WHERE status IN ('running','cancelling')""",
                (time.time(), json.dumps({"message":"Server stopped during execution; not automatically retried"})))
            return {row[0] for row in db.execute("SELECT id FROM jobs WHERE status='queued'")}

    def lookup(self, key, fingerprint):
        if key is None:
            return None
        with self.connect() as db:
            row = db.execute("SELECT * FROM submissions WHERE key=?", (key,)).fetchone()
            if row:
                if row["fingerprint"] != fingerprint:
                    raise HTTPException(409, "Idempotency-Key already used with different inputs")
                return json.loads(row["job_ids"])
        return None

    def admit(self, rows, key, fingerprint):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if key:
                old = db.execute("SELECT * FROM submissions WHERE key=?", (key,)).fetchone()
                if old:
                    if old["fingerprint"] != fingerprint:
                        raise HTTPException(409, "Idempotency-Key already used with different inputs")
                    return json.loads(old["job_ids"]), False
            count, size = db.execute("""SELECT count(*),coalesce(sum(input_bytes),0) FROM jobs
                WHERE status IN ('queued','running','cancelling')""").fetchone()
            incoming = sum(row["input_bytes"] for row in rows)
            if incoming > self.max_bytes:
                raise HTTPException(413, "Submission exceeds the queue input byte budget")
            if count + len(rows) > self.max_pending or size + incoming > self.max_bytes:
                raise HTTPException(429, "Queue capacity exceeded; retry later with the same Idempotency-Key",
                                    headers={"Retry-After":"2"})
            db.executemany("""INSERT INTO jobs
                (id,status,conversation_id,request,images,input_bytes,created_at)
                VALUES (:id,'queued',:conversation_id,:request,:images,:input_bytes,:created_at)""", rows)
            ids = [row["id"] for row in rows]
            if key:
                db.execute("INSERT INTO submissions VALUES (?,?,?)", (key, fingerprint, json.dumps(ids)))
            return ids, True

    def claim(self, count):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            query = """SELECT * FROM jobs j WHERE j.status='queued'
                AND (j.conversation_id IS NULL OR NOT EXISTS (
                    SELECT 1 FROM jobs earlier WHERE earlier.conversation_id=j.conversation_id
                    AND earlier.seq<j.seq AND earlier.status IN ('queued','running','cancelling')))
                ORDER BY j.seq"""
            rows = db.execute(query if count is None else query + " LIMIT ?",
                              () if count is None else (count,)).fetchall()
            db.executemany("UPDATE jobs SET status='running',started_at=? WHERE id=?",
                           [(time.time(), row["id"]) for row in rows])
            return [dict(row) for row in rows]

    def finish(self, job_id, status, response, error, status_code):
        with self.connect() as db:
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["status"] in TERMINAL:
                return
            cid = response.get("conversation_id") if response else None
            if not cid and isinstance(error, dict):
                cid = error.get("conversation_id")
            db.execute("""UPDATE jobs SET status=?, response=?,error=?,status_code=?,finished_at=?,
                conversation_id=coalesce(?,conversation_id),request=NULL,images=NULL,input_bytes=0 WHERE id=?""",
                (status, json.dumps(response) if response else None, json.dumps(error) if error else None,
                 status_code, time.time(), cid, job_id))

    def conversation_pending(self, cid):
        return self.connect().execute(
            "SELECT 1 FROM jobs WHERE conversation_id=? AND status IN ('queued','running','cancelling') LIMIT 1",
            (cid,)).fetchone() is not None

    def mark_cancelling(self, job_id):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='cancelling' WHERE id=? AND status='running'", (job_id,))

    def get_many(self, ids):
        with self.connect() as db:
            marks = ",".join("?" for _ in ids)
            rows = db.execute(f"SELECT {PUBLIC_COLUMNS} FROM jobs WHERE id IN ({marks})", ids).fetchall()
            found = {row["id"]:public(row) for row in rows}
            return [found.get(job_id) for job_id in ids]

    def stats(self):
        with self.connect() as db:
            counts = {row[0]:row[1] for row in db.execute("SELECT status,count(*) FROM jobs GROUP BY status")}
            pending_bytes = db.execute("SELECT coalesce(sum(input_bytes),0) FROM jobs WHERE status IN ('queued','running','cancelling')").fetchone()[0]
            oldest = db.execute("SELECT min(created_at) FROM jobs WHERE status='queued'").fetchone()[0]
            recent = db.execute("""SELECT count(*) AS finished,
                coalesce(sum(status='completed'),0) AS completed,
                coalesce(sum(status='failed'),0) AS failed,
                avg(CASE WHEN started_at IS NOT NULL THEN started_at-created_at END) AS avg_queue_seconds,
                avg(CASE WHEN started_at IS NOT NULL THEN finished_at-started_at END) AS avg_execution_seconds
                FROM jobs WHERE finished_at >= ?""", (time.time()-60,)).fetchone()
            recent = dict(recent)
            recent["window_seconds"] = 60
            recent["completed_per_second"] = round(recent["completed"] / 60, 3)
        return {"counts":counts, "pending":sum(counts.get(s, 0) for s in ("queued","running","cancelling")),
                "max_pending":self.max_pending, "pending_input_bytes":pending_bytes,
                "max_pending_input_bytes":self.max_bytes,
                "oldest_queued_seconds":round(time.time()-oldest, 3) if oldest else 0,
                "recent":recent}


class JobQueue:
    def __init__(self, settings, execute, validate):
        self.settings, self.execute, self.validate = settings, execute, validate
        self.store = QueueStore(settings.runtime / "jobs.sqlite3", settings.max_pending, settings.max_pending_bytes)
        self.inputs = settings.runtime / "job-inputs"
        self.database_lock = asyncio.Lock()
        self.control = asyncio.Lock()
        self.ingest = asyncio.Semaphore(2)
        self.wakeup = asyncio.Event()
        self.active = {}
        self.finalizers = set()
        self.watchers = {}
        self.stopping = False
        self.dispatcher = None

    async def db(self, method, *args):
        async with self.database_lock:
            # Shield the thread so a cancelled caller cannot overlap a still-running
            # commit with cleanup or another database operation.
            task = asyncio.create_task(asyncio.to_thread(method, *args))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise

    async def start(self):
        self.inputs.mkdir(exist_ok=True, mode=0o700)
        queued = await self.db(self.store.initialize)

        def cleanup_orphans():
            for directory in self.inputs.iterdir():
                if directory.is_dir() and directory.name not in queued:
                    shutil.rmtree(directory)

        await asyncio.to_thread(cleanup_orphans)
        self.dispatcher = asyncio.create_task(self.dispatch(), name="codex-queue-dispatcher")
        self.wakeup.set()

    async def close(self):
        self.stopping = True
        self.wakeup.set()
        if self.dispatcher:
            await self.dispatcher
        for task in list(self.active.values()):
            task.cancel()
        await asyncio.gather(*self.active.values(), return_exceptions=True)
        # done callbacks have scheduled finalizers by this point.
        while self.finalizers:
            await asyncio.gather(*list(self.finalizers))
        await self.db(self.store.close)

    async def submit(self, requests, key=None):
        if self.stopping:
            raise HTTPException(503, "Server is shutting down")
        # Fingerprints include the complete input; image data is not stored in SQLite.
        encoded = json.dumps([r.model_dump(mode="json") for r in requests], sort_keys=True, ensure_ascii=False)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        del encoded
        old = await self.db(self.store.lookup, key, fingerprint)
        if old:
            return await self.db(self.store.get_many, old)
        if len(requests) > self.settings.max_pending:
            raise HTTPException(413, "Submission has more jobs than the queue capacity")
        for request in requests:
            await self.validate(request)
        rows, directories = [], []

        def stage():
            for request in requests:
                job_id = str(uuid4())
                paths = []
                if request.images:
                    directory = self.inputs / job_id
                    directory.mkdir(mode=0o700)
                    directories.append(directory)
                    paths = prepare_images(request.images, directory, self.settings.image_roots)
                payload = request.model_copy(update={"images":[]}).model_dump_json()
                rows.append({"id":job_id, "conversation_id":str(request.conversation_id) if request.conversation_id else None,
                             "request":payload, "images":json.dumps([str(p) for p in paths]),
                             "input_bytes":len(payload.encode()) + sum(p.stat().st_size for p in paths),
                             "created_at":time.time()})

        def cleanup():
            for directory in directories:
                shutil.rmtree(directory, ignore_errors=True)

        async with self.ingest:
            try:
                staging = asyncio.create_task(asyncio.to_thread(stage))
                try:
                    await asyncio.shield(staging)
                except asyncio.CancelledError:
                    await staging
                    raise
                # Never cancel between commit and establishing ownership of snapshots.
                admission = asyncio.create_task(self.db(self.store.admit, rows, key, fingerprint))
                try:
                    ids, inserted = await asyncio.shield(admission)
                except asyncio.CancelledError:
                    ids, inserted = await admission
                    if inserted:
                        directories.clear()
                        self.wakeup.set()
                    raise
                if inserted:
                    directories.clear()
                    self.wakeup.set()
                return await self.db(self.store.get_many, ids)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            finally:
                await asyncio.to_thread(cleanup)

    async def dispatch(self):
        while not self.stopping:
            self.wakeup.clear()
            try:
                async with self.control:
                    available = (self.settings.max_concurrency - len(self.active)
                                 if self.settings.max_concurrency else None)
                    rows = await self.db(self.store.claim, available) if available is None or available > 0 else []
                    for row in rows:
                        task = asyncio.create_task(self.run(row), name=f"codex-job-{row['id']}")
                        self.active[row["id"]] = task
                        task.add_done_callback(lambda done, job=row: self.completed(job, done))
            except Exception:
                logger.exception("Queue dispatch failed; retrying")
                await asyncio.sleep(1)
                self.wakeup.set()
            # A periodic wakeup also recovers if an HTTP handler stopped immediately
            # after committing a submission.
            try:
                await asyncio.wait_for(self.wakeup.wait(), 1)
            except TimeoutError:
                pass

    async def run(self, row):
        request = ChatRequest.model_validate_json(row["request"])
        activity = {}
        # Keep telemetry on the worker until its result has been persisted.
        asyncio.current_task().codex_activity = activity
        token = job_activity.set(activity)
        try:
            return await self.execute(request, [Path(p) for p in json.loads(row["images"])])
        finally:
            job_activity.reset(token)

    def completed(self, row, task):
        finalizer = asyncio.create_task(self.finalize(row, task))
        self.finalizers.add(finalizer)
        finalizer.add_done_callback(self.finalizers.discard)

    async def finalize(self, row, task):
        response, error, code = None, None, 200
        try:
            response = task.result().model_dump()
            status = "completed"
        except asyncio.CancelledError:
            status = "interrupted" if self.stopping else "cancelled"
            error, code = {"message":"Server stopped" if self.stopping else "Job cancelled"}, 503 if self.stopping else 409
            cid = getattr(task, "codex_activity", {}).get("conversation_id")
            if cid:
                error["conversation_id"] = cid
            elif not row.get("conversation_id"):
                error["session_not_started"] = True
        except HTTPException as exc:
            status, error, code = "failed", exc.detail, exc.status_code
        except Exception:
            logger.exception("Unexpected queue worker failure: %s", row["id"])
            status, error, code = "failed", {"message":"Unexpected worker failure; inspect server logs"}, 500
        # Keep the slot occupied until the durable terminal state has committed.
        persisted = False
        while True:
            try:
                await self.db(self.store.finish, row["id"], status, response, error, code)
                persisted = True
                break
            except Exception:
                if self.stopping:
                    logger.exception("Result could not be persisted during shutdown: %s; startup will mark it interrupted", row["id"])
                    break
                logger.exception("Cannot persist job result: %s; retrying", row["id"])
                await asyncio.sleep(1)
        if persisted and json.loads(row["images"]):
            await asyncio.to_thread(shutil.rmtree, self.inputs / row["id"], True)
        self.active.pop(row["id"], None)
        if row["id"] in self.watchers:
            self.watchers[row["id"]][0].set()
        self.wakeup.set()

    async def get(self, job_id):
        result = (await self.db(self.store.get_many, [job_id]))[0]
        if result is None:
            raise HTTPException(404, "Job not found")
        return self.with_activity(result)

    def with_activity(self, result):
        if result is None:
            return None
        task = self.active.get(result["id"])
        activity = getattr(task, "codex_activity", {})
        process = activity.get("process")
        now = time.monotonic()
        result["activity"] = {
            "worker_alive": task is not None and not task.done(),
            "process_alive": process.returncode is None if process is not None else None,
            "pid": process.pid if process is not None else None,
            "exit_code": process.returncode if process is not None else None,
            "elapsed_seconds": round(now - activity["started_at"], 1) if "started_at" in activity else None,
            "last_activity_seconds_ago": round(now - activity["last_activity_at"], 1) if "last_activity_at" in activity else None,
            "last_event": activity.get("last_event"),
            "event_count": activity.get("event_count", 0),
        }
        return result

    async def wait(self, job_id, timeout=None):
        entry = self.watchers.setdefault(job_id, [asyncio.Event(), 0])
        entry[1] += 1
        deadline = time.monotonic() + timeout if timeout is not None else None
        try:
            while True:
                entry[0].clear()
                result = await self.get(job_id)
                remaining = deadline - time.monotonic() if deadline is not None else None
                if result["status"] in TERMINAL or (remaining is not None and remaining <= 0):
                    return result
                try:
                    await asyncio.wait_for(entry[0].wait(), remaining)
                except TimeoutError:
                    return await self.get(job_id)
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                self.watchers.pop(job_id, None)

    async def cancel(self, job_id):
        async with self.control:
            result = await self.get(job_id)
            if result["status"] == "queued":
                await self.db(self.store.finish, job_id, "cancelled", None, {"message":"Cancelled before execution"}, 409)
                await asyncio.to_thread(shutil.rmtree, self.inputs / job_id, True)
                if job_id in self.watchers:
                    self.watchers[job_id][0].set()
                self.wakeup.set()
            elif result["status"] == "running":
                await self.db(self.store.mark_cancelling, job_id)
                task = self.active.get(job_id)
                if task and not task.done():
                    task.cancel()
        return await self.get(job_id)

    async def stats(self):
        result = await self.db(self.store.stats)
        return {**result, "max_concurrency":self.settings.max_concurrency,
                "active_workers":len(self.active), "scheduler_alive":self.dispatcher is not None and not self.dispatcher.done()}
