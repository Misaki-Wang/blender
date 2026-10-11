"""Conversation history with atomic, grouped durable commits."""
import asyncio
import json
import sqlite3
import time
from collections import deque

class Store:
    def __init__(self, path):
        self.path = path
        self.pending = deque()
        self.flusher = None
        self.closing = False
        self.commit_count = self.operation_count = self.max_commit_operations = 0
        self.connection = None
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, thread_id TEXT, model TEXT, reasoning_effort TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL,
                    prompt TEXT NOT NULL, image_count INTEGER NOT NULL, model TEXT,
                    reasoning_effort TEXT, status TEXT NOT NULL, answer TEXT, usage TEXT,
                    error TEXT, created_at REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS turns_conversation ON turns(conversation_id, id);
                CREATE TABLE IF NOT EXISTS deleted_local_sessions (
                    conversation_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, deleted_at REAL NOT NULL);
            """)
            db.execute("UPDATE turns SET status='interrupted', error='Server stopped during this turn' WHERE status='running'")

    async def call(self, method, *args):
        if self.closing:
            raise RuntimeError("History store is closing")
        future = asyncio.get_running_loop().create_future()
        self.pending.append((method, args, future))
        if self.flusher is None:
            self.flusher = asyncio.create_task(self.flush())
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # A database commit must finish before a caller handles cancellation.
            try:
                await asyncio.shield(future)
            except Exception:
                pass
            raise

    def transact(self, operations):
        db = self.connect()
        outcomes = []
        try:
            db.execute("BEGIN")
            for method, args, _ in operations:
                db.execute("SAVEPOINT history_operation")
                try:
                    result = method(*args)
                except Exception as exc:
                    db.execute("ROLLBACK TO history_operation")
                    outcomes.append((None, exc))
                else:
                    outcomes.append((result, None))
                db.execute("RELEASE history_operation")
            db.commit()
            return outcomes
        except Exception:
            db.rollback()
            raise

    async def flush(self):
        try:
            while self.pending:
                # A small coalescing window amortizes fsync across concurrent turns.
                await asyncio.sleep(0.002)
                operations = [self.pending.popleft() for _ in range(min(128, len(self.pending)))]
                try:
                    outcomes = await asyncio.to_thread(self.transact, operations)
                except Exception as exc:
                    outcomes = [(None, exc)] * len(operations)
                else:
                    self.commit_count += 1
                    self.operation_count += len(operations)
                    self.max_commit_operations = max(self.max_commit_operations, len(operations))
                for (_, _, future), (result, error) in zip(operations, outcomes):
                    if error is None:
                        future.set_result(result)
                    else:
                        future.set_exception(error)
        finally:
            self.flusher = None

    async def aclose(self):
        self.closing = True
        if self.flusher:
            await self.flusher
        await asyncio.to_thread(self.close)

    def metrics(self):
        return {"transactions":self.commit_count, "operations":self.operation_count,
                "max_operations_per_transaction":self.max_commit_operations, "pending_operations":len(self.pending)}

    def connect(self):
        if self.connection is None:
            self.connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA journal_mode=PERSIST")
        return self.connection

    def execute(self, sql, args=()):
        return self.connect().execute(sql, args).lastrowid

    def local_session_deleted(self, cid):
        return self.connect().execute(
            "SELECT 1 FROM deleted_local_sessions WHERE conversation_id=?", (cid,)).fetchone() is not None

    def get(self, cid):
        row = self.connect().execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
        return dict(row) if row else None

    def history(self, cid, limit, offset):
        rows = self.connect().execute("SELECT * FROM turns WHERE conversation_id=? ORDER BY id LIMIT ? OFFSET ?",
                                      (cid, limit, offset)).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            row["usage"] = json.loads(row["usage"]) if row["usage"] else None
        return result

    def begin_turn(self, cid, new, model, effort, prompt, image_count):
        now = time.time()
        db = self.connect()
        if new:
            db.execute("INSERT INTO conversations VALUES (?, NULL, ?, ?, ?, ?)", (cid,model,effort,now,now))
        return db.execute("""INSERT INTO turns
            (conversation_id,prompt,image_count,model,reasoning_effort,status,created_at)
            VALUES (?,?,?,?,?,'running',?)""", (cid,prompt,image_count,model,effort,now)).lastrowid

    def complete_turn(self, turn_id, cid, result, model, effort):
        db = self.connect()
        db.execute("UPDATE turns SET status='completed',answer=?,usage=? WHERE id=?",
                   (result.answer,json.dumps(result.usage),turn_id))
        db.execute("UPDATE conversations SET thread_id=?,model=?,reasoning_effort=?,updated_at=? WHERE id=?",
                   (result.thread_id,model,effort,time.time(),cid))

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
