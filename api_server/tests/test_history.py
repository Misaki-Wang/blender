import asyncio
import sqlite3
import threading

import pytest

from api_server.history import Store


def test_group_commit_isolates_failed_operation_and_is_durable(tmp_path):
    async def run():
        store = Store(tmp_path / "history.sqlite3")

        def bad_operation():
            store.execute("INSERT INTO conversations VALUES ('bad',NULL,NULL,NULL,0,0)")
            raise ValueError("Rollback this operation only")

        try:
            operations = [store.call(store.execute, "INSERT INTO conversations VALUES (?,NULL,NULL,NULL,0,0)", (str(i),))
                          for i in range(30)]
            operations.insert(10, store.call(bad_operation))
            results = await asyncio.gather(*operations, return_exceptions=True)
            assert isinstance(results[10], ValueError)
            assert sum(isinstance(result, Exception) for result in results) == 1
            # A separate connection must see the entire committed batch already.
            with sqlite3.connect(store.path) as observer:
                assert observer.execute("SELECT count(*) FROM conversations").fetchone()[0] == 30
                assert observer.execute("SELECT count(*) FROM conversations WHERE id='bad'").fetchone()[0] == 0
            assert store.commit_count == 1
            assert store.max_commit_operations == 31
        finally:
            await store.aclose()

    asyncio.run(run())


def test_cancel_waits_for_commit_boundary(tmp_path):
    async def run():
        store = Store(tmp_path / "history.sqlite3")
        entered = threading.Event()
        release = threading.Event()

        def blocked_write():
            entered.set()
            assert release.wait(5)
            store.execute("INSERT INTO conversations VALUES ('committed',NULL,NULL,NULL,0,0)")

        try:
            caller = asyncio.create_task(store.call(blocked_write))
            while not entered.is_set():
                await asyncio.sleep(0.001)
            caller.cancel()
            await asyncio.sleep(0.01)
            assert not caller.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert (await store.call(store.get, "committed"))["id"] == "committed"
        finally:
            release.set()
            await store.aclose()

    asyncio.run(run())


def test_commit_failure_reaches_every_waiter_and_next_batch_recovers(tmp_path):
    async def run():
        store = Store(tmp_path / "history.sqlite3")
        transact = store.transact

        def fail(_):
            raise sqlite3.OperationalError("simulated disk failure")

        store.transact = fail
        try:
            results = await asyncio.gather(*(store.call(store.get, str(i)) for i in range(4)), return_exceptions=True)
            assert all(isinstance(result, sqlite3.OperationalError) for result in results)
            store.transact = transact
            assert await store.call(store.get, "missing") is None
        finally:
            await store.aclose()

    asyncio.run(run())
