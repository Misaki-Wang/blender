import json

from api_server.scheduler import QueueStore


def test_recent_window_excludes_old_results_and_counts_failures(tmp_path, monkeypatch):
    monkeypatch.setattr("api_server.scheduler.time.time", lambda:1000)
    store = QueueStore(tmp_path / "jobs.sqlite3", 100, 100000)
    try:
        store.initialize()
        store.admit([{"id":str(i),"conversation_id":None,"request":"{}","images":"[]","input_bytes":2,"created_at":900}
                     for i in range(4)], None, "unused")
        store.claim(4)
        for i, status in enumerate(["completed","failed","cancelled","completed"]):
            store.finish(str(i),status,{"answer":"ok"} if status == "completed" else None,None,200)
        with store.connect() as db:
            db.execute("UPDATE jobs SET created_at=970,started_at=980,finished_at=990")
            db.execute("UPDATE jobs SET finished_at=900 WHERE id='3'")
        stats = store.stats()
        assert stats["counts"] == {"completed":2,"failed":1,"cancelled":1}
        assert stats["recent"]["finished"] == 3
        assert stats["recent"]["completed"] == 1
        assert stats["recent"]["failed"] == 1
        assert stats["recent"]["avg_queue_seconds"] == 10
        assert stats["recent"]["avg_execution_seconds"] == 10
        assert stats["recent"]["completed_per_second"] == round(1/60,3)
    finally:
        store.close()
