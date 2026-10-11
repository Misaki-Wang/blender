"""Remove only rollout files belonging to a known, stopped Codex thread."""
import json
from pathlib import Path
from uuid import UUID


def delete_session_files(codex_home, thread_id):
    thread_id = str(UUID(thread_id))
    home = Path(codex_home).resolve()
    deleted = []
    for name in ("sessions", "archived_sessions"):
        configured_root = home / name
        if not configured_root.exists() and not configured_root.is_symlink():
            continue
        # Operators may relocate these two directories to local storage. Resolve
        # only this configured root; links inside it are still rejected/skipped.
        # A dangling root must fail rather than falsely report successful cleanup.
        root = configured_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Session directory must be a directory: {root}")
        for path in root.rglob(f"rollout-*-{thread_id}.jsonl"):
            if path.is_symlink() or path.resolve() != path or not path.is_file():
                raise ValueError(f"Session file must be an ordinary file: {path}")
            # Check ownership without reading the (potentially huge) transcript.
            with path.open(encoding="utf-8") as source:
                metadata = json.loads(source.readline(1024 * 1024))
            if metadata.get("type") != "session_meta" or metadata.get("payload", {}).get("id") != thread_id:
                raise ValueError(f"Session metadata does not match thread {thread_id}")
            path.unlink()
            deleted.append(str(path))
    return deleted
