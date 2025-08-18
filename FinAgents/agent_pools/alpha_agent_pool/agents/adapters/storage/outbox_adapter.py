from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Any

from corepkg.ports.result import ResultPort


class FileOutboxAdapter(ResultPort):
    """Durable outbox writing results/events to the filesystem (at-least-once)."""

    def __init__(self, base_dir: str):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def publish(self, result: Dict[str, Any]) -> None:
        content = json.dumps(result, ensure_ascii=False, indent=2)
        with self._lock:
            path = self._base / f"result_{result.get('task_id','unknown')}.json"
            path.write_text(content)

    def store_artifact(self, blob_ref: str, metadata: Dict[str, Any]) -> None:
        # metadata-only stub
        with self._lock:
            path = self._base / f"artifact_{metadata.get('artifact_id','unknown')}.json"
            path.write_text(json.dumps({"ref": blob_ref, "metadata": metadata}, ensure_ascii=False, indent=2))

    def emit_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            path = self._base / f"event_{event.get('type','generic')}_{event.get('ts','')}.json"
            path.write_text(json.dumps(event, ensure_ascii=False, indent=2))

