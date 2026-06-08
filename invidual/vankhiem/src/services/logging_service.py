"""JSONL logging for retrieval and generation traces."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import GENERATION_LOG_PATH, RETRIEVAL_LOG_PATH


class LoggingService:
    def __init__(
        self,
        retrieval_log_path: Path = RETRIEVAL_LOG_PATH,
        generation_log_path: Path = GENERATION_LOG_PATH,
    ) -> None:
        self.retrieval_log_path = retrieval_log_path
        self.generation_log_path = generation_log_path

    def log_retrieval(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.retrieval_log_path, record)

    def log_generation(self, record: dict[str, Any]) -> None:
        self._append_jsonl(self.generation_log_path, record)

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
