"""Append-only JSONL run logging.

CONTRACT: one JSON object per line, each stamped with an injected "ts" field
(ISO-8601 UTC). This is the only logging mechanism used inside training/eval/
XAI loops — never `print` structured results that later code needs to parse.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Appends one JSON object per line to `path`, creating parent dirs as needed.

    Each call to `.log(row)` injects a fresh `"ts"` key (ISO-8601 UTC,
    `datetime.isoformat()`) into a copy of `row`, serializes it, and flushes
    immediately so partial runs still leave a readable log on disk.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        """Append `row` as one JSON line, with a "ts" field injected."""
        record = {**row, "ts": datetime.now(UTC).isoformat()}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
