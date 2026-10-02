"""Statistical disclosure control and the node's release (airlock) log."""

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class PrivacyPolicy:
    """Output rules applied to every released result.

    min_cell_count: any count (or group within a result) below this is suppressed.
    round_counts_to: released counts are rounded to this base.
    """

    min_cell_count: int = 10
    round_counts_to: int = 1

    def releasable(self, n: int) -> bool:
        return n >= self.min_cell_count

    def round_count(self, n: int) -> int:
        return int(self.round_counts_to * round(n / self.round_counts_to))


class AuditLog:
    """Append-only JSONL record of every request a node receives and what it released."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, **entry) -> None:
        line = json.dumps({"timestamp": datetime.now(UTC).isoformat(), **entry}, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
