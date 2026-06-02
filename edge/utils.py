from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Event:
    kind: str  # "normal" | "near_fall" | "fall" | "immobile"
    confidence: float
    timestamp_ms: int
    details: dict[str, Any]


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def now_ms() -> int:
    # local, monotonic-ish enough for demo; replace with time.time_ns() if needed
    import time

    return int(time.time() * 1000)

