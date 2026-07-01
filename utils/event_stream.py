from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

_sink_path: Path | None = None
_lock = Lock()


def configure_event_sink(path: Path | None) -> None:
    global _sink_path
    _sink_path = path
    if _sink_path is not None:
        _sink_path.parent.mkdir(parents=True, exist_ok=True)


def _classify(component: str) -> dict[str, str]:
    c = component.lower()
    if c.startswith("stage."):
        return {"layer": "stage", "scope": c.split(".")[1] if "." in c else "unknown"}
    if c.startswith("agent."):
        return {"layer": "agent", "scope": c.split(".")[1] if "." in c else "unknown"}
    if c.startswith("mcts.node"):
        return {"layer": "node", "scope": "mcts"}
    if c.startswith("pipeline"):
        return {"layer": "pipeline", "scope": "pipeline"}
    return {"layer": "misc", "scope": c.split(".")[0] if c else "unknown"}


def emit_event(component: str, event: str, **fields: Any) -> None:
    if _sink_path is None:
        return
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "event": event,
        "fields": fields,
        "classification": _classify(component),
    }
    with _lock:
        with _sink_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
