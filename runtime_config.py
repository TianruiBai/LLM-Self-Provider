"""Persistent, mutable runtime configuration set by the UI.

Stored at provider/data/runtime_config.json. Values are also pushed to
``os.environ`` so existing helpers (e.g. ``tools.py`` reading
``TAVILY_API_KEY``) pick them up without code changes.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)

# Map of "config key" -> "environment variable to mirror it into".
ENV_BINDINGS: dict[str, str] = {
    "tavily_api_key": "TAVILY_API_KEY",
}

_lock = threading.Lock()
_path: Path | None = None
_state: dict[str, Any] = {}


def _config_path() -> Path:
    global _path
    if _path is None:
        base = Path(__file__).resolve().parent / "data"
        base.mkdir(parents=True, exist_ok=True)
        _path = base / "runtime_config.json"
    return _path


def load() -> dict[str, Any]:
    """Load runtime config from disk and apply env bindings. Idempotent."""
    global _state
    with _lock:
        p = _config_path()
        if p.exists():
            try:
                _state = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                log.warning("Failed to parse runtime_config.json: %s", e)
                _state = {}
        else:
            _state = {}
        _apply_env_locked()
        return dict(_state)


def get_all() -> dict[str, Any]:
    with _lock:
        return _redact_secrets(dict(_state))


def update(patch: dict[str, Any]) -> dict[str, Any]:
    """Merge a partial update, persist, and re-apply env bindings."""
    global _state
    with _lock:
        _state.update({k: v for k, v in (patch or {}).items() if v is not None})
        # Allow explicit deletion via empty string for env-bound keys.
        for k, v in list((patch or {}).items()):
            if k in ENV_BINDINGS and (v is None or v == ""):
                _state.pop(k, None)
        try:
            _config_path().write_text(json.dumps(_state, indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to persist runtime_config.json: %s", e)
        _apply_env_locked()
        return _redact_secrets(dict(_state))


def _apply_env_locked() -> None:
    for key, env in ENV_BINDINGS.items():
        val = _state.get(key)
        if val:
            os.environ[env] = str(val)
        else:
            os.environ.pop(env, None)


def _redact_secrets(d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)
    for k in ENV_BINDINGS:
        if k in out and out[k]:
            s = str(out[k])
            out[k] = (s[:4] + "…" + s[-4:]) if len(s) > 10 else "set"
    return out
