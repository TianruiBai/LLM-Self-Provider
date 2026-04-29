"""Per-actor token-bucket rate limit (A11).

Buckets live in the ``rate_buckets`` SQLite table; refilled lazily on each
:func:`consume`. Bucket key examples::

    user:<id>:v1            chat / completion / embeddings (per-user budget)
    ip:<ip>:login           failed-login throttle (anonymous)
    key:<id>:v1             optional per-API-key cap

Three layers are wired by :class:`RateLimitMiddleware`:

1. ``ip:<ip>:login`` — 10 / 15 min on POST /auth/login + /auth/bootstrap.
2. ``user:<id>:v1``  — configurable in ``config_kv`` (default 60 / minute).
3. ``key:<key_id>:v1`` — optional, only if the api_key row sets ``rate_per_min``
   (future column; for now we read from ``config_kv['ratelimit.api_key.<id>.rpm']``).

Returns ``(allowed: bool, retry_after_s: float)``.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

from . import db

log = logging.getLogger("provider.ratelimit")


# ---------------------------------------------------------- defaults

# bucket key → (capacity, refill_per_second)
DEFAULTS: dict[str, tuple[float, float]] = {
    # 60 req/min sustained, 60 burst.
    "user.v1": (60.0, 60.0 / 60.0),
    # 10 attempts / 15 min — strict.
    "ip.login": (10.0, 10.0 / (15 * 60)),
}


def _config_override(key: str) -> Optional[tuple[float, float]]:
    """Look up an override in ``config_kv`` (key = 'ratelimit.<bucket>')."""
    try:
        row = db.fetchone("SELECT value FROM config_kv WHERE key=?", (f"ratelimit.{key}",))
    except Exception:  # noqa: BLE001
        return None
    if not row:
        return None
    try:
        v = json.loads(row["value"])
        return float(v["capacity"]), float(v["refill_per_s"])
    except Exception:  # noqa: BLE001
        return None


def get_policy(name: str) -> tuple[float, float]:
    return _config_override(name) or DEFAULTS.get(name) or (1e9, 1e9)


# ---------------------------------------------------------- consume

@dataclass
class Decision:
    allowed: bool
    retry_after_s: float
    remaining: float
    capacity: float


def consume(bucket: str, policy: str, cost: float = 1.0) -> Decision:
    """Atomic consume of ``cost`` tokens from ``bucket`` under ``policy``.

    Single-process SQLite serialises writes so a transaction is enough.
    """
    capacity, refill = get_policy(policy)
    now = time.time()
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT tokens, updated_at FROM rate_buckets WHERE key = ?",
            (bucket,),
        ).fetchone()
        if row is None:
            tokens = capacity
            updated = now
        else:
            tokens = float(row["tokens"])
            updated = float(row["updated_at"])
            tokens = min(capacity, tokens + (now - updated) * refill)

        allowed = tokens >= cost
        if allowed:
            tokens -= cost
            retry = 0.0
        else:
            deficit = cost - tokens
            retry = deficit / refill if refill > 0 else 60.0

        conn.execute(
            """
            INSERT INTO rate_buckets(key, tokens, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET tokens=excluded.tokens, updated_at=excluded.updated_at
            """,
            (bucket, tokens, int(now)),
        )

    return Decision(allowed=allowed, retry_after_s=retry, remaining=tokens, capacity=capacity)


def reset(bucket: str) -> None:
    """Wipe a single bucket (admin tool)."""
    db.execute("DELETE FROM rate_buckets WHERE key = ?", (bucket,))


__all__ = ["consume", "reset", "get_policy", "Decision", "DEFAULTS"]
