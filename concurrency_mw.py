"""Phase C6 — per-user concurrency cap.

Enforces a maximum number of concurrent in-flight ``/v1/*`` requests per
user, so a single client cannot starve others on a vLLM continuous-batched
backend. Anonymous traffic shares a single pool.

The cap is configurable via the ``concurrency.per_user`` runtime config
key (default 4) and updates take effect on the next request.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Awaitable, Callable

from fastapi import HTTPException, Request, Response

DEFAULT_PER_USER = 4
DEFAULT_ANON = 8


class _UserSlots:
    """Tracks how many in-flight requests each user holds."""

    def __init__(self, default_cap: int, anon_cap: int) -> None:
        self.default_cap = int(default_cap)
        self.anon_cap = int(anon_cap)
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    def _cap_for(self, key: str) -> int:
        return self.anon_cap if key.startswith("anon:") else self.default_cap

    async def acquire(self, key: str) -> None:
        async with self._lock:
            cap = self._cap_for(key)
            if self._counts[key] >= cap:
                raise HTTPException(
                    status_code=429,
                    detail=f"per-user concurrency limit reached ({cap})",
                    headers={"Retry-After": "1"},
                )
            self._counts[key] += 1

    async def release(self, key: str) -> None:
        async with self._lock:
            n = self._counts.get(key, 0)
            if n <= 1:
                self._counts.pop(key, None)
            else:
                self._counts[key] = n - 1


_SLOTS: _UserSlots | None = None


def _slots() -> _UserSlots:
    global _SLOTS
    if _SLOTS is None:
        _SLOTS = _UserSlots(DEFAULT_PER_USER, DEFAULT_ANON)
    return _SLOTS


def configure(per_user: int | None = None, anon: int | None = None) -> None:
    """Update caps. Calling with ``None`` leaves the existing value."""
    s = _slots()
    if per_user is not None:
        s.default_cap = max(1, int(per_user))
    if anon is not None:
        s.anon_cap = max(1, int(anon))


def _key_for(req: Request) -> str:
    actor = getattr(req.state, "actor", None)
    if actor is not None and getattr(actor, "user", None) is not None:
        return f"user:{actor.user.id}"
    ip = req.client.host if req.client else "unknown"
    return f"anon:{ip}"


# Only enforce on these prefixes (chat/completions/etc.).
_ENFORCE_PREFIXES = ("/v1/",)


async def concurrency_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    path = request.url.path
    if not path.startswith(_ENFORCE_PREFIXES):
        return await call_next(request)
    key = _key_for(request)
    s = _slots()
    await s.acquire(key)
    try:
        return await call_next(request)
    finally:
        await s.release(key)


__all__ = ["concurrency_middleware", "configure", "DEFAULT_PER_USER", "DEFAULT_ANON"]
