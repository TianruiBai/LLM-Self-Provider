"""ASGI middleware applying token-bucket rate limits.

Order: must be installed *outside* AuthMiddleware so we can read the resolved
actor (set by AuthMiddleware) before running the bucket math.

Layered checks:

* Anonymous POSTs to /auth/login + /auth/bootstrap → ``ip:<ip>:login`` bucket
  (10 / 15 min).
* Any authenticated /v1/* request → ``user:<id>:v1`` bucket (60 / min default).
* /v1/* via API key with override → ``key:<key_id>:v1`` bucket (admin-set).

On reject: returns 429 with ``Retry-After`` and a JSON body. Audit middleware
still records the row.
"""
from __future__ import annotations

import json
import logging
import os

from . import ratelimit

log = logging.getLogger("provider.ratelimit_mw")

_LOGIN_PATHS: tuple[str, ...] = ("/auth/login", "/auth/login-totp", "/auth/bootstrap")


def _ip_from_scope(scope) -> str | None:
    headers = dict(scope.get("headers") or [])
    if os.environ.get("PROVIDER_TRUST_PROXY", "1") == "1":
        xff = headers.get(b"x-forwarded-for")
        if xff:
            first = xff.decode("latin-1").split(",")[0].strip()
            if first:
                return first
    client = scope.get("client")
    return client[0] if client else None


class RateLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "") or ""
        method = scope.get("method", "") or ""

        # ---- 1) Login throttle (anonymous, by IP) ----
        if method == "POST" and path in _LOGIN_PATHS:
            ip = _ip_from_scope(scope) or "unknown"
            d = ratelimit.consume(f"ip:{ip}:login", "ip.login")
            if not d.allowed:
                await self._reject(send, d, "too many login attempts")
                return

        # ---- 2) Per-user limit on /v1/* ----
        if path.startswith("/v1/"):
            state = scope.get("state")
            actor = None
            if isinstance(state, dict):
                actor = state.get("actor")
            else:
                actor = getattr(state, "actor", None) if state is not None else None
            if actor is not None and actor.user is not None:
                # Per-API-key override takes precedence over per-user.
                if actor.api_key is not None:
                    pol = f"api_key.{actor.api_key.id}.v1"
                    if ratelimit._config_override(pol):
                        d = ratelimit.consume(f"key:{actor.api_key.id}:v1", pol)
                        if not d.allowed:
                            await self._reject(send, d, "api key rate limit exceeded")
                            return
                # Per-user bucket.
                d = ratelimit.consume(f"user:{actor.user.id}:v1", "user.v1")
                if not d.allowed:
                    await self._reject(send, d, "rate limit exceeded")
                    return

        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(send, decision: ratelimit.Decision, msg: str) -> None:
        retry = max(1, int(decision.retry_after_s) + 1)
        body = json.dumps({
            "detail": msg,
            "retry_after": retry,
            "capacity": decision.capacity,
        }).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"retry-after", str(retry).encode("ascii")),
                (b"x-ratelimit-capacity", str(int(decision.capacity)).encode("ascii")),
                (b"x-ratelimit-remaining", str(max(0, int(decision.remaining))).encode("ascii")),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body, "more_body": False})


__all__ = ["RateLimitMiddleware"]
