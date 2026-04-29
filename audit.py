"""Per-request audit logging.

Writes a row to ``request_audit`` for every protected request (``/v1/*``,
``/auth/*``, ``/admin/*``, ``/rag/*``, ``/events``) with: timestamp, user id,
api key id, client IP, method, path, status, byte counts, duration. Used by
the admin audit pane and by per-key IP/usage tracking.

Implemented as a pure ASGI middleware so it does not break SSE streaming
(``BaseHTTPMiddleware`` would buffer the body). Must be installed *inside*
:class:`~provider.auth_deps.AuthMiddleware` so ``request.state.actor`` is
already populated.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from . import db

log = logging.getLogger("provider.audit")


# Paths that are recorded. Anything else (UI assets, /health) is skipped to
# keep the table small.
_AUDIT_PREFIXES: tuple[str, ...] = (
    "/v1/",
    "/auth/",
    "/admin/",
    "/rag/",
    "/events",
)
# Loud spam we never want in the table.
_AUDIT_SKIP: tuple[str, ...] = (
    "/auth/me-anonymous",
)
# Default rolling window: keep N most-recent rows. Override with env.
_DEFAULT_RETENTION = int(os.environ.get("PROVIDER_AUDIT_RETENTION", "50000"))


def _should_audit(path: str) -> bool:
    if path in _AUDIT_SKIP:
        return False
    return any(path.startswith(p) for p in _AUDIT_PREFIXES)


def write_row(
    *,
    user_id: int | None,
    api_key_id: int | None,
    ip: str | None,
    method: str,
    path: str,
    status: int,
    bytes_in: int,
    bytes_out: int,
    duration_ms: int,
) -> None:
    """Insert one audit row. Best-effort — never raises into the request path."""
    try:
        db.execute(
            """
            INSERT INTO request_audit (ts, user_id, api_key_id, ip, method, path,
                                       status, bytes_in, bytes_out, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (db.now_ts(), user_id, api_key_id, ip, method, path,
             status, bytes_in, bytes_out, duration_ms),
        )
    except Exception:  # noqa: BLE001
        log.exception("audit insert failed for %s %s", method, path)


def trim_audit(retain: int = _DEFAULT_RETENTION) -> int:
    """Trim ``request_audit`` to ``retain`` newest rows. Returns rows deleted."""
    row = db.fetchone("SELECT COUNT(*) AS n FROM request_audit")
    n = int(row["n"]) if row else 0
    if n <= retain:
        return 0
    cutoff = db.fetchone(
        "SELECT id FROM request_audit ORDER BY id DESC LIMIT 1 OFFSET ?",
        (retain,),
    )
    if not cutoff:
        return 0
    cur = db.execute("DELETE FROM request_audit WHERE id <= ?", (cutoff["id"],))
    return cur.rowcount or 0


# ----------------------------------------------------------- middleware

class AuditMiddleware:
    """ASGI middleware emitting one ``request_audit`` row per request.

    Runs *inside* :class:`AuthMiddleware`, so ``request.state.actor`` is
    populated before we read it. Body byte counts are observed via the ASGI
    ``http.request`` / ``http.response.body`` events without buffering.
    """

    # Trim every N inserts. Cheap counter, no thread-safety needed beyond
    # eventual consistency.
    _trim_every = 1000

    def __init__(self, app):
        self.app = app
        self._counter = 0

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "") or ""
        method = scope.get("method", "") or ""
        if not _should_audit(path):
            await self.app(scope, receive, send)
            return

        t0 = time.time()
        bytes_in = 0
        bytes_out = 0
        status_code = 0

        async def recv_wrapper():
            nonlocal bytes_in
            msg = await receive()
            if msg.get("type") == "http.request":
                body = msg.get("body") or b""
                bytes_in += len(body)
            return msg

        async def send_wrapper(msg):
            nonlocal bytes_out, status_code
            t = msg.get("type")
            if t == "http.response.start":
                status_code = int(msg.get("status", 0))
            elif t == "http.response.body":
                body = msg.get("body") or b""
                bytes_out += len(body)
            await send(msg)

        try:
            await self.app(scope, recv_wrapper, send_wrapper)
        finally:
            duration_ms = int((time.time() - t0) * 1000)
            # AuthMiddleware sets ``request.state.actor``. Depending on
            # Starlette version, ``scope["state"]`` is either a dict or a
            # ``State`` object — handle both.
            state = scope.get("state")
            if isinstance(state, dict):
                actor = state.get("actor")
            else:
                actor = getattr(state, "actor", None) if state is not None else None

            user_id = getattr(getattr(actor, "user", None), "id", None)
            api_key_id = getattr(getattr(actor, "api_key", None), "id", None)
            ip = _client_ip_from_scope(scope)

            write_row(
                user_id=user_id,
                api_key_id=api_key_id,
                ip=ip,
                method=method,
                path=path,
                status=status_code,
                bytes_in=bytes_in,
                bytes_out=bytes_out,
                duration_ms=duration_ms,
            )

            self._counter += 1
            if self._counter % self._trim_every == 0:
                try:
                    trim_audit()
                except Exception:  # noqa: BLE001
                    log.exception("audit trim failed")


def _client_ip_from_scope(scope) -> str | None:
    headers = dict(scope.get("headers") or [])
    if os.environ.get("PROVIDER_TRUST_PROXY", "1") == "1":
        xff = headers.get(b"x-forwarded-for")
        if xff:
            first = xff.decode("latin-1").split(",")[0].strip()
            if first:
                return first
    client = scope.get("client")
    if client and len(client) >= 1:
        return client[0]
    return None


__all__ = ["AuditMiddleware", "write_row", "trim_audit"]
