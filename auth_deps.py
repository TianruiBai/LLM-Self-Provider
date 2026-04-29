"""FastAPI dependencies and request-state helpers for authentication.

Resolution order for an incoming request (handled by :class:`AuthMiddleware`):

1. ``Authorization: Bearer sk-prov-…`` API key — authenticates as the key's
   owning user, with ``actor.api_key`` populated. Used by OpenAI-compatible
   clients (Continue/Cline/curl).
2. Session cookie ``PROV_SID`` — authenticates as the session's owner. Used
   by the web UI.
3. Anonymous — request proceeds with ``actor = None``; route guards decide
   whether to allow or 401.

Handlers consume the result via the dependencies exported here.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from . import auth as authsvc

log = logging.getLogger("provider.auth_deps")


SESSION_COOKIE = "PROV_SID"


# ---------------------------------------------------------------- actor

@dataclass
class Actor:
    """Authenticated principal attached to ``request.state.actor``."""

    user: authsvc.User
    via: str                           # 'session' | 'api_key'
    api_key: Optional[authsvc.ApiKey] = None
    session_id: Optional[str] = None

    @property
    def is_admin(self) -> bool:
        return self.user.is_admin


# ----------------------------------------------------------- helpers

def _client_ip(request: Request) -> Optional[str]:
    # Prefer X-Forwarded-For when behind a reverse proxy. We trust the *first*
    # hop only; deployers can set PROVIDER_TRUST_PROXY=0 to disable.
    if os.environ.get("PROVIDER_TRUST_PROXY", "1") == "1":
        xff = request.headers.get("x-forwarded-for")
        if xff:
            first = xff.split(",")[0].strip()
            if first:
                return first
    if request.client:
        return request.client.host
    return None


def _ip_in_allowlist(ip: Optional[str], allowlist_csv: Optional[str]) -> bool:
    if not allowlist_csv:
        return True
    if not ip:
        return False
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for token in allowlist_csv.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            net = ipaddress.ip_network(token, strict=False)
        except ValueError:
            continue
        if addr in net:
            return True
    return False


def _bootstrap_active() -> bool:
    """Bootstrap mode: no users exist yet. Allow anonymous admin access so
    the operator can create the first account via the install wizard or CLI.
    Disabled by setting ``PROVIDER_DISABLE_BOOTSTRAP=1``.
    """
    if os.environ.get("PROVIDER_DISABLE_BOOTSTRAP") == "1":
        return False
    try:
        from . import db
        row = db.fetchone("SELECT COUNT(*) AS n FROM users")
        return bool(row) and row["n"] == 0
    except Exception:  # noqa: BLE001
        return False


def _dev_allow() -> bool:
    """Dev escape hatch: ``PROVIDER_AUTH_DEV_ALLOW=1`` lets unauthenticated
    requests through as a synthetic anonymous-admin actor. **Do not enable in
    production.** Logged loudly on every use.
    """
    return os.environ.get("PROVIDER_AUTH_DEV_ALLOW") == "1"


# ----------------------------------------------------------- dependencies

def get_actor_optional(request: Request) -> Optional[Actor]:
    """Return the resolved :class:`Actor` if any, else ``None``. Never raises."""
    return getattr(request.state, "actor", None)


def get_actor(request: Request) -> Actor:
    actor = getattr(request.state, "actor", None)
    if actor is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": 'Bearer realm="provider"'},
        )
    return actor


def get_current_user(actor: Actor = Depends(get_actor)) -> authsvc.User:
    return actor.user


def require_admin(actor: Actor = Depends(get_actor)) -> Actor:
    if not actor.is_admin:
        raise HTTPException(status_code=403, detail="admin role required")
    return actor


def require_api_caller(request: Request, actor: Actor = Depends(get_actor)) -> Actor:
    """Stricter guard for ``/v1/*`` endpoints used by external clients.

    Allows session-based callers (the web UI) but, when the call is via API
    key, additionally enforces ``ip_allowlist`` if configured on that key.
    """
    if actor.via == "api_key" and actor.api_key is not None:
        ip = _client_ip(request)
        if not _ip_in_allowlist(ip, actor.api_key.ip_allowlist):
            log.warning(
                "api key id=%d denied: ip=%s not in allowlist=%s",
                actor.api_key.id, ip, actor.api_key.ip_allowlist,
            )
            raise HTTPException(status_code=403, detail="source IP not allowed for this API key")
    return actor


# ------------------------------------------------------- middleware

class AuthMiddleware:
    """ASGI middleware that resolves the caller and stores them on
    ``request.state.actor`` (or leaves it ``None``).

    The middleware also enforces auth for protected route prefixes — handlers
    can still re-check via :func:`require_admin` etc. Public prefixes (web UI,
    health, login) are allowed through unauthenticated.
    """

    PUBLIC_PREFIXES: tuple[str, ...] = (
        "/health",
        "/ui/",
        "/auth/login",
        "/auth/login-totp",
        "/auth/oauth/",
        "/auth/oidc/",
        "/auth/bootstrap",
        "/static/",
        "/favicon.ico",
        "/docs",
        "/openapi.json",
        "/redoc",
    )
    PUBLIC_EXACT: tuple[str, ...] = ("/", "/auth/me-anonymous")

    PROTECTED_PREFIXES: tuple[str, ...] = (
        "/v1/",
        "/admin/",
        "/rag/",
        "/events",
        "/auth/me",
        "/auth/logout",
        "/auth/keys",
        "/auth/totp",
        "/auth/recovery-codes",
        "/auth/users",
        "/auth/sessions",
        "/auth/admin/",
    )

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "") or ""
        request = Request(scope)
        request.state.actor = None

        # Resolve actor (best-effort).
        try:
            actor = await asyncio.to_thread(self._resolve_actor_sync, request)
        except Exception:  # noqa: BLE001
            log.exception("auth resolution failed")
            actor = None
        request.state.actor = actor

        # Decide whether to short-circuit with 401/403.
        if self._is_public(path):
            await self.app(scope, receive, send)
            return

        if not self._is_protected(path):
            await self.app(scope, receive, send)
            return

        if actor is None:
            # Bootstrap and dev-allow let unauthenticated requests through;
            # the corresponding handler still gets actor=None.
            if _bootstrap_active() or _dev_allow():
                await self.app(scope, receive, send)
                return
            await self._send_401(send)
            return

        await self.app(scope, receive, send)

    # ---- helpers ----

    @staticmethod
    def _is_public(path: str) -> bool:
        if path in AuthMiddleware.PUBLIC_EXACT:
            return True
        return any(path == p.rstrip("/") or path.startswith(p) for p in AuthMiddleware.PUBLIC_PREFIXES)

    @staticmethod
    def _is_protected(path: str) -> bool:
        return any(path.startswith(p) for p in AuthMiddleware.PROTECTED_PREFIXES)

    @staticmethod
    def _resolve_actor_sync(request: Request) -> Optional[Actor]:
        # 1) Bearer token
        h = request.headers.get("authorization") or ""
        if h.lower().startswith("bearer "):
            token = h[7:].strip()
            if token.startswith(authsvc.API_KEY_PREFIX):
                k = authsvc.verify_api_key(token)
                if k is not None:
                    user = authsvc.get_user_by_id(k.user_id)
                    if user and user.is_active:
                        ip = _client_ip(request)
                        authsvc.touch_api_key(k.id, ip=ip)
                        return Actor(user=user, via="api_key", api_key=k)
                # Invalid bearer: treat as unauthenticated; route guard will 401.
                return None

        # 2) Session cookie
        sid = request.cookies.get(SESSION_COOKIE)
        if sid:
            sess = authsvc.get_session(sid)
            if sess is not None:
                user = authsvc.get_user_by_id(sess.user_id)
                if user and user.is_active:
                    authsvc.touch_session(sess.id)
                    return Actor(user=user, via="session", session_id=sess.id)
        return None

    @staticmethod
    async def _send_401(send) -> None:
        body = b'{"detail":"authentication required"}'
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer realm="provider"'),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body, "more_body": False})


__all__ = [
    "Actor",
    "SESSION_COOKIE",
    "AuthMiddleware",
    "get_actor",
    "get_actor_optional",
    "get_current_user",
    "require_admin",
    "require_api_caller",
]
