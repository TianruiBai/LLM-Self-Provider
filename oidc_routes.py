"""HTTP routes for OIDC / OAuth2 sign-in (A8).

Mounted under ``/auth/oidc/``. Public (no session required) since the
whole purpose is to *establish* a session.

Endpoints:

* ``GET  /auth/oidc/providers``               — list configured providers
  (no secrets returned).
* ``GET  /auth/oidc/{name}/start?next=…``     — generate state + PKCE,
  redirect to the provider's authorize URL.
* ``GET  /auth/oidc/{name}/callback``         — exchange ``code`` for an
  access token, fetch user-info, and either log the matching user in or
  create a brand-new ``oidc_subject``-stamped user.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from . import auth as authsvc
from . import oidc
from .auth_deps import SESSION_COOKIE, _client_ip  # type: ignore[attr-defined]

log = logging.getLogger("provider.oidc_routes")

router = APIRouter(prefix="/auth/oidc", tags=["oidc"])


# --------------------------------------------------------------- helpers


def _redirect_uri_for(request: Request, name: str) -> str:
    """Compute the absolute callback URL the provider must whitelist.

    Honors ``PROVIDER_PUBLIC_BASE_URL`` when set (typical when behind a
    reverse proxy), falling back to the request's own scheme/host.
    """
    base = os.environ.get("PROVIDER_PUBLIC_BASE_URL")
    if base:
        base = base.rstrip("/")
        return f"{base}/auth/oidc/{name}/callback"
    return str(request.url_for("oidc_callback", name=name))


def _safe_next(target: str | None) -> str:
    """Restrict ``next`` to a same-origin path (no host change)."""
    if not target:
        return "/ui/"
    if target.startswith("/") and not target.startswith("//"):
        return target
    return "/ui/"


def _cookie_kwargs(request: Request) -> dict[str, Any]:
    secure = request.url.scheme == "https" or os.environ.get("PROVIDER_FORCE_SECURE_COOKIE") == "1"
    return {
        "httponly": True,
        "secure": secure,
        "samesite": "lax",
        "path": "/",
    }


# --------------------------------------------------------------- routes


@router.get("/providers")
async def list_providers() -> dict[str, Any]:
    providers = oidc.load_providers()
    return {
        "providers": [
            {
                "name": p.name,
                "label": p.display_name(),
                "scope": p.scope,
            }
            for p in providers.values()
        ]
    }


@router.get("/{name}/start")
async def oidc_start(name: str, request: Request, next: str | None = None) -> RedirectResponse:
    provider = oidc.get_provider(name)
    if provider is None:
        raise HTTPException(status_code=404, detail="unknown OIDC provider")
    redirect_to = _safe_next(next)
    state, _verifier, challenge = await asyncio.to_thread(
        oidc.create_state, name, redirect_to, with_pkce=provider.use_pkce
    )
    redirect_uri = _redirect_uri_for(request, name)
    auth_url = oidc.authorize_url_for(
        provider, state=state, redirect_uri=redirect_uri, code_challenge=challenge
    )
    log.info("OIDC start name=%s redirect_uri=%s", name, redirect_uri)
    return RedirectResponse(url=auth_url, status_code=302)


@router.get("/{name}/callback", name="oidc_callback")
async def oidc_callback(
    name: str,
    request: Request,
    response: Response,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> Response:
    if error:
        raise HTTPException(status_code=400, detail=f"oauth error: {error}")
    if not code or not state:
        raise HTTPException(status_code=400, detail="missing code/state")

    provider = oidc.get_provider(name)
    if provider is None:
        raise HTTPException(status_code=404, detail="unknown OIDC provider")

    state_row = await asyncio.to_thread(oidc.consume_state, state)
    if state_row is None:
        raise HTTPException(status_code=400, detail="invalid or expired state")
    if state_row["provider"] != name:
        raise HTTPException(status_code=400, detail="state/provider mismatch")

    redirect_uri = _redirect_uri_for(request, name)
    try:
        info = await oidc.exchange_and_fetch(
            provider,
            code=code,
            redirect_uri=redirect_uri,
            code_verifier=state_row["code_verifier"],
        )
    except Exception as e:  # noqa: BLE001
        log.warning("OIDC callback exchange failed (%s): %s", name, e)
        raise HTTPException(status_code=502, detail=f"oauth exchange failed: {e}") from e

    subject = info["subject"]
    user = await asyncio.to_thread(authsvc.get_user_by_oidc, subject)
    if user is None:
        # First sign-in: create a fresh user with no password set.
        username = oidc.derive_username(info["raw"])
        # Ensure uniqueness — append a numeric suffix if needed.
        base = username
        i = 1
        while await asyncio.to_thread(authsvc.get_user_by_username, username):
            i += 1
            username = f"{base}{i}"
        try:
            user = await asyncio.to_thread(
                authsvc.create_user,
                username,
                None,
                email=info.get("email"),
                role="user",
                oidc_subject=subject,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("OIDC user creation failed: %s", e)
            raise HTTPException(status_code=500, detail="could not create user") from e
        log.info("OIDC: created user %r (subject=%s)", username, subject)

    if not user.is_active:
        raise HTTPException(status_code=403, detail="user is disabled")

    ip = _client_ip(request)
    ua = request.headers.get("user-agent")
    sess = await asyncio.to_thread(authsvc.create_session, user.id, ip=ip, user_agent=ua)
    await asyncio.to_thread(authsvc.record_login, user.id, ip)

    redirect_to = _safe_next(state_row.get("redirect_to"))
    resp = RedirectResponse(url=redirect_to, status_code=302)
    resp.set_cookie(
        SESSION_COOKIE, sess.id, max_age=authsvc.SESSION_TTL_S, **_cookie_kwargs(request)
    )
    return resp


__all__ = ["router"]
