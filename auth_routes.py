"""HTTP routes for authentication, session, and API-key management.

Mounted at ``/auth`` from :mod:`provider.gateway`. The web UI uses session
cookies; OpenAI-compatible clients use ``Authorization: Bearer sk-prov-…``.

Endpoints:

* ``POST /auth/login``                    — username + password, returns
  either ``{ok: true}`` (session set) or ``{totp_required: true}``.
* ``POST /auth/login-totp``               — finish a 2-factor login.
* ``POST /auth/logout``                   — destroy current session.
* ``GET  /auth/me``                       — current user info.
* ``POST /auth/change-password``
* ``GET  /auth/keys``                     — list current user's API keys.
* ``POST /auth/keys``                     — issue a new key (plaintext returned **once**).
* ``PATCH /auth/keys/{id}``               — update IP allowlist / name.
* ``DELETE /auth/keys/{id}``              — revoke.
* ``POST /auth/totp/begin``               — start enrollment (returns otpauth URL + b32 secret).
* ``POST /auth/totp/finish``              — verify a code, mark TOTP enabled.
* ``POST /auth/totp/disable``             — disable + drop secret + recovery codes.
* ``POST /auth/recovery-codes``           — (re)issue recovery codes.
* ``POST /auth/bootstrap``                — create the first admin (only when zero users).
* ``GET/POST/DELETE /auth/users``         — admin-only user management.
* ``GET  /auth/sessions``                 — admin: list active sessions.
* ``GET  /auth/admin/keys``               — admin: list all API keys (masked).
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from . import auth as authsvc
from . import db
from .auth_deps import (
    SESSION_COOKIE,
    Actor,
    _client_ip,                   # type: ignore[attr-defined]
    get_actor,
    get_actor_optional,
    require_admin,
)

log = logging.getLogger("provider.auth_routes")

router = APIRouter(prefix="/auth", tags=["auth"])


def _cookie_kwargs(request: Request) -> dict[str, Any]:
    secure = request.url.scheme == "https" or os.environ.get("PROVIDER_FORCE_SECURE_COOKIE") == "1"
    return {
        "httponly": True,
        "secure": secure,
        "samesite": "lax",
        "path": "/",
    }


# --------------------------------------------------------------- payloads

class LoginIn(BaseModel):
    username: str
    password: str
    totp: Optional[str] = None


class TotpFinishIn(BaseModel):
    code: str


class ChangePwIn(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)


class ApiKeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    ip_allowlist: Optional[str] = None
    expires_at: Optional[int] = None


class ApiKeyPatchIn(BaseModel):
    name: Optional[str] = Field(default=None, max_length=120)
    ip_allowlist: Optional[str] = None  # empty string clears it


class BootstrapIn(BaseModel):
    username: str
    password: str = Field(min_length=8)
    email: Optional[str] = None


class CreateUserIn(BaseModel):
    username: str
    password: str = Field(min_length=8)
    email: Optional[str] = None
    role: str = "user"


# ----------------------------------------------------------- helpers

def _user_dto(u: authsvc.User) -> dict[str, Any]:
    return {
        "id": u.id,
        "username": u.username,
        "email": u.email,
        "role": u.role,
        "is_active": u.is_active,
        "totp_enabled": u.totp_enabled,
        "oidc_subject": u.oidc_subject,
    }


def _key_dto(k: authsvc.ApiKey) -> dict[str, Any]:
    return {
        "id": k.id,
        "name": k.name,
        "masked": k.masked,
        "ip_allowlist": k.ip_allowlist,
        "created_at": k.created_at,
        "last_used_at": k.last_used_at,
        "last_used_ip": k.last_used_ip,
        "expires_at": k.expires_at,
        "revoked": bool(k.revoked_at),
    }


# --------------------------------------------------------- session auth

@router.post("/login")
async def login(payload: LoginIn, request: Request, response: Response) -> dict[str, Any]:
    user = await asyncio.to_thread(authsvc.authenticate, payload.username, payload.password)
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="invalid credentials")
    if user.totp_enabled:
        if not payload.totp:
            return {"ok": False, "totp_required": True}
        ok = await asyncio.to_thread(authsvc.verify_totp, user.id, payload.totp)
        if not ok:
            raise HTTPException(status_code=401, detail="invalid 2FA code")
    ip = _client_ip(request)
    ua = request.headers.get("user-agent")
    sess = await asyncio.to_thread(authsvc.create_session, user.id, ip=ip, user_agent=ua)
    await asyncio.to_thread(authsvc.record_login, user.id, ip)
    response.set_cookie(
        SESSION_COOKIE, sess.id, max_age=authsvc.SESSION_TTL_S, **_cookie_kwargs(request)
    )
    return {"ok": True, "user": _user_dto(user)}


@router.post("/logout")
async def logout(request: Request, response: Response, actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    if actor.session_id:
        await asyncio.to_thread(authsvc.delete_session, actor.session_id)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/me")
async def me(actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    return {"user": _user_dto(actor.user), "via": actor.via}


@router.get("/me-anonymous")
async def me_anonymous(actor: Optional[Actor] = Depends(get_actor_optional)) -> dict[str, Any]:
    """Public probe used by the login page to detect bootstrap mode."""
    bootstrap = False
    try:
        row = db.fetchone("SELECT COUNT(*) AS n FROM users")
        bootstrap = bool(row) and row["n"] == 0
    except Exception:  # noqa: BLE001
        pass
    return {
        "authenticated": actor is not None,
        "user": _user_dto(actor.user) if actor else None,
        "bootstrap": bootstrap,
    }


@router.post("/change-password")
async def change_password(payload: ChangePwIn, actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    ok = await asyncio.to_thread(authsvc.authenticate, actor.user.username, payload.current_password)
    if ok is None:
        raise HTTPException(status_code=401, detail="current password incorrect")
    await asyncio.to_thread(authsvc.set_password, actor.user.id, payload.new_password)
    return {"ok": True}


# --------------------------------------------------------- API keys

@router.get("/keys")
async def list_my_keys(actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    keys = await asyncio.to_thread(authsvc.list_api_keys, actor.user.id)
    return {"keys": [_key_dto(k) for k in keys]}


@router.post("/keys")
async def create_my_key(payload: ApiKeyIn, actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    plain, k = await asyncio.to_thread(
        authsvc.create_api_key,
        actor.user.id,
        payload.name,
        ip_allowlist=payload.ip_allowlist,
        expires_at=payload.expires_at,
    )
    return {"key": _key_dto(k), "plaintext": plain}


@router.patch("/keys/{key_id}")
async def patch_my_key(key_id: int, payload: ApiKeyPatchIn, actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    keys = await asyncio.to_thread(authsvc.list_api_keys, actor.user.id)
    if not any(k.id == key_id for k in keys):
        raise HTTPException(status_code=404, detail="key not found")
    if payload.ip_allowlist is not None:
        await asyncio.to_thread(
            authsvc.update_api_key_ip_allowlist,
            key_id,
            payload.ip_allowlist or None,
        )
    if payload.name:
        db.execute("UPDATE api_keys SET name=? WHERE id=?", (payload.name, key_id))
    return {"ok": True}


@router.delete("/keys/{key_id}")
async def delete_my_key(key_id: int, actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    revoked = await asyncio.to_thread(authsvc.revoke_api_key, key_id, user_id=actor.user.id)
    if not revoked:
        raise HTTPException(status_code=404, detail="key not found or already revoked")
    return {"ok": True}


# --------------------------------------------------------------- TOTP

@router.post("/totp/begin")
async def totp_begin(actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    url, secret = await asyncio.to_thread(authsvc.begin_totp_enrollment, actor.user.id)
    return {"otpauth_url": url, "secret_base32": secret}


@router.post("/totp/finish")
async def totp_finish(payload: TotpFinishIn, actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    ok = await asyncio.to_thread(authsvc.finish_totp_enrollment, actor.user.id, payload.code)
    if not ok:
        raise HTTPException(status_code=400, detail="invalid TOTP code")
    return {"ok": True}


@router.post("/totp/disable")
async def totp_disable(payload: ChangePwIn, actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    """Disabling 2FA requires the password as a re-auth step."""
    if await asyncio.to_thread(authsvc.authenticate, actor.user.username, payload.current_password) is None:
        raise HTTPException(status_code=401, detail="password incorrect")
    await asyncio.to_thread(authsvc.disable_totp, actor.user.id)
    return {"ok": True}


@router.post("/recovery-codes")
async def issue_recovery(actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    codes = await asyncio.to_thread(authsvc.issue_recovery_codes, actor.user.id)
    return {"codes": codes}


# ----------------------------------------------------------- bootstrap

@router.post("/bootstrap")
async def bootstrap(payload: BootstrapIn, request: Request, response: Response) -> dict[str, Any]:
    """Create the first admin account. Refuses if any user already exists."""
    row = await asyncio.to_thread(db.fetchone, "SELECT COUNT(*) AS n FROM users")
    if row and row["n"] > 0:
        raise HTTPException(status_code=403, detail="bootstrap unavailable: users already exist")
    user = await asyncio.to_thread(
        authsvc.create_user,
        payload.username,
        payload.password,
        email=payload.email,
        role="admin",
    )
    ip = _client_ip(request)
    sess = await asyncio.to_thread(authsvc.create_session, user.id, ip=ip, user_agent=request.headers.get("user-agent"))
    await asyncio.to_thread(authsvc.record_login, user.id, ip)
    response.set_cookie(SESSION_COOKIE, sess.id, max_age=authsvc.SESSION_TTL_S, **_cookie_kwargs(request))
    log.info("bootstrap admin created: %s", user.username)
    return {"ok": True, "user": _user_dto(user)}


# ----------------------------------------------------------- admin: users

@router.get("/users")
async def admin_list_users(_admin: Actor = Depends(require_admin)) -> dict[str, Any]:
    users = await asyncio.to_thread(authsvc.list_users)
    return {"users": [_user_dto(u) for u in users]}


@router.post("/users")
async def admin_create_user(payload: CreateUserIn, _admin: Actor = Depends(require_admin)) -> dict[str, Any]:
    if payload.role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="role must be 'user' or 'admin'")
    if await asyncio.to_thread(authsvc.get_user_by_username, payload.username):
        raise HTTPException(status_code=409, detail="username already taken")
    user = await asyncio.to_thread(
        authsvc.create_user,
        payload.username,
        payload.password,
        email=payload.email,
        role=payload.role,
    )
    return {"user": _user_dto(user)}


@router.patch("/users/{user_id}")
async def admin_update_user(user_id: int, body: dict = Body(...), _admin: Actor = Depends(require_admin)) -> dict[str, Any]:
    target = await asyncio.to_thread(authsvc.get_user_by_id, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="user not found")

    if "role" in body:
        new_role = body["role"]
        if new_role not in ("user", "admin"):
            raise HTTPException(status_code=400, detail="role must be 'user' or 'admin'")
        # Don't allow demoting the last admin.
        if target.role == "admin" and new_role != "admin":
            count = await asyncio.to_thread(authsvc.admin_count)
            if count <= 1:
                raise HTTPException(status_code=400, detail="cannot demote the last admin")
        await asyncio.to_thread(authsvc.set_role, user_id, new_role)

    if "is_active" in body:
        active = bool(body["is_active"])
        if not active and target.is_admin:
            count = await asyncio.to_thread(authsvc.admin_count)
            if count <= 1:
                raise HTTPException(status_code=400, detail="cannot disable the last active admin")
        await asyncio.to_thread(authsvc.set_active, user_id, active)

    if "password" in body and body["password"]:
        if len(body["password"]) < 8:
            raise HTTPException(status_code=400, detail="password too short")
        await asyncio.to_thread(authsvc.set_password, user_id, body["password"])

    fresh = await asyncio.to_thread(authsvc.get_user_by_id, user_id)
    return {"user": _user_dto(fresh) if fresh else None}


# ----------------------------------------------------- admin: api keys

@router.get("/admin/keys")
async def admin_list_keys(_admin: Actor = Depends(require_admin)) -> dict[str, Any]:
    keys = await asyncio.to_thread(authsvc.list_all_api_keys)
    rows = []
    for k in keys:
        d = _key_dto(k)
        d["user_id"] = k.user_id
        rows.append(d)
    return {"keys": rows}


@router.delete("/admin/keys/{key_id}")
async def admin_revoke_key(key_id: int, _admin: Actor = Depends(require_admin)) -> dict[str, Any]:
    revoked = await asyncio.to_thread(authsvc.revoke_api_key, key_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="key not found or already revoked")
    return {"ok": True}


# ----------------------------------------------------- admin: sessions

@router.get("/sessions")
async def admin_list_sessions(_admin: Actor = Depends(require_admin)) -> dict[str, Any]:
    rows = await asyncio.to_thread(
        db.fetchall,
        "SELECT s.id, s.user_id, u.username, s.created_at, s.expires_at, s.ip, s.user_agent "
        "FROM sessions s JOIN users u ON u.id = s.user_id "
        "WHERE s.expires_at >= ? ORDER BY s.created_at DESC",
        (db.now_ts(),),
    )
    return {
        "sessions": [
            {
                "id_prefix": r["id"][:12] + "…",
                "user_id": r["user_id"],
                "username": r["username"],
                "created_at": r["created_at"],
                "expires_at": r["expires_at"],
                "ip": r["ip"],
                "user_agent": r["user_agent"],
            }
            for r in rows
        ]
    }


@router.delete("/sessions/{sid_prefix}")
async def admin_kill_session(sid_prefix: str, _admin: Actor = Depends(require_admin)) -> dict[str, Any]:
    if len(sid_prefix) < 8:
        raise HTTPException(status_code=400, detail="prefix too short")
    cur = db.execute("DELETE FROM sessions WHERE id LIKE ?", (sid_prefix + "%",))
    return {"deleted": cur.rowcount}


# ----------------------------------------------------- admin: audit log

@router.get("/admin/audit")
async def admin_audit(
    _admin: Actor = Depends(require_admin),
    limit: int = 200,
    offset: int = 0,
    user_id: Optional[int] = None,
    api_key_id: Optional[int] = None,
    method: Optional[str] = None,
    path_prefix: Optional[str] = None,
    status_min: Optional[int] = None,
    status_max: Optional[int] = None,
    since: Optional[int] = None,
    ip: Optional[str] = None,
) -> dict[str, Any]:
    """Paginated read of ``request_audit`` with optional filters.

    All filters are AND'd. ``limit`` is capped at 1000.
    """
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))

    where: list[str] = []
    args: list[Any] = []
    if user_id is not None:
        where.append("a.user_id = ?"); args.append(user_id)
    if api_key_id is not None:
        where.append("a.api_key_id = ?"); args.append(api_key_id)
    if method:
        where.append("a.method = ?"); args.append(method.upper())
    if path_prefix:
        where.append("a.path LIKE ?"); args.append(path_prefix + "%")
    if status_min is not None:
        where.append("a.status >= ?"); args.append(int(status_min))
    if status_max is not None:
        where.append("a.status <= ?"); args.append(int(status_max))
    if since is not None:
        where.append("a.ts >= ?"); args.append(int(since))
    if ip:
        where.append("a.ip = ?"); args.append(ip)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    total_row = await asyncio.to_thread(
        db.fetchone,
        f"SELECT COUNT(*) AS n FROM request_audit a {where_sql}",
        tuple(args),
    )
    rows = await asyncio.to_thread(
        db.fetchall,
        f"""
        SELECT a.id, a.ts, a.user_id, a.api_key_id, a.ip, a.method, a.path,
               a.status, a.bytes_in, a.bytes_out, a.duration_ms,
               u.username AS username,
               k.name     AS key_name,
               k.key_prefix AS key_prefix
        FROM request_audit a
        LEFT JOIN users    u ON u.id = a.user_id
        LEFT JOIN api_keys k ON k.id = a.api_key_id
        {where_sql}
        ORDER BY a.id DESC
        LIMIT ? OFFSET ?
        """,
        tuple(args) + (limit, offset),
    )
    return {
        "total": int(total_row["n"]) if total_row else 0,
        "limit": limit,
        "offset": offset,
        "rows": [
            {
                "id": r["id"],
                "ts": r["ts"],
                "user_id": r["user_id"],
                "username": r["username"],
                "api_key_id": r["api_key_id"],
                "key_name": r["key_name"],
                "key_prefix": r["key_prefix"],
                "ip": r["ip"],
                "method": r["method"],
                "path": r["path"],
                "status": r["status"],
                "bytes_in": r["bytes_in"],
                "bytes_out": r["bytes_out"],
                "duration_ms": r["duration_ms"],
            }
            for r in rows
        ],
    }


@router.get("/admin/audit/summary")
async def admin_audit_summary(
    _admin: Actor = Depends(require_admin),
    since: Optional[int] = None,
) -> dict[str, Any]:
    """Lightweight aggregates for the admin dashboard."""
    where_sql = ""
    args: tuple[Any, ...] = ()
    if since is not None:
        where_sql = "WHERE ts >= ?"
        args = (int(since),)

    by_user = await asyncio.to_thread(
        db.fetchall,
        f"""
        SELECT a.user_id, u.username, COUNT(*) AS n,
               SUM(a.bytes_in)  AS bytes_in,
               SUM(a.bytes_out) AS bytes_out,
               AVG(a.duration_ms) AS avg_ms
        FROM request_audit a LEFT JOIN users u ON u.id = a.user_id
        {where_sql}
        GROUP BY a.user_id ORDER BY n DESC LIMIT 50
        """,
        args,
    )
    by_status = await asyncio.to_thread(
        db.fetchall,
        f"SELECT status, COUNT(*) AS n FROM request_audit {where_sql} GROUP BY status ORDER BY status",
        args,
    )
    return {
        "by_user":   [dict(r) for r in by_user],
        "by_status": [dict(r) for r in by_status],
    }


__all__ = ["router"]
