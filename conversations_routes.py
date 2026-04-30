"""HTTP routes for synced WebUI chat history.

Mounted at ``/conversations`` from :mod:`provider.gateway`. Each conversation
row is owned by exactly one user (resolved via session cookie or API key) so
two browsers logged in as the same account see the same history.

Conversation payloads are opaque JSON blobs from the WebUI client; the server
treats them as a sync-store and only extracts a handful of fields (``title``,
``model``) for listing.

Endpoints:

* ``GET    /conversations``         — list metadata for current user.
* ``GET    /conversations/all``     — list metadata + full ``data`` payload
  (used at WebUI bootstrap to hydrate the sidebar in one round trip).
* ``GET    /conversations/{id}``    — fetch one conversation (full payload).
* ``PUT    /conversations/{id}``    — upsert (full conversation body).
* ``DELETE /conversations/{id}``    — remove.

Private chats stay client-only: the WebUI never PUTs them here.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from . import db
from .auth_deps import Actor, get_actor

log = logging.getLogger("provider.conversations_routes")

router = APIRouter(prefix="/conversations", tags=["conversations"])


# ----------------------------------------------------- payloads


class ConversationIn(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    title: Optional[str] = ""
    model: Optional[str] = ""
    created_at: Optional[int] = None  # client epoch ms
    updated_at: Optional[int] = None
    data: dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------- helpers


_MAX_BLOB_BYTES = 4 * 1024 * 1024  # 4 MiB safety cap per conversation


def _row_to_meta(row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "model": row["model"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_full(row) -> dict[str, Any]:
    meta = _row_to_meta(row)
    try:
        meta["data"] = json.loads(row["data"]) if row["data"] else {}
    except json.JSONDecodeError:
        meta["data"] = {}
    return meta


def _list_for_user_sync(user_id: int, *, with_data: bool) -> list[dict[str, Any]]:
    cols = "id, title, model, created_at, updated_at"
    if with_data:
        cols += ", data"
    rows = db.fetchall(
        f"SELECT {cols} FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,),
    )
    fn = _row_to_full if with_data else _row_to_meta
    return [fn(r) for r in rows]


def _get_one_sync(user_id: int, conv_id: str):
    return db.fetchone(
        "SELECT id, title, model, created_at, updated_at, data "
        "FROM conversations WHERE user_id = ? AND id = ?",
        (user_id, conv_id),
    )


def _upsert_sync(user_id: int, payload: ConversationIn) -> dict[str, Any]:
    blob = json.dumps(payload.data, ensure_ascii=False, separators=(",", ":"))
    if len(blob.encode("utf-8")) > _MAX_BLOB_BYTES:
        raise HTTPException(status_code=413, detail="conversation too large")
    now = db.now_ts()
    created_at = int(payload.created_at) if payload.created_at else now
    updated_at = int(payload.updated_at) if payload.updated_at else now
    title = (payload.title or "")[:200]
    model = (payload.model or "")[:200]
    db.execute(
        """
        INSERT INTO conversations (id, user_id, title, model, created_at, updated_at, data)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title       = excluded.title,
            model       = excluded.model,
            updated_at  = excluded.updated_at,
            data        = excluded.data
        WHERE conversations.user_id = excluded.user_id
        """,
        (payload.id, user_id, title, model, created_at, updated_at, blob),
    )
    # Guard against a foreign id being PUT by a different user — the WHERE
    # clause above silently no-ops in that case, so re-check ownership.
    row = _get_one_sync(user_id, payload.id)
    if row is None:
        raise HTTPException(status_code=409, detail="conversation id belongs to another user")
    return _row_to_full(row)


def _delete_sync(user_id: int, conv_id: str) -> bool:
    cur = db.execute(
        "DELETE FROM conversations WHERE user_id = ? AND id = ?",
        (user_id, conv_id),
    )
    return cur.rowcount > 0


# ----------------------------------------------------- routes


@router.get("")
async def list_conversations(actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    items = await asyncio.to_thread(_list_for_user_sync, actor.user.id, with_data=False)
    return {"items": items}


@router.get("/all")
async def list_conversations_full(actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    items = await asyncio.to_thread(_list_for_user_sync, actor.user.id, with_data=True)
    return {"items": items}


@router.get("/{conv_id}")
async def get_conversation(conv_id: str, actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    row = await asyncio.to_thread(_get_one_sync, actor.user.id, conv_id)
    if row is None:
        raise HTTPException(status_code=404, detail="not found")
    return _row_to_full(row)


@router.put("/{conv_id}")
async def upsert_conversation(
    conv_id: str,
    payload: ConversationIn,
    actor: Actor = Depends(get_actor),
) -> dict[str, Any]:
    if payload.id != conv_id:
        raise HTTPException(status_code=400, detail="id mismatch")
    return await asyncio.to_thread(_upsert_sync, actor.user.id, payload)


@router.delete("/{conv_id}")
async def delete_conversation(conv_id: str, actor: Actor = Depends(get_actor)) -> dict[str, Any]:
    removed = await asyncio.to_thread(_delete_sync, actor.user.id, conv_id)
    return {"deleted": bool(removed)}
