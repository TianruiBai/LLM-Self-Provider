"""RAG: ingest + retrieve against MongoDB Atlas Local using Qwen3-Embedding-8B.

The MongoDB Atlas Local image (`mongodb/mongodb-atlas-local`) supports the
same `$vectorSearch` aggregation stage as Atlas. We create a vector index
on the `embedding` field at startup if it doesn't exist.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
from pymongo import AsyncMongoClient
from pymongo.errors import OperationFailure

from .registry import ProviderConfig

log = logging.getLogger("provider.rag")


@dataclass
class RagHit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


# ----------------------------------------------------- scope helpers

# Allowed values for the per-chunk ``scope`` field.
SCOPE_USER = "user"
SCOPE_GLOBAL = "global"
_VALID_SCOPES = (SCOPE_USER, SCOPE_GLOBAL)


def _scope_match(
    viewer_id: int | None,
    *,
    is_admin: bool,
    scopes: list[str] | None,
) -> dict[str, Any]:
    """Build a Mongo ``$match`` clause restricting visibility per A9 rules.

    * ``viewer_id=None`` (anonymous in dev mode): treated like a regular user
      with no private docs → ``scope:'global'`` only.
    * ``is_admin=True``: sees everything; only the explicit ``scopes`` filter
      applies.
    * Regular user: sees ``scope='global'`` plus their own ``scope='user'``
      docs. Legacy docs (no ``scope`` field) are treated as global for
      backwards compatibility.
    """
    requested = [s for s in (scopes or [SCOPE_USER, SCOPE_GLOBAL]) if s in _VALID_SCOPES]
    if not requested:
        requested = [SCOPE_USER, SCOPE_GLOBAL]

    if is_admin:
        if SCOPE_USER in requested and SCOPE_GLOBAL in requested:
            return {}
        if SCOPE_USER in requested:
            return {"scope": SCOPE_USER}
        return {"$or": [{"scope": SCOPE_GLOBAL}, {"scope": {"$exists": False}}]}

    clauses: list[dict[str, Any]] = []
    if SCOPE_GLOBAL in requested:
        # Legacy chunks without a scope are visible-as-global.
        clauses.append({"$or": [{"scope": SCOPE_GLOBAL}, {"scope": {"$exists": False}}]})
    if SCOPE_USER in requested and viewer_id is not None:
        clauses.append({"scope": SCOPE_USER, "owner_id": int(viewer_id)})

    if not clauses:
        # No accessible docs.
        return {"_id": {"$exists": False}}
    if len(clauses) == 1:
        return clauses[0]
    return {"$or": clauses}


def _chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    step = max(1, chunk_chars - overlap)
    out: list[str] = []
    for i in range(0, len(text), step):
        piece = text[i : i + chunk_chars].strip()
        if piece:
            out.append(piece)
        if i + chunk_chars >= len(text):
            break
    return out


def _doc_id(source: str, idx: int, content: str) -> str:
    h = hashlib.sha1(f"{source}::{idx}::{content}".encode("utf-8")).hexdigest()
    return h[:24]


class RagService:
    def __init__(self, cfg: ProviderConfig, embedder_base_url_provider):
        """`embedder_base_url_provider` is a callable returning the embedder URL,
        which lets us defer until the embedder process has spun up."""
        self.cfg = cfg
        self._embedder_url = embedder_base_url_provider
        self._client = AsyncMongoClient(cfg.rag.mongo_uri)
        self._db = self._client[cfg.rag.database]
        self._coll = self._db[cfg.rag.collection]
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=300.0))
        self._index_ready = False

    async def startup(self) -> None:
        await self._ensure_index()

    async def shutdown(self) -> None:
        await self._http.aclose()
        await self._client.close()

    # ------------- embeddings -------------

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        provider = self._embedder_url
        base = provider() if not asyncio.iscoroutinefunction(provider) else await provider()
        if base is None:
            raise RuntimeError("Embedder is not running")
        # llama-server OpenAI endpoint
        embed_model = self.cfg.embedding_model
        model_id = embed_model.id if embed_model else "embedding"
        r = await self._http.post(
            f"{base}/v1/embeddings",
            json={"model": model_id, "input": inputs},
        )
        r.raise_for_status()
        payload = r.json()
        # OpenAI shape: {"data": [{"embedding": [...]}, ...]}
        return [d["embedding"] for d in payload["data"]]

    # ------------- index management -------------

    async def _ensure_index(self) -> None:
        if self._index_ready:
            return
        # Atlas Local requires the collection to exist before a search index
        # can be created. Create it idempotently.
        try:
            existing_colls = await self._db.list_collection_names()
            if self.cfg.rag.collection not in existing_colls:
                await self._db.create_collection(self.cfg.rag.collection)
                log.info("Created collection %s.%s", self.cfg.rag.database, self.cfg.rag.collection)
        except OperationFailure as e:
            log.warning("create_collection failed (continuing): %s", e)

        name = self.cfg.rag.vector_index
        try:
            cursor = await self._coll.list_search_indexes(name=name)
            existing = await cursor.to_list(length=1)
            if existing:
                self._index_ready = True
                return
        except OperationFailure as e:
            log.warning("list_search_indexes failed (continuing): %s", e)

        definition = {
            "name": name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": self.cfg.rag.embedding_dim,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "source"},
                    {"type": "filter", "path": "tags"},
                ]
            },
        }
        try:
            await self._coll.create_search_index(definition)
            log.info("Created vector index %r on %s.%s", name,
                     self.cfg.rag.database, self.cfg.rag.collection)
            # Wait briefly for it to come online
            for _ in range(30):
                cursor = await self._coll.list_search_indexes(name=name)
                idx = await cursor.to_list(length=1)
                if idx and idx[0].get("status") in ("READY", "STEADY"):
                    break
                await asyncio.sleep(1.0)
            self._index_ready = True
        except OperationFailure as e:
            log.error("Failed to create vector index: %s", e)
            raise

    # ------------- ingestion -------------

    async def ingest(
        self,
        documents: Iterable[dict[str, Any]],
        *,
        source: str = "api",
        tags: list[str] | None = None,
        owner_id: int | None = None,
        scope: str = SCOPE_GLOBAL,
    ) -> dict[str, Any]:
        """`documents` is an iterable of {"text": str, "metadata": {...}}.

        ``owner_id`` and ``scope`` (``'user'`` | ``'global'``) are stamped on
        every chunk for A9 KB scoping. The caller (gateway) is responsible
        for enforcing role rules (only admins may set ``scope='global'``;
        regular users get ``scope='user'`` + their own ``owner_id``).
        """
        if scope not in _VALID_SCOPES:
            raise ValueError(f"scope must be one of {_VALID_SCOPES}")
        chunks: list[dict[str, Any]] = []
        for d in documents:
            text = d.get("text", "")
            meta = d.get("metadata") or {}
            for i, piece in enumerate(_chunk_text(text, self.cfg.rag.chunk_chars, self.cfg.rag.chunk_overlap)):
                chunks.append({
                    "_id": _doc_id(source + str(meta.get("id", "")) + (f":{owner_id}" if scope == SCOPE_USER and owner_id else ""), i, piece),
                    "source": source,
                    "tags": list(tags or []),
                    "text": piece,
                    "metadata": meta,
                    "chunk_index": i,
                    "ingested_at": time.time(),
                    "scope": scope,
                    "owner_id": int(owner_id) if owner_id is not None else None,
                })

        if not chunks:
            return {"inserted": 0, "skipped": 0}

        # Embed in batches
        BATCH = 16
        for start in range(0, len(chunks), BATCH):
            batch = chunks[start : start + BATCH]
            vecs = await self.embed([c["text"] for c in batch])
            for c, v in zip(batch, vecs):
                c["embedding"] = v

        inserted = 0
        skipped = 0
        for c in chunks:
            try:
                await self._coll.replace_one({"_id": c["_id"]}, c, upsert=True)
                inserted += 1
            except OperationFailure as e:
                log.warning("upsert failed: %s", e)
                skipped += 1
        return {"inserted": inserted, "skipped": skipped, "chunks": len(chunks)}

    # ------------- retrieval -------------

    async def query(
        self,
        text: str,
        *,
        top_k: int | None = None,
        source: str | None = None,
        tags: list[str] | None = None,
        viewer_id: int | None = None,
        is_admin: bool = False,
        scopes: list[str] | None = None,
    ) -> list[RagHit]:
        if not text.strip():
            return []
        k = top_k or self.cfg.rag.default_top_k
        [vec] = await self.embed([text])

        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": self.cfg.rag.vector_index,
                    "path": "embedding",
                    "queryVector": vec,
                    "numCandidates": max(50, k * 10),
                    "limit": k,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "metadata": 1,
                    "source": 1,
                    "tags": 1,
                    "scope": 1,
                    "owner_id": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        match: dict[str, Any] = {}
        if source:
            match["source"] = source
        if tags:
            match["tags"] = {"$in": tags}
        scope_clause = _scope_match(viewer_id, is_admin=is_admin, scopes=scopes)
        if scope_clause:
            match = {**match, **scope_clause} if not match else {"$and": [match, scope_clause]}
        if match:
            pipeline.insert(1, {"$match": match})

        hits: list[RagHit] = []
        async for doc in await self._coll.aggregate(pipeline):
            hits.append(RagHit(
                id=str(doc["_id"]),
                text=doc.get("text", ""),
                score=float(doc.get("score", 0.0)),
                metadata={
                    "source": doc.get("source"),
                    "tags": doc.get("tags", []),
                    "scope": doc.get("scope") or SCOPE_GLOBAL,
                    "owner_id": doc.get("owner_id"),
                    **(doc.get("metadata") or {}),
                },
            ))
        return hits

    @staticmethod
    def format_context(hits: list[RagHit]) -> str:
        if not hits:
            return ""
        parts = []
        for i, h in enumerate(hits, 1):
            parts.append(f"[{i}] (score={h.score:.3f}) {h.text}")
        return (
            "You have access to the following retrieved context. "
            "Cite source numbers like [1], [2] when you use them.\n\n"
            + "\n\n".join(parts)
        )

    # ------------- knowledge management -------------

    async def stats(
        self,
        *,
        viewer_id: int | None = None,
        is_admin: bool = False,
        scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        scope_match = _scope_match(viewer_id, is_admin=is_admin, scopes=scopes)
        total = await self._coll.count_documents(scope_match or {})
        pipeline: list[dict[str, Any]] = []
        if scope_match:
            pipeline.append({"$match": scope_match})
        pipeline += [
            {"$group": {
                "_id": "$source",
                "chunks": {"$sum": 1},
                "docs": {"$addToSet": "$metadata.id"},
                "tags": {"$addToSet": "$tags"},
                "last": {"$max": "$ingested_at"},
            }},
            {"$sort": {"last": -1}},
        ]
        sources: list[dict[str, Any]] = []
        async for d in await self._coll.aggregate(pipeline):
            tag_set: set[str] = set()
            for arr in d.get("tags") or []:
                if isinstance(arr, list):
                    tag_set.update(arr)
            sources.append({
                "source": d["_id"] or "(none)",
                "chunks": d.get("chunks", 0),
                "documents": [x for x in (d.get("docs") or []) if x],
                "tags": sorted(tag_set),
                "last_ingested_at": d.get("last"),
            })
        return {"total_chunks": total, "sources": sources}

    async def list_documents(
        self,
        *,
        source: str | None = None,
        tag: str | None = None,
        limit: int = 200,
        viewer_id: int | None = None,
        is_admin: bool = False,
        scopes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Group chunks back into "knowledge cards" keyed by metadata.id (or source)."""
        match: dict[str, Any] = {}
        if source:
            match["source"] = source
        if tag:
            match["tags"] = tag
        scope_clause = _scope_match(viewer_id, is_admin=is_admin, scopes=scopes)
        if scope_clause:
            match = {**match, **scope_clause} if not match else {"$and": [match, scope_clause]}
        pipeline: list[dict[str, Any]] = []
        if match:
            pipeline.append({"$match": match})
        pipeline += [
            {"$sort": {"chunk_index": 1}},
            {"$group": {
                "_id": {
                    "source": "$source",
                    "doc_id": {"$ifNull": ["$metadata.id", "$_id"]},
                },
                "title": {"$first": "$metadata.title"},
                "tags": {"$first": "$tags"},
                "scope": {"$first": "$scope"},
                "owner_id": {"$first": "$owner_id"},
                "chunks": {"$sum": 1},
                "first_text": {"$first": "$text"},
                "ingested_at": {"$max": "$ingested_at"},
                "chunk_ids": {"$push": "$_id"},
            }},
            {"$sort": {"ingested_at": -1}},
            {"$limit": int(limit)},
        ]
        cards: list[dict[str, Any]] = []
        async for d in await self._coll.aggregate(pipeline):
            preview = (d.get("first_text") or "").strip()
            if len(preview) > 280:
                preview = preview[:280].rstrip() + "…"
            cards.append({
                "source": d["_id"]["source"],
                "doc_id": d["_id"]["doc_id"],
                "title": d.get("title") or d["_id"]["doc_id"],
                "tags": d.get("tags") or [],
                "scope": d.get("scope") or SCOPE_GLOBAL,
                "owner_id": d.get("owner_id"),
                "chunks": d.get("chunks", 0),
                "preview": preview,
                "ingested_at": d.get("ingested_at"),
                "chunk_ids": d.get("chunk_ids", []),
            })
        return cards

    async def delete(
        self,
        *,
        chunk_ids: list[str] | None = None,
        source: str | None = None,
        doc_id: str | None = None,
        tag: str | None = None,
        viewer_id: int | None = None,
        is_admin: bool = False,
        scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        q: dict[str, Any] = {}
        if chunk_ids:
            q["_id"] = {"$in": list(chunk_ids)}
        if source:
            q["source"] = source
        if doc_id:
            q["metadata.id"] = doc_id
        if tag:
            q["tags"] = tag
        if not q:
            raise ValueError("delete requires at least one of: chunk_ids, source, doc_id, tag")
        scope_clause = _scope_match(viewer_id, is_admin=is_admin, scopes=scopes)
        if scope_clause:
            q = {"$and": [q, scope_clause]}
        res = await self._coll.delete_many(q)
        return {"deleted": int(res.deleted_count)}

    async def get_chunks(
        self,
        chunk_ids: list[str],
        *,
        viewer_id: int | None = None,
        is_admin: bool = False,
        scopes: list[str] | None = None,
    ) -> list[dict]:
        """Fetch full chunk records by id, preserving the requested order."""
        if not chunk_ids:
            return []
        ids = list(chunk_ids)
        q: dict[str, Any] = {"_id": {"$in": ids}}
        scope_clause = _scope_match(viewer_id, is_admin=is_admin, scopes=scopes)
        if scope_clause:
            q = {"$and": [q, scope_clause]}
        cur = self._coll.find(q)
        by_id: dict = {}
        async for d in cur:
            by_id[d["_id"]] = {
                "id": d["_id"],
                "text": d.get("text", ""),
                "chunk_index": d.get("chunk_index"),
                "source": d.get("source"),
                "tags": d.get("tags") or [],
                "scope": d.get("scope") or SCOPE_GLOBAL,
                "owner_id": d.get("owner_id"),
                "metadata": d.get("metadata") or {},
                "ingested_at": d.get("ingested_at"),
            }
        return [by_id[i] for i in ids if i in by_id]
