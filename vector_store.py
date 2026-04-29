"""LanceDB-backed vector store (Phase B1).

Implements the same surface as ``provider.rag.RagService`` so the gateway
can switch backends via a single ``rag.backend`` config flag. Layout:

    data/lance/
      kb_global/chunks.lance        ← admin-managed, readable by all
      kb_user/<user_id>/chunks.lance ← owned by that user

Each chunk row has the columns documented in ARCHITECTURE.md §3.2.

Design notes
------------
* The LanceDB Python API is mostly sync; we wrap calls in ``asyncio.to_thread``
  so the public methods can stay ``async def`` like the Mongo-backed service.
* Upsert uses ``merge_insert`` keyed on ``id`` (the chunk id). This keeps
  the operation idempotent across re-ingests.
* ``query`` is dense-only for B1. Hybrid retrieval (FTS + RRF) lands in B5.
* ``stats`` / ``list_documents`` / ``delete`` / ``get_chunks`` mirror the
  Mongo service's filtering semantics, including A9 KB-scoping rules.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
import pyarrow as pa
import lancedb

from .registry import ProviderConfig

log = logging.getLogger("provider.vector_store")

SCOPE_USER = "user"
SCOPE_GLOBAL = "global"
_VALID_SCOPES = (SCOPE_USER, SCOPE_GLOBAL)


# ---------- helpers ----------------------------------------------------------


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


def _chunk_id(source: str, doc_key: str, idx: int, content: str) -> str:
    h = hashlib.sha1(f"{source}::{doc_key}::{idx}::{content}".encode("utf-8")).hexdigest()
    return h[:32]


def _doc_id_for(source: str, meta: dict[str, Any]) -> str:
    explicit = meta.get("id") if isinstance(meta, dict) else None
    if explicit:
        return str(explicit)
    return hashlib.sha1(f"{source}::{json.dumps(meta, sort_keys=True, default=str)}".encode("utf-8")).hexdigest()[:24]


@dataclass
class RagHit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


# ---------- the store --------------------------------------------------------


_TABLE_NAME = "chunks"


def _make_schema(embedding_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("title", pa.string()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("text", pa.string()),
            pa.field("chunk_idx", pa.int32()),
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
            pa.field("ingested_at", pa.float64()),
            pa.field("scope", pa.string()),
            pa.field("owner_id", pa.int64()),
            pa.field("meta_json", pa.string()),
        ]
    )


class LanceVectorStore:
    """Lance-backed RAG service. API parity with :class:`provider.rag.RagService`."""

    def __init__(self, cfg: ProviderConfig, embedder_base_url_provider):
        self.cfg = cfg
        self._embedder_url = embedder_base_url_provider
        # cfg.rag has no `lance_dir` field yet — derive a default.
        self._root = Path(getattr(cfg.rag, "lance_dir", None) or "data/lance").resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._dim = int(cfg.rag.embedding_dim)
        self._schema = _make_schema(self._dim)
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=300.0))
        self._db = lancedb.connect(str(self._root))
        # cache of opened tables, keyed by directory name
        self._tables: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    # ----------------- lifecycle -----------------

    async def startup(self) -> None:
        # Eagerly open kb_global so first query is fast.
        await asyncio.to_thread(self._open_or_create, "kb_global")

    async def shutdown(self) -> None:
        await self._http.aclose()

    # ----------------- table management -----------------

    def _table_name_for(self, *, scope: str, owner_id: int | None) -> str:
        if scope == SCOPE_GLOBAL:
            return "kb_global"
        if scope == SCOPE_USER:
            if owner_id is None:
                raise ValueError("scope='user' requires owner_id")
            return f"kb_user_{int(owner_id)}"
        raise ValueError(f"invalid scope {scope!r}")

    def _open_or_create(self, name: str):
        if name in self._tables:
            return self._tables[name]
        try:
            tbl = self._db.open_table(name)
        except (FileNotFoundError, ValueError):
            tbl = self._db.create_table(name, schema=self._schema, mode="create")
        except Exception as e:  # noqa: BLE001
            # LanceDB raises a generic error when missing on some versions.
            if "not found" not in str(e).lower() and "does not exist" not in str(e).lower():
                raise
            tbl = self._db.create_table(name, schema=self._schema, mode="create")
        self._tables[name] = tbl
        return tbl

    def _existing_table(self, name: str):
        if name in self._tables:
            return self._tables[name]
        try:
            tbl = self._db.open_table(name)
        except Exception:  # noqa: BLE001
            return None
        self._tables[name] = tbl
        return tbl

    def _list_visible_tables(
        self, *, viewer_id: int | None, is_admin: bool, scopes: list[str] | None
    ) -> list[tuple[str, str, int | None]]:
        """Return [(table_name, scope, owner_id), …] for the visibility set."""
        requested = [s for s in (scopes or [SCOPE_USER, SCOPE_GLOBAL]) if s in _VALID_SCOPES]
        if not requested:
            requested = [SCOPE_USER, SCOPE_GLOBAL]
        out: list[tuple[str, str, int | None]] = []
        if SCOPE_GLOBAL in requested:
            out.append(("kb_global", SCOPE_GLOBAL, None))
        if SCOPE_USER in requested:
            if is_admin:
                # All user tables.
                try:
                    names = self._db.table_names()
                except Exception:  # noqa: BLE001
                    names = []
                for n in names:
                    if n.startswith("kb_user_"):
                        try:
                            uid = int(n.removeprefix("kb_user_"))
                        except ValueError:
                            continue
                        out.append((n, SCOPE_USER, uid))
            elif viewer_id is not None:
                out.append((f"kb_user_{int(viewer_id)}", SCOPE_USER, int(viewer_id)))
        return out

    # ----------------- embeddings -----------------

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        provider = self._embedder_url
        base = provider() if not asyncio.iscoroutinefunction(provider) else await provider()
        if base is None:
            raise RuntimeError("Embedder is not running")
        embed_model = self.cfg.embedding_model
        model_id = embed_model.id if embed_model else "embedding"
        r = await self._http.post(
            f"{base}/v1/embeddings",
            json={"model": model_id, "input": inputs},
        )
        r.raise_for_status()
        payload = r.json()
        return [d["embedding"] for d in payload["data"]]

    # ----------------- ingestion -----------------

    async def ingest(
        self,
        documents: Iterable[dict[str, Any]],
        *,
        source: str = "api",
        tags: list[str] | None = None,
        owner_id: int | None = None,
        scope: str = SCOPE_GLOBAL,
    ) -> dict[str, Any]:
        if scope not in _VALID_SCOPES:
            raise ValueError(f"scope must be one of {_VALID_SCOPES}")

        rows: list[dict[str, Any]] = []
        now = time.time()
        for d in documents:
            text = d.get("text", "")
            meta = d.get("metadata") or {}
            doc_id = _doc_id_for(source, meta)
            title = (meta.get("title") if isinstance(meta, dict) else None) or doc_id
            for i, piece in enumerate(_chunk_text(text, self.cfg.rag.chunk_chars, self.cfg.rag.chunk_overlap)):
                key_seed = source + ":" + doc_id + (f":{owner_id}" if scope == SCOPE_USER and owner_id is not None else "")
                rows.append(
                    {
                        "id": _chunk_id(key_seed, doc_id, i, piece),
                        "doc_id": doc_id,
                        "source": source,
                        "title": str(title),
                        "tags": list(tags or []),
                        "text": piece,
                        "chunk_idx": i,
                        "ingested_at": float(now),
                        "scope": scope,
                        "owner_id": int(owner_id) if owner_id is not None else 0,
                        "meta_json": json.dumps(meta or {}, default=str),
                    }
                )

        if not rows:
            return {"inserted": 0, "skipped": 0, "chunks": 0}

        # Embed in batches.
        BATCH = 16
        for start in range(0, len(rows), BATCH):
            batch = rows[start : start + BATCH]
            vecs = await self.embed([r["text"] for r in batch])
            for r, v in zip(batch, vecs):
                if len(v) != self._dim:
                    raise ValueError(
                        f"embedding dim mismatch: got {len(v)}, expected {self._dim}"
                    )
                r["embedding"] = [float(x) for x in v]

        table_name = self._table_name_for(scope=scope, owner_id=owner_id)

        def _do_upsert() -> int:
            tbl = self._open_or_create(table_name)
            arrow = pa.Table.from_pylist(rows, schema=self._schema)
            try:
                tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(arrow)
            except AttributeError:
                # Older LanceDB: fall back to delete+add.
                ids = [r["id"] for r in rows]
                escaped = ",".join(f"'{i}'" for i in ids)
                tbl.delete(f"id IN ({escaped})")
                tbl.add(arrow)
            return len(rows)

        async with self._lock:
            inserted = await asyncio.to_thread(_do_upsert)

        return {"inserted": int(inserted), "skipped": 0, "chunks": int(inserted)}

    # ----------------- retrieval -----------------

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
        hybrid: bool = True,
    ) -> list[RagHit]:
        if not text.strip():
            return []
        k = int(top_k or self.cfg.rag.default_top_k)
        [vec] = await self.embed([text])

        targets = self._list_visible_tables(
            viewer_id=viewer_id, is_admin=is_admin, scopes=scopes
        )

        def _filter_clause() -> str | None:
            parts: list[str] = []
            if source:
                parts.append(f"source = '{source}'")
            if tags:
                # any-tag match: array_has_any(tags, [...])
                quoted = ",".join(f"'{t}'" for t in tags)
                parts.append(f"array_has_any(tags, [{quoted}])")
            return " AND ".join(parts) if parts else None

        flt = _filter_clause()
        cap = max(50, k * 4)

        def _ensure_fts(tbl) -> bool:
            try:
                tbl.create_fts_index("text", replace=False)
                return True
            except Exception:  # noqa: BLE001
                # Already exists or unsupported — search may still work.
                return True

        def _search_one(name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            tbl = self._existing_table(name)
            if tbl is None:
                return [], []
            # Dense search.
            try:
                qd = tbl.search(vec).limit(cap)
                if flt:
                    qd = qd.where(flt, prefilter=True)
                dense_rows = qd.to_list()
            except Exception as e:  # noqa: BLE001
                log.warning("dense search on %s failed: %s", name, e)
                dense_rows = []
            sparse_rows: list[dict[str, Any]] = []
            if hybrid:
                try:
                    _ensure_fts(tbl)
                    qs = tbl.search(text, query_type="fts").limit(cap)
                    if flt:
                        qs = qs.where(flt, prefilter=True)
                    sparse_rows = qs.to_list()
                except Exception as e:  # noqa: BLE001
                    # FTS unsupported / empty index — fall back to dense-only.
                    log.debug("fts search on %s skipped: %s", name, e)
                    sparse_rows = []
            return dense_rows, sparse_rows

        per_table: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = await asyncio.gather(
            *[asyncio.to_thread(_search_one, name) for name, _, _ in targets]
        )

        dense_all: list[dict[str, Any]] = []
        sparse_all: list[dict[str, Any]] = []
        for d, s in per_table:
            dense_all.extend(d)
            sparse_all.extend(s)

        # Lance returns _distance for vector and _score for FTS. Build per-rank
        # contributions via Reciprocal Rank Fusion (k0=60, the canonical value).
        def _dense_rank_key(row: dict[str, Any]) -> float:
            if "_distance" in row:
                return float(row["_distance"])  # smaller = better
            return -float(row.get("score") or 0.0)

        def _sparse_rank_key(row: dict[str, Any]) -> float:
            # FTS score = larger is better; negate so sort ascending.
            for key in ("_score", "score"):
                if key in row:
                    return -float(row[key])
            return 0.0

        dense_sorted = sorted(dense_all, key=_dense_rank_key)
        sparse_sorted = sorted(sparse_all, key=_sparse_rank_key)

        K0 = 60.0
        scores: dict[str, float] = {}
        rows_by_id: dict[str, dict[str, Any]] = {}
        for rank, row in enumerate(dense_sorted):
            rid = str(row.get("id"))
            if not rid:
                continue
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (K0 + rank)
            rows_by_id.setdefault(rid, row)
        for rank, row in enumerate(sparse_sorted):
            rid = str(row.get("id"))
            if not rid:
                continue
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (K0 + rank)
            rows_by_id.setdefault(rid, row)

        # If hybrid was disabled or FTS produced nothing, fall back to a pure
        # dense ordering using the converted similarity score.
        if not scores:
            return []

        ranked_ids = sorted(scores.keys(), key=lambda i: -scores[i])[:k]

        hits: list[RagHit] = []
        for rid in ranked_ids:
            row = rows_by_id[rid]
            try:
                meta = json.loads(row.get("meta_json") or "{}")
            except Exception:  # noqa: BLE001
                meta = {}
            hits.append(
                RagHit(
                    id=rid,
                    text=row.get("text") or "",
                    score=float(scores[rid]),
                    metadata={
                        "source": row.get("source"),
                        "tags": list(row.get("tags") or []),
                        "scope": row.get("scope") or SCOPE_GLOBAL,
                        "owner_id": row.get("owner_id"),
                        "doc_id": row.get("doc_id"),
                        "title": row.get("title"),
                        **(meta or {}),
                    },
                )
            )
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

    # ----------------- knowledge management -----------------

    def _all_rows(
        self,
        *,
        viewer_id: int | None,
        is_admin: bool,
        scopes: list[str] | None,
        where: str | None = None,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        targets = self._list_visible_tables(
            viewer_id=viewer_id, is_admin=is_admin, scopes=scopes
        )
        out: list[dict[str, Any]] = []
        for name, _, _ in targets:
            tbl = self._existing_table(name)
            if tbl is None:
                continue
            try:
                q = tbl.search().select(columns) if columns else tbl.search()
                if where:
                    q = q.where(where)
                # tbl.search() with no vector returns all rows (subject to limit).
                # Use a generous cap; KB views aren't expected to be huge.
                rows = q.limit(100_000).to_list()
            except Exception:  # noqa: BLE001
                # Fall back to to_pandas for full scan.
                try:
                    df = tbl.to_pandas()
                    rows = df.to_dict("records")
                except Exception:  # noqa: BLE001
                    rows = []
            out.extend(rows)
        return out

    async def stats(
        self,
        *,
        viewer_id: int | None = None,
        is_admin: bool = False,
        scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        def _build() -> dict[str, Any]:
            rows = self._all_rows(
                viewer_id=viewer_id,
                is_admin=is_admin,
                scopes=scopes,
                columns=["source", "tags", "ingested_at", "doc_id"],
            )
            by_source: dict[str, dict[str, Any]] = {}
            for r in rows:
                src = r.get("source") or "(none)"
                bucket = by_source.setdefault(
                    src, {"chunks": 0, "documents": set(), "tags": set(), "last": 0.0}
                )
                bucket["chunks"] += 1
                if r.get("doc_id"):
                    bucket["documents"].add(r["doc_id"])
                for t in r.get("tags") or []:
                    bucket["tags"].add(t)
                bucket["last"] = max(bucket["last"], float(r.get("ingested_at") or 0.0))
            sources = [
                {
                    "source": k,
                    "chunks": v["chunks"],
                    "documents": sorted(v["documents"]),
                    "tags": sorted(v["tags"]),
                    "last_ingested_at": v["last"],
                }
                for k, v in sorted(by_source.items(), key=lambda kv: -kv[1]["last"])
            ]
            return {"total_chunks": len(rows), "sources": sources}

        return await asyncio.to_thread(_build)

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
        def _build() -> list[dict[str, Any]]:
            where_parts: list[str] = []
            if source:
                where_parts.append(f"source = '{source}'")
            if tag:
                where_parts.append(f"array_has(tags, '{tag}')")
            where = " AND ".join(where_parts) if where_parts else None
            rows = self._all_rows(
                viewer_id=viewer_id,
                is_admin=is_admin,
                scopes=scopes,
                where=where,
            )
            rows.sort(key=lambda r: int(r.get("chunk_idx") or 0))
            grouped: dict[tuple[str, str], dict[str, Any]] = {}
            for r in rows:
                key = (r.get("source") or "", r.get("doc_id") or r.get("id"))
                bucket = grouped.get(key)
                if bucket is None:
                    bucket = {
                        "source": key[0],
                        "doc_id": key[1],
                        "title": r.get("title") or key[1],
                        "tags": list(r.get("tags") or []),
                        "scope": r.get("scope") or SCOPE_GLOBAL,
                        "owner_id": int(r.get("owner_id") or 0) or None,
                        "chunks": 0,
                        "preview": "",
                        "ingested_at": float(r.get("ingested_at") or 0.0),
                        "chunk_ids": [],
                    }
                    grouped[key] = bucket
                bucket["chunks"] += 1
                bucket["chunk_ids"].append(r.get("id"))
                if not bucket["preview"]:
                    txt = (r.get("text") or "").strip()
                    if len(txt) > 280:
                        txt = txt[:280].rstrip() + "…"
                    bucket["preview"] = txt
                bucket["ingested_at"] = max(
                    bucket["ingested_at"], float(r.get("ingested_at") or 0.0)
                )
            cards = sorted(grouped.values(), key=lambda c: -c["ingested_at"])
            return cards[: int(limit)]

        return await asyncio.to_thread(_build)

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
        if not (chunk_ids or source or doc_id or tag):
            raise ValueError("delete requires at least one of: chunk_ids, source, doc_id, tag")

        def _build_where() -> str:
            parts: list[str] = []
            if chunk_ids:
                quoted = ",".join(f"'{i}'" for i in chunk_ids)
                parts.append(f"id IN ({quoted})")
            if source:
                parts.append(f"source = '{source}'")
            if doc_id:
                parts.append(f"doc_id = '{doc_id}'")
            if tag:
                parts.append(f"array_has(tags, '{tag}')")
            return " AND ".join(parts)

        where = _build_where()

        def _do_delete() -> int:
            targets = self._list_visible_tables(
                viewer_id=viewer_id, is_admin=is_admin, scopes=scopes
            )
            total = 0
            for name, _, _ in targets:
                tbl = self._existing_table(name)
                if tbl is None:
                    continue
                # Count first (Lance returns no count from delete()).
                try:
                    n = tbl.search().where(where).limit(1_000_000).to_list()
                    count = len(n)
                except Exception:  # noqa: BLE001
                    count = 0
                if count:
                    try:
                        tbl.delete(where)
                    except Exception as e:  # noqa: BLE001
                        log.warning("lance delete on %s failed: %s", name, e)
                        continue
                    total += count
            return total

        deleted = await asyncio.to_thread(_do_delete)
        return {"deleted": int(deleted)}

    async def get_chunks(
        self,
        chunk_ids: list[str],
        *,
        viewer_id: int | None = None,
        is_admin: bool = False,
        scopes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        ids = list(chunk_ids)
        quoted = ",".join(f"'{i}'" for i in ids)
        where = f"id IN ({quoted})"

        def _fetch() -> list[dict[str, Any]]:
            rows = self._all_rows(
                viewer_id=viewer_id, is_admin=is_admin, scopes=scopes, where=where
            )
            by_id: dict[str, dict[str, Any]] = {}
            for r in rows:
                try:
                    meta = json.loads(r.get("meta_json") or "{}")
                except Exception:  # noqa: BLE001
                    meta = {}
                by_id[str(r["id"])] = {
                    "id": str(r["id"]),
                    "text": r.get("text") or "",
                    "chunk_index": int(r.get("chunk_idx") or 0),
                    "source": r.get("source"),
                    "tags": list(r.get("tags") or []),
                    "scope": r.get("scope") or SCOPE_GLOBAL,
                    "owner_id": int(r.get("owner_id") or 0) or None,
                    "metadata": meta,
                    "ingested_at": float(r.get("ingested_at") or 0.0),
                }
            return [by_id[i] for i in ids if i in by_id]

        return await asyncio.to_thread(_fetch)


__all__ = [
    "LanceVectorStore",
    "RagHit",
    "SCOPE_USER",
    "SCOPE_GLOBAL",
]
