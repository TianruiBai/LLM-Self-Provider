"""B3 — Migrate documents from MongoDB Atlas Local to LanceDB.

Reads every chunk from the Mongo collection used by ``provider.rag.RagService``
and writes them into the Lance directory layout described in
ARCHITECTURE.md §3.2. Embeddings stored on the Mongo side are reused
verbatim — we never call the embedder during migration.

Usage::

    python -m provider.scripts.migrate_mongo_to_lance \\
        --mongo-uri mongodb://127.0.0.1:27017 \\
        --database provider_rag \\
        --collection documents \\
        --lance-dir data/lance \\
        --embedding-dim 4096 \\
        [--batch 500] [--dry-run]

The migration is idempotent: re-running upserts by chunk id, so partial
runs can be resumed without dedupe issues.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import lancedb

log = logging.getLogger("migrate_mongo_to_lance")


def _make_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("title", pa.string()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("text", pa.string()),
            pa.field("chunk_idx", pa.int32()),
            pa.field("embedding", pa.list_(pa.float32(), dim)),
            pa.field("ingested_at", pa.float64()),
            pa.field("scope", pa.string()),
            pa.field("owner_id", pa.int64()),
            pa.field("meta_json", pa.string()),
        ]
    )


def _table_name_for(scope: str, owner_id: int | None) -> str:
    if scope == "global" or scope is None:
        return "kb_global"
    if owner_id is None:
        return "kb_global"
    return f"kb_user_{int(owner_id)}"


def _convert(doc: dict[str, Any], dim: int) -> dict[str, Any] | None:
    """Translate a Mongo chunk row into the Lance schema. Returns None on skip."""
    text = doc.get("text") or ""
    embedding = doc.get("embedding")
    if not text or not embedding:
        return None
    if len(embedding) != dim:
        log.warning("dim mismatch on _id=%s (got %d, want %d) — skipping", doc.get("_id"), len(embedding), dim)
        return None

    meta = doc.get("metadata") or {}
    scope = doc.get("scope") or "global"
    owner_id = doc.get("owner_id")
    if scope not in ("user", "global"):
        scope = "global"

    doc_id = (meta.get("id") if isinstance(meta, dict) else None) or str(doc.get("_id"))
    title = (meta.get("title") if isinstance(meta, dict) else None) or str(doc_id)

    return {
        "id": str(doc["_id"]),
        "doc_id": str(doc_id),
        "source": str(doc.get("source") or ""),
        "title": str(title),
        "tags": [str(t) for t in (doc.get("tags") or [])],
        "text": str(text),
        "chunk_idx": int(doc.get("chunk_index") or 0),
        "embedding": [float(x) for x in embedding],
        "ingested_at": float(doc.get("ingested_at") or time.time()),
        "scope": scope,
        "owner_id": int(owner_id) if owner_id is not None else 0,
        "meta_json": json.dumps(meta or {}, default=str),
    }


async def _migrate(args: argparse.Namespace) -> int:
    from pymongo import AsyncMongoClient

    schema = _make_schema(args.embedding_dim)
    out_root = Path(args.lance_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(out_root))

    # Per-table buffers + opened table cache.
    buffers: dict[str, list[dict[str, Any]]] = {}
    tables: dict[str, Any] = {}
    counts: dict[str, int] = {}

    def _flush(name: str) -> None:
        rows = buffers.get(name) or []
        if not rows:
            return
        if name not in tables:
            try:
                tables[name] = db.open_table(name)
            except Exception:  # noqa: BLE001
                tables[name] = db.create_table(name, schema=schema, mode="create")
        tbl = tables[name]
        arrow = pa.Table.from_pylist(rows, schema=schema)
        if args.dry_run:
            log.info("[dry-run] would upsert %d rows -> %s", len(rows), name)
        else:
            try:
                tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(arrow)
            except AttributeError:
                ids = [r["id"] for r in rows]
                escaped = ",".join(f"'{i}'" for i in ids)
                tbl.delete(f"id IN ({escaped})")
                tbl.add(arrow)
        counts[name] = counts.get(name, 0) + len(rows)
        buffers[name] = []

    client = AsyncMongoClient(args.mongo_uri)
    try:
        coll = client[args.database][args.collection]
        total_seen = 0
        skipped = 0
        cursor = coll.find({}, batch_size=int(args.batch))
        async for doc in cursor:
            total_seen += 1
            row = _convert(doc, args.embedding_dim)
            if row is None:
                skipped += 1
                continue
            name = _table_name_for(row["scope"], row["owner_id"] or None)
            buffers.setdefault(name, []).append(row)
            if len(buffers[name]) >= int(args.batch):
                _flush(name)
        for name in list(buffers):
            _flush(name)
    finally:
        await client.close()

    print("\n=== migration summary ===")
    print(f"  source         : mongodb://{args.database}.{args.collection}")
    print(f"  destination    : {out_root}")
    print(f"  embedding dim  : {args.embedding_dim}")
    print(f"  rows scanned   : {total_seen}")
    print(f"  rows skipped   : {skipped}")
    print(f"  rows migrated  : {sum(counts.values())}")
    for name, c in sorted(counts.items()):
        try:
            row_count = len(tables[name].to_pandas()) if name in tables else c
        except Exception:  # noqa: BLE001
            row_count = c
        print(f"    - {name:<30} +{c} (table now ~{row_count} rows)")
    print("  dry-run        :", args.dry_run)
    return 0


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Migrate Mongo RAG chunks to LanceDB.")
    ap.add_argument("--mongo-uri", default="mongodb://127.0.0.1:27017/?directConnection=true")
    ap.add_argument("--database", default="provider_rag")
    ap.add_argument("--collection", default="documents")
    ap.add_argument("--lance-dir", default="data/lance")
    ap.add_argument("--embedding-dim", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    args = _parse()
    rc = asyncio.run(_migrate(args))
    sys.exit(int(rc))


if __name__ == "__main__":
    main()
