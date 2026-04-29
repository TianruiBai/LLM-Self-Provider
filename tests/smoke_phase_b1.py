"""Smoke test for B1: LanceVectorStore parity with RagService.

Exercises the full surface (ingest, query, stats, list_documents,
get_chunks, delete) plus A9 KB-scoping rules (admin/user/global).
Uses a tiny fake embedder so the test runs without GPU/llama-server.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import shutil
import tempfile

# Tiny embeddings for fast tests.
DIM = 8
TMP = pathlib.Path(tempfile.mkdtemp(prefix="lance_smoke_"))

from provider.registry import ProviderConfig, ServerConfig, GatewayConfig, RagConfig
from provider import vector_store as vs

cfg = ProviderConfig(
    server=ServerConfig(llama_server_bin="unused"),
    gateway=GatewayConfig(),
    rag=RagConfig(
        backend="lance",
        embedding_dim=DIM,
        chunk_chars=400,
        chunk_overlap=40,
        default_top_k=3,
        lance_dir=str(TMP),
    ),
    models=[],
)


def fake_embed(texts):
    """Deterministic 8-dim embedding from char-bag hashing."""
    out = []
    for t in texts:
        v = [0.0] * DIM
        for ch in t.lower():
            v[ord(ch) % DIM] += 1.0
        # L2-normalize
        n = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / n for x in v])
    return out


async def fake_url():
    return "http://fake-embedder"


store = vs.LanceVectorStore(cfg, fake_url)

# Patch the embed call to skip HTTP entirely.
async def _embed(self, inputs):
    return fake_embed(inputs)
vs.LanceVectorStore.embed = _embed  # type: ignore[assignment]


async def main():
    await store.startup()

    # 1) Global ingest (admin uploads).
    r1 = await store.ingest(
        [
            {"text": "Pythagoras theorem about right triangles.", "metadata": {"id": "g1", "title": "Math"}},
            {"text": "Mitochondria are the powerhouse of the cell.", "metadata": {"id": "g2", "title": "Biology"}},
        ],
        source="docs",
        tags=["sci"],
        scope=vs.SCOPE_GLOBAL,
    )
    assert r1["chunks"] >= 2, r1
    print("[1/8] global ingest:", r1)

    # 2) Per-user ingest for users 7 and 9.
    r2 = await store.ingest(
        [{"text": "Alice's private journal entry about her cat.", "metadata": {"id": "u7"}}],
        source="journal",
        owner_id=7,
        scope=vs.SCOPE_USER,
    )
    r3 = await store.ingest(
        [{"text": "Bob's notes on Rust borrow checker semantics.", "metadata": {"id": "u9"}}],
        source="notes",
        owner_id=9,
        scope=vs.SCOPE_USER,
    )
    assert r2["chunks"] and r3["chunks"]
    print("[2/8] user ingests:", r2, r3)

    # 3) Visibility — user 7 should not see user 9's notes.
    hits_u7 = await store.query("Rust borrow checker", viewer_id=7, is_admin=False)
    assert all(h.metadata.get("owner_id") in (0, None, 7) for h in hits_u7), [
        (h.id, h.metadata.get("owner_id")) for h in hits_u7
    ]
    hits_u9 = await store.query("powerhouse cell", viewer_id=9, is_admin=False)
    sources = {h.metadata.get("source") for h in hits_u9}
    assert "docs" in sources, sources  # global is visible
    print("[3/8] scoping enforced (user 7 ≠ user 9, both see global)")

    # 4) Admin sees everything.
    hits_admin = await store.query("Rust borrow checker", viewer_id=1, is_admin=True)
    owner_ids = {h.metadata.get("owner_id") for h in hits_admin}
    assert 9 in owner_ids or any(h.metadata.get("source") == "notes" for h in hits_admin)
    print("[4/8] admin can read all user tables")

    # 5) Stats per viewer.
    s_user = await store.stats(viewer_id=7, is_admin=False)
    user_sources = {x["source"] for x in s_user["sources"]}
    assert "journal" in user_sources and "docs" in user_sources, user_sources
    assert "notes" not in user_sources, "user 7 must not see user 9's source"
    print("[5/8] stats scoping OK:", user_sources)

    # 6) list_documents groups chunks.
    cards = await store.list_documents(viewer_id=7, is_admin=False)
    titles = {c["title"] for c in cards}
    assert "Math" in titles or "Biology" in titles
    print("[6/8] list_documents:", sorted(titles))

    # 7) get_chunks round-trip.
    g_card = next(c for c in cards if c["source"] == "journal")
    fetched = await store.get_chunks(g_card["chunk_ids"], viewer_id=7, is_admin=False)
    assert fetched and fetched[0]["text"].startswith("Alice"), fetched
    # User 9 must not be able to fetch user 7's chunks.
    blocked = await store.get_chunks(g_card["chunk_ids"], viewer_id=9, is_admin=False)
    assert blocked == [], blocked
    print("[7/8] get_chunks honours scope")

    # 8) Delete.
    res = await store.delete(source="journal", viewer_id=7, is_admin=False)
    assert res["deleted"] >= 1, res
    s_after = await store.stats(viewer_id=7, is_admin=False)
    assert "journal" not in {x["source"] for x in s_after["sources"]}
    print("[8/8] delete + scope-aware:", res)

    await store.shutdown()
    print("ALL GREEN")


try:
    asyncio.run(main())
finally:
    shutil.rmtree(TMP, ignore_errors=True)
