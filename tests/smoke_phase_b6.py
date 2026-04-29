"""Smoke test for B5+B6: hybrid retrieval + kb_search tool."""
from __future__ import annotations

import asyncio
import pathlib
import shutil
import tempfile

DIM = 8
TMP = pathlib.Path(tempfile.mkdtemp(prefix="lance_b6_"))

from provider.registry import ProviderConfig, ServerConfig, GatewayConfig, RagConfig
from provider import vector_store as vs, tools as builtin_tools

cfg = ProviderConfig(
    server=ServerConfig(llama_server_bin="unused"),
    gateway=GatewayConfig(),
    rag=RagConfig(
        backend="lance", embedding_dim=DIM, chunk_chars=300, chunk_overlap=30,
        default_top_k=3, lance_dir=str(TMP),
    ),
    models=[],
)


def fake_embed(texts):
    out = []
    for t in texts:
        v = [0.0] * DIM
        for ch in t.lower():
            v[ord(ch) % DIM] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / n for x in v])
    return out


async def fake_url():
    return "http://fake-embedder"


vs.LanceVectorStore.embed = lambda self, inputs: asyncio.sleep(0, fake_embed(inputs))  # type: ignore

# Replace with a proper async lambda:
async def _embed(self, inputs):
    return fake_embed(inputs)
vs.LanceVectorStore.embed = _embed  # type: ignore

store = vs.LanceVectorStore(cfg, fake_url)


async def main():
    await store.startup()
    await store.ingest(
        [
            {"text": "The Pythagoras theorem describes right triangle sides.", "metadata": {"id": "m1", "title": "Math"}},
            {"text": "Mitochondria are the powerhouse of the cell, supplying ATP energy.", "metadata": {"id": "m2", "title": "Bio"}},
        ],
        source="docs",
        scope=vs.SCOPE_GLOBAL,
    )
    await store.ingest(
        [{"text": "Alice's note: my cat Whiskers learned a new trick today.", "metadata": {"id": "u7"}}],
        source="journal", owner_id=7, scope=vs.SCOPE_USER,
    )

    # 1) hybrid query — both dense and FTS contribute.
    hits = await store.query("Whiskers", viewer_id=7, is_admin=False, top_k=2, hybrid=True)
    assert hits, "hybrid retrieval returned no hits"
    assert any("Whiskers" in h.text for h in hits), [h.text for h in hits]
    print("[1/3] hybrid retrieval finds Whiskers:", [h.metadata["source"] for h in hits])

    # 2) kb_search tool dispatches through execute_tool.
    res = await builtin_tools.execute_tool(
        "kb_search",
        {"query": "powerhouse of the cell", "top_k": 2, "scope": "both"},
        rag=store,
        viewer_id=7,
        is_admin=False,
    )
    assert "hits" in res and res["hits"], res
    assert any("powerhouse" in (h.get("preview") or "").lower() for h in res["hits"]), res
    print("[2/3] kb_search via execute_tool:", [(h["score"], h["source"]) for h in res["hits"]])

    # 3) scope='user' restricts to private KB.
    res2 = await builtin_tools.execute_tool(
        "kb_search",
        {"query": "cat trick", "scope": "user"},
        rag=store, viewer_id=7, is_admin=False,
    )
    assert res2.get("hits"), res2
    sources = {h.get("source") for h in res2["hits"]}
    assert sources == {"journal"}, sources
    print("[3/3] kb_search scope=user enforced:", sources)

    # 4) merge_tools advertises kb_search when has_kb=True.
    merged = builtin_tools.merge_tools(None, want_builtin=False, has_kb=True)
    names = {t["function"]["name"] for t in merged}
    assert "kb_search" in names, names

    await store.shutdown()
    print("ALL GREEN")


try:
    asyncio.run(main())
finally:
    shutil.rmtree(TMP, ignore_errors=True)
