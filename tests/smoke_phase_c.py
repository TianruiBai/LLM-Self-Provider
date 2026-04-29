"""Phase C smoke: vLLM lifecycle short-circuit + per-user concurrency cap."""
from __future__ import annotations

import asyncio
import os
import pathlib
import tempfile

tmp = pathlib.Path(tempfile.mkdtemp(prefix="phase_c_"))
os.environ["PROVIDER_AUTH_PEPPER"] = "a" * 64
os.environ["PROVIDER_MASTER_KEY"] = "b" * 64

from provider.registry import ProviderConfig, ServerConfig, GatewayConfig, RagConfig, ModelConfig
from provider.lifecycle import LifecycleManager

cfg = ProviderConfig(
    server=ServerConfig(llama_server_bin=""),
    gateway=GatewayConfig(),
    rag=RagConfig(backend="lance", embedding_dim=8, lance_dir=str(tmp / "lance")),
    models=[
        ModelConfig(id="chat", kind="chat", path="", backend="vllm", endpoint="http://vllm-chat:8000"),
        ModelConfig(id="embedding", kind="embedding", path="", backend="vllm", endpoint="http://vllm-embed:8000"),
        ModelConfig(id="vision", kind="vision", path="", backend="vllm", endpoint="http://vllm-vision:8000"),
    ],
)


async def main():
    lm = LifecycleManager(cfg)
    chat_url = await lm.ensure_chat("chat")
    emb_url = await lm.ensure_embedder()
    vis_url = await lm.ensure_vision()
    assert chat_url == "http://vllm-chat:8000", chat_url
    assert emb_url == "http://vllm-embed:8000", emb_url
    assert vis_url == "http://vllm-vision:8000", vis_url
    # No subprocess should have been spawned.
    assert lm._chat is None and lm._embed is None and lm._vision is None
    print("[1/3] vLLM short-circuit: no llama-server children spawned")


asyncio.run(main())

# Concurrency middleware:
from provider import concurrency_mw

concurrency_mw.configure(per_user=2)
slots = concurrency_mw._slots()


async def conc_test():
    await slots.acquire("user:7")
    await slots.acquire("user:7")
    try:
        await slots.acquire("user:7")
    except Exception as e:
        assert "concurrency" in str(e).lower(), e
    else:
        raise AssertionError("third acquire should have raised")
    # User 9 has its own pool.
    await slots.acquire("user:9")
    await slots.release("user:7")
    await slots.acquire("user:7")
    print("[2/3] per-user concurrency cap = 2 enforced")


asyncio.run(conc_test())

# Gateway boots with the vLLM-only config.
from provider.gateway import create_app
app = create_app(cfg)
assert app.title.startswith("Self-hosted")
print("[3/3] gateway boots with backend=vllm + lance")

print("ALL GREEN")
