# Self-hosted Model Provider

OpenAI-compatible gateway over `llama-server` (Vulkan) with on-demand model
swapping and RAG via MongoDB Atlas Local + Qwen3-Embedding-8B.

## Architecture

```
                     +-------------------------------+
  HTTP (OpenAI API)  |   FastAPI gateway (port 8000) |
  ---------------->  |   /v1/{models,chat,embed,...} |
                     |   /rag/{ingest,query}         |
                     +---------------+---------------+
                                     |
            +------------------------+--------------------------+
            |                                                   |
   spawn/swap (one at a time)                          persistent
            v                                                   v
  +---------------------+                          +-----------------------+
  | llama-server (chat) |                          | llama-server (embed)  |
  | port 18001          |                          | port 18002            |
  | Vulkan backend      |                          | Qwen3-Embedding-8B    |
  +---------------------+                          +-----------+-----------+
                                                               |
                                                       /v1/embeddings
                                                               |
                                                               v
                                                  +-------------------------+
                                                  | MongoDB Atlas Local     |
                                                  | (docker, port 27017)    |
                                                  | $vectorSearch           |
                                                  +-------------------------+
```

- **One chat model is active at a time** on `chat_port` (18001). When a
  request arrives for a different model id, the current child is shut down
  and a new one is spawned. Switching is serialized.
- **The embedder is persistent** on `embedding_port` (18002). It is also
  used by the RAG pipeline.
- All llama-server children are launched with the **Vulkan-built** binary
  from the antirez DeepSeek-V4-Flash fork (`build-vulkan-ds4`).

## Setup

```powershell
# 1. Install Python deps
& "C:/Users/Tass/.unsloth/studio/unsloth_studio/Scripts/python.exe" -m pip install -r requirements-provider.txt

# 2. Start MongoDB Atlas Local
docker compose up -d mongodb

# 3. Start the provider (will spawn the embedder; chat models load on demand)
& "C:/Users/Tass/.unsloth/studio/unsloth_studio/Scripts/python.exe" -m provider
```

## API

### List registered models
`GET /v1/models`

### Chat completions (auto-loads / swaps the model)
`POST /v1/chat/completions`
```json
{
  "model": "qwen/qwen3.6-27b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}
```

### Chat completions with RAG augmentation
```json
{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "What is X?"}],
  "rag": {"enabled": true, "top_k": 4, "source": "docs"}
}
```

### Embeddings
`POST /v1/embeddings` — proxied to the persistent embedder.

### RAG ingest
`POST /rag/ingest`
```json
{
  "documents": [
    {"text": "...", "metadata": {"id": "doc-1", "title": "..."}},
    {"text": "..."}
  ],
  "source": "docs",
  "tags": ["onboarding"]
}
```

### RAG query
`POST /rag/query`
```json
{"text": "what is X?", "top_k": 4, "source": "docs"}
```

### Admin (force-load a chat model)
`POST /admin/load` `{"model": "minimax-m2.7"}`

## Configuration

All registered models, ports, and RAG settings live in
[`provider/models.yaml`](models.yaml). Per-model `args` are appended verbatim
to `llama-server`, so tune ctx, devices, batch sizes, split mode there.

## Notes & known constraints

- Switching between large models has a cold-load cost (DeepSeek-V4-Flash 158B
  takes minutes; the gateway awaits `/health` before serving).
- DeepSeek-V4-Flash currently uses two-GPU layer split with weighted
  fit-targets; do not enable pipeline parallelism (the fork patches it off).
- The CUDA→Vulkan switch reuses the same antirez fork rebuilt with
  `-DGGML_VULKAN=ON`. Vulkan device order is **inverted** vs CUDA: the 24 GiB
  card is `Vulkan0`, the 8 GiB laptop card is `Vulkan1`.
