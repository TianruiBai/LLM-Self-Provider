# Self-Hosted LLM Provider

A small, opinionated **OpenAI-compatible gateway** you run on your own
hardware. It is the single entry point for chat / embeddings / RAG /
multimodal calls and ships with multi-user auth, a per-user knowledge base,
hybrid (dense + FTS) retrieval, request auditing and one-click Docker
deploys.

```
                ┌──────────────────────────────────────────────────────────┐
   curl /       │ FastAPI gateway · port 8088                              │
   Continue /   │ ──────────────────────────────────────────────────────── │
   Cline /      │ • argon2id login + Bearer API keys + TOTP + OIDC         │
   Web UI       │ • per-user concurrency cap + token-bucket rate limit     │
   ────────────►│ • request audit log                                      │
                │ • OpenAI router (/v1/chat, /v1/embeddings, /v1/models)   │
                │ • RAG augment + kb_search tool                           │
                └──┬──────────────────┬───────────────────┬────────────────┘
                   │                  │                   │
        llama-runner://│  local binary  │           rag.backend=lance
                   ▼                  ▼                   ▼
            ┌─────────────┐    ┌──────────────┐   ┌────────────────────┐
            │ llama-server│    │ llama-server │   │  LanceDB on disk   │
            │ containers  │    │ (CUDA/Vulkan │   │  per-tenant tables │
            │ chat /      │    │  child proc) │   │  kb_global,        │
            │ embed       │    │  one-at-a-   │   │  kb_user_<id>      │
            │ (DeepSeek-V4│    │  time swap)  │   │  hybrid: dense+FTS │
            │  fork)      │    └──────────────┘   │  + RRF fusion      │
            └─────────────┘                       └────────────────────┘
                                            ▲
                                            │
                              ┌─────────────┴────────────┐
                              │  SQLite control DB       │
                              │  data/control.db (WAL)   │
                              │  users · sessions ·      │
                              │  api_keys · audit ·      │
                              │  oauth_state · config ·  │
                              │  model_publish           │
                              └──────────────────────────┘
```

## What's interesting

- **OpenAI wire-compatible.** Anything that talks `Authorization: Bearer
  sk-…` to `/v1/chat/completions` works (Continue, Cline, `openai-python`,
  curl).
- **Inference: swappable `llama-server` sibling containers.** Built from
  the antirez `llama.cpp-deepseek-v4-flash` fork, so the same image
  serves DeepSeek-V4-Flash *and* every other GGUF in your LM Studio
  cache. The gateway holds the docker socket and starts/swaps one
  container per slot (`chat`, `embed`) on demand. See
  [docs/LLAMA_CPP_CONTAINER.md](docs/LLAMA_CPP_CONTAINER.md) for the build
  command, knobs, and migration notes from the older vLLM architecture.
  A legacy local-binary `llama-server` path remains for setups where
  Docker isn't available.
- **LM Studio model autodiscovery.** GGUFs under
  `~/.lmstudio/models/<pub>/<model>/*.gguf` are auto-registered as
  `lmstudio/<pub>/<model>` ids. They stay invisible to end users until an
  admin publishes them — see *Admin curation*.
- **Per-user knowledge base.** LanceDB tables are partitioned `kb_global`
  and `kb_user_<id>`. Users see global + their own; admins see all.
  Retrieval is hybrid — dense (Qwen3-Embedding-8B) + FTS (tantivy) merged
  via Reciprocal Rank Fusion.
- **Reasoning over the KB.** A built-in `kb_search` tool is advertised in
  `/v1/tools` so models can issue mid-stream retrieval calls during
  reasoning, scoped by `{user, global}`.
- **One-click deploy.** `provider/scripts/install.sh` (Linux/macOS) or
  `install.ps1` (Windows) generates secrets, prompts for the first admin,
  then `docker compose up -d`.

## Authentication

| Caller            | Mechanism                            | Header / cookie                            |
|-------------------|--------------------------------------|--------------------------------------------|
| Web UI            | session cookie (HttpOnly + SameSite) | `PROV_SID=…`                               |
| OpenAI clients    | bearer API key                       | `Authorization: Bearer sk-prov-<22 chars>` |
| OAuth login       | one-shot state nonce                 | `provider_oauth_state` cookie + PKCE       |
| Bootstrap (1st run) | `PROVIDER_BOOTSTRAP_ADMIN_*` env vars | first admin auto-created on startup     |

Passwords are stored as **argon2id** hashes peppered with the env var
`PROVIDER_AUTH_PEPPER`. API keys are stored as `sha256(plain)` plus a
prefix preview; the plaintext is shown once at issue time. TOTP secrets are
sealed with `PROVIDER_MASTER_KEY` via XChaCha20-Poly1305.

## Admin curation of models

Auto-discovery (`models.yaml` + `models_dir` + LM Studio cache) typically
yields more GGUFs than you actually want exposed. The control DB has a
`model_publish` table — only admin-published ids are returned by
`/v1/models` for non-admin callers, and `/v1/chat/completions` rejects
unpublished ids with `403`.

```bash
# As admin
curl -X POST -H "Authorization: Bearer $ADMIN_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"label":"Qwen3 9B (chat)"}' \
     http://localhost:8088/admin/models/lmstudio/qwen/qwen3-9b-instruct/publish

curl -X POST -H "Authorization: Bearer $ADMIN_KEY" \
     http://localhost:8088/admin/models/lmstudio/qwen/qwen3-9b-instruct/unpublish
```

`GET /admin/models` returns the full registry with `published`, `label`,
`backend`, `kind`, `path` and `folder` for each entry.

## Knowledge base

| Scope          | Owner | Read                       | Write             |
|----------------|-------|----------------------------|-------------------|
| `kb_global`    | admin | every authenticated user   | admin only        |
| `kb_user_<id>` | user  | the user + admin           | the user + admin  |

Per-request override:

```jsonc
{
  "rag": {"enabled": true, "scope": ["user", "global"], "top_k": 6}
}
```

Use `python -m provider.scripts.migrate_mongo_to_lance` to migrate an
existing Mongo Atlas Local corpus into LanceDB without re-embedding.

## Quick start (local dev, llama.cpp backend)

```powershell
pip install -r provider/requirements.txt

# Generate secrets (or let install.sh do it for you)
$env:PROVIDER_AUTH_PEPPER = -join (1..32 | %{ '{0:x2}' -f (Get-Random -Maximum 256) })
$env:PROVIDER_MASTER_KEY  = -join (1..32 | %{ '{0:x2}' -f (Get-Random -Maximum 256) })
$env:PROVIDER_BOOTSTRAP_ADMIN_USER     = "admin"
$env:PROVIDER_BOOTSTRAP_ADMIN_PASSWORD = "<choose one>"

# (Optional) point at an LM Studio cache; default is ~/.lmstudio/models.
# Set provider/models.yaml -> server.lmstudio_dir if it lives elsewhere.

python -m provider --host 127.0.0.1 --port 8088
# UI:     http://127.0.0.1:8088/ui/
# Health: http://127.0.0.1:8088/health
```

The first admin lands at `/ui/` and can publish models, create users,
issue keys, and view the audit log.

## Quick start (Docker / llama-server backend)

```bash
cd provider
./scripts/install.sh                 # interactive (Linux/macOS)
# or on Windows:
./scripts/install.ps1
```

The installer:

1. Verifies Docker + Compose v2 (and on Linux, the NVIDIA container runtime).
2. Generates `provider/.env` with random `PROVIDER_AUTH_PEPPER` and
   `PROVIDER_MASTER_KEY`.
3. Prompts for the first admin username and password.
4. Builds the inference image once: `make build-llama` (compiles the
   antirez `llama.cpp-deepseek-v4-flash` fork against CUDA).
5. `docker compose up -d`.

`provider/compose.yml` brings up the **gateway** (8088) plus a build-only
`llama-server` service that produces the
`provider-llama-server:local` image. The gateway then spawns one sibling
container per slot on demand (`provider-llama-runner-chat`,
`provider-llama-runner-embed`) using the bind-mounted Docker socket.
Volumes `hf-cache`, `control-db` and `lance` persist between rebuilds.
The gateway mounts `provider/models.docker.yaml` over
`/app/provider/models.yaml`, which is configured for `rag.backend: lance`
and auto-discovers GGUFs from `~/.lmstudio/models`.

See [docs/LLAMA_CPP_CONTAINER.md](docs/LLAMA_CPP_CONTAINER.md) for build
options, CLI knobs (`LLAMA_IMAGE`, `LLAMA_CTX_SIZE`, `LLAMA_NGL`), preset
buttons in the web UI, and migration notes from the older vLLM
sibling-container architecture.

Convenience targets in `provider/Makefile`:

```bash
make up           # docker compose up -d
make build-llama  # build provider-llama-server:local from the fork
make logs         # tail gateway logs
make migrate      # Mongo -> Lance one-shot
make backup       # tarball of control.db + lance/ + .env (sans password)
make smoke        # run all phase smoke tests against the source tree
```

## API surface (highlights)

| Method · Path                          | Purpose                                  |
|----------------------------------------|------------------------------------------|
| `GET  /health`                         | Liveness + active-model snapshot         |
| `GET  /v1/models`                      | OpenAI list — filtered by publish state  |
| `POST /v1/chat/completions`            | OpenAI chat (with `rag`, `tools_builtin`)|
| `POST /v1/completions`                 | OpenAI completion (sub-agent default)    |
| `POST /v1/embeddings`                  | OpenAI embeddings (persistent embedder)  |
| `GET  /v1/tools`                       | List built-in tools (`kb_search`, …)     |
| `POST /rag/ingest`                     | Ingest documents (scope=user, global)    |
| `POST /rag/query`                      | Hybrid retrieval (dense + FTS + RRF)     |
| `GET  /rag/stats`, `/rag/documents`    | KB introspection (scope-aware)           |
| `POST /auth/login`, `/auth/logout`     | Web session                              |
| `*    /auth/keys/*`, `/auth/totp/*`    | Per-user API key + TOTP management       |
| `*    /auth/oidc/*`                    | GitHub / Google / generic OIDC           |
| `GET  /admin/models`                   | Full model registry (admin)              |
| `POST /admin/models/<id>/publish`      | Publish to end users (admin)             |
| `POST /admin/models/<id>/unpublish`    | Hide from end users (admin)              |
| `GET  /admin/users`, `/admin/audit`    | User CRUD + request audit log (admin)    |
| `GET  /events`                         | SSE bus (model-load progress, etc.)      |

Chat-with-RAG example:

```jsonc
POST /v1/chat/completions
{
  "model": "lmstudio/qwen/qwen3-9b-instruct",
  "messages": [{"role": "user", "content": "Summarize doc-42."}],
  "rag":  {"enabled": true, "top_k": 4, "scope": ["user", "global"]},
  "tools_builtin": true
}
```

## Repository layout (under `provider/`)

```
provider/
├── gateway.py              FastAPI app + all routes
├── auth.py / auth_deps.py  argon2 + sessions + API keys + TOTP
├── auth_routes.py          /auth/* HTTP routes
├── oidc.py / oidc_routes.py  GitHub + Google + generic OIDC
├── audit.py                request audit log middleware
├── ratelimit*.py           token-bucket rate limit middleware
├── concurrency_mw.py       per-user concurrency cap (Phase C6)
├── lifecycle.py            llama-server child supervisor + idle unload
├── registry.py             models.yaml + LM Studio + folder discovery
├── rag.py                  Mongo-backed RagService (legacy backend)
├── vector_store.py         LanceDB hybrid vector + FTS + RRF
├── tools.py                Built-in tool catalog (kb_search, web, code…)
├── db.py / migrations/     SQLite control plane + schema migrations
├── compose.yml             Docker stack (llama-server build + gateway)
├── Dockerfile              Gateway image
├── models.yaml             Local-dev model registry
├── models.docker.yaml      Docker registry (mounted over models.yaml)
├── scripts/
│   ├── install.sh / install.ps1   one-click installers
│   ├── backup.sh                  sqlite + lance + .env snapshot
│   └── migrate_mongo_to_lance.py  one-shot KB migrator
├── tests/                  Phase smoke tests (a8, b1, b6, c, publish)
└── web/                    Static UI (login, account, app)
```

## Phase status (per `docs/ARCHITECTURE.md` §7)

- **A** ✅ — auth, sessions, API keys, TOTP, OIDC, KB scoping, audit log,
  rate limit, admin user CRUD.
- **B** ✅ — LanceDB store, hybrid retrieval, `kb_search` tool, Mongo →
  Lance migration, backend selector.
- **C** ✅ — swappable llama-server containers (DeepSeek-V4 fork), per-user concurrency cap, Docker stack.
- **D** ✅ — installers (sh/ps1), Makefile, backup script, healthchecks.

Detailed design + outstanding decisions live in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Security notes

- All passwords are argon2id-with-pepper; API keys are sha256-hashed at rest.
- Sessions are HttpOnly + SameSite=Lax; secure-cookie can be forced via
  `PROVIDER_FORCE_SECURE_COOKIE=1` once you're behind HTTPS.
- OAuth uses state + PKCE; bootstrap (no users yet) is the only path that
  lets anonymous traffic reach admin routes — disable explicitly with
  `PROVIDER_DISABLE_BOOTSTRAP=1` after the first admin exists.
- `PROVIDER_AUTH_DEV_ALLOW=1` is a dev escape hatch; it logs loudly on
  every use and must not be set in production.
- Per-key IP allow-list (CIDR) and per-user concurrency cap protect against
  noisy neighbours and credential reuse from unexpected origins.
