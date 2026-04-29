# LLM Self-Provider — Production Architecture & Migration Plan

> **Status:** design document for the v2 production rewrite.
> **Owner:** TianruiBai / BaiTian6641
> **Target completion:** phased; each phase ships a working system.

---

## 1. Goals (from product brief)

| # | Goal | Phase |
|---|------|-------|
| 1 | Multi-user concurrency + per-request throughput | C |
| 2 | Production-grade vector search | B |
| 3 | Reasoning/prompt search through the vector DB | B + A |
| 4 | Multi-user UI with admin/user roles, per-user API keys, IP audit | A |
| 5 | Per-user **and** global knowledge bases | A → B |
| 6 | One-click deploy on Linux / Windows / macOS | D |

## 2. Target stack (v2)

```
┌────────────────────── Client (UI / Continue / Cline / curl) ──────────────────────┐
│  Auth: session cookie  (web UI)  |  Authorization: Bearer sk-…  (OpenAI clients) │
└──────────────────────────────────────┬────────────────────────────────────────────┘
                                       │ HTTPS
                                       ▼
                       ┌─────────────────────────────────┐
                       │   FastAPI gateway (asyncio)     │
                       │   ─────────────────────────────  │
                       │   • auth middleware              │
                       │   • per-user rate limit          │
                       │   • request audit log            │
                       │   • OpenAI-compatible router     │
                       │   • RAG enrichment               │
                       └──┬──────────────────┬──────────┬─┘
                          │                  │          │
                          ▼                  ▼          ▼
                   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                   │   vLLM       │  │   LanceDB    │  │   SQLite     │
                   │ (Docker)     │  │ (vectors)    │  │ (control DB) │
                   │ openai api   │  │ on-disk      │  │ users, keys, │
                   │ /v1/chat ... │  │ partitioned  │  │ sessions,    │
                   └──────┬───────┘  │ per-tenant   │  │ audit, conf  │
                          │          └──────┬───────┘  └──────────────┘
                          │                 │
                          │           ┌─────▼──────┐
                          └──────────►│ Embedder   │ (vLLM /v1/embeddings)
                                      │ Qwen3-Emb  │
                                      └────────────┘
```

### 2.1 Why the swap

| Concern | v1 (today) | v2 |
|---|---|---|
| Concurrency | llama-server: single in-flight chat per process | vLLM PagedAttention + continuous batching |
| Vector store | MongoDB `$vectorSearch` (Atlas-Local) | LanceDB (columnar, on-disk, native ANN) |
| Auth | none | bcrypt/argon2 + sessions + per-user API keys + optional OIDC + TOTP |
| Multi-tenant KB | single shared collection | per-user lance dataset + `kb_global/` |
| Deploy | manual | `compose.yml` + bootstrap scripts per OS |

## 3. Data model (v2)

### 3.1 SQLite — `data/control.db`

Pure relational; no secrets stored in plaintext. WAL mode, single writer.

```sql
-- Users -----------------------------------------------------------------
CREATE TABLE users (
  id              INTEGER PRIMARY KEY,
  username        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
  email           TEXT             UNIQUE COLLATE NOCASE,
  password_hash   TEXT,                       -- argon2id; NULL if OIDC-only
  role            TEXT    NOT NULL DEFAULT 'user',  -- 'user' | 'admin'
  is_active       INTEGER NOT NULL DEFAULT 1,
  totp_secret     TEXT,                       -- base32, encrypted at rest with master key
  totp_enabled    INTEGER NOT NULL DEFAULT 0,
  oidc_subject    TEXT,                       -- 'github:1234' / 'google:abc'
  created_at      INTEGER NOT NULL,
  last_login_at   INTEGER,
  last_login_ip   TEXT
);

-- API keys (multiple per user) ------------------------------------------
CREATE TABLE api_keys (
  id              INTEGER PRIMARY KEY,
  user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name            TEXT    NOT NULL,
  key_prefix      TEXT    NOT NULL,           -- first 8 chars after 'sk-' for display
  key_hash        TEXT    NOT NULL UNIQUE,    -- sha256(plain) — never store plain
  created_at      INTEGER NOT NULL,
  last_used_at    INTEGER,
  last_used_ip    TEXT,
  expires_at      INTEGER,
  revoked_at      INTEGER
);

-- Sessions (web UI) -----------------------------------------------------
CREATE TABLE sessions (
  id              TEXT    PRIMARY KEY,        -- 256-bit random, base64url
  user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at      INTEGER NOT NULL,
  expires_at      INTEGER NOT NULL,
  ip              TEXT,
  user_agent      TEXT
);

-- Per-key request audit (truncated; rolling) -----------------------------
CREATE TABLE request_audit (
  id              INTEGER PRIMARY KEY,
  ts              INTEGER NOT NULL,
  user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
  api_key_id      INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
  ip              TEXT,
  method          TEXT,
  path            TEXT,
  status          INTEGER,
  bytes_in        INTEGER,
  bytes_out       INTEGER,
  duration_ms     INTEGER
);
CREATE INDEX idx_audit_ts ON request_audit(ts);

-- OAuth state / nonces (TTL'd) ------------------------------------------
CREATE TABLE oauth_states (
  state           TEXT PRIMARY KEY,
  provider        TEXT NOT NULL,
  expires_at      INTEGER NOT NULL,
  redirect_to     TEXT
);

-- Server config (key/value, replaces runtime_config.json) ---------------
CREATE TABLE config_kv (
  key             TEXT PRIMARY KEY,
  value           TEXT NOT NULL,
  updated_at      INTEGER NOT NULL,
  updated_by      INTEGER REFERENCES users(id) ON DELETE SET NULL
);
```

### 3.2 LanceDB — `data/lance/`

```
data/lance/
├── kb_global/                 ← admin-managed, readable by all users
│   └── chunks.lance/          ← arrow columns: id, doc_id, source, tags[], text, embedding
└── kb_user/<user_id>/
    └── chunks.lance/
```

**Schema** (Arrow):

| column | type | notes |
|---|---|---|
| `id` | string | `<source>::<doc_id>::<chunk_idx>` |
| `doc_id` | string | sha1 of canonical key (preserved across re-ingests) |
| `source` | string | logical source / collection |
| `title` | string | best-effort doc title |
| `tags` | list&lt;string&gt; | for facet filter |
| `text` | string | the chunk |
| `chunk_idx` | int32 | |
| `embedding` | fixed_size_list&lt;float32, 4096&gt; | Qwen3-Embed-8B dim |
| `created_at` | timestamp[ms] | |
| `meta` | string (JSON) | extras |

**Indexes:**
- `IVF_PQ` ANN index on `embedding` (configurable `nlist`, `m`).
- `BTREE` on `(source, tags)` for facet pre-filter.
- `FTS` on `text` (LanceDB built-in tantivy) — used for *hybrid* retrieval (Goal 3).

### 3.3 LanceDB query — hybrid reasoning/prompt search (Goal 3)

```python
hits_dense  = tbl.search(query_emb).where(filter_sql).limit(k * 4).to_list()
hits_sparse = tbl.search(query_text, query_type="fts").where(filter_sql).limit(k * 4).to_list()
fused = reciprocal_rank_fusion(hits_dense, hits_sparse, k=60)[:k]
```

The **prompt-search** mode means the gateway, before passing the prompt to the LLM, can execute a hybrid retrieval over the user's KB(s) (private + global) and prepend a retrieval block. The same vector backend additionally serves a stand-alone reasoning search mode where the model can issue intermediate `kb.search` tool calls during streaming reasoning — already wired through the existing tool-call loop in `gateway.py`.

## 4. Auth design (Goal 4)

### 4.1 Authentication paths

| Caller | Mechanism | Header / cookie |
|---|---|---|
| Web UI | session cookie | `provider_session=<sid>; HttpOnly; Secure; SameSite=Lax` |
| OpenAI-compatible client (Continue, Cline, curl) | bearer | `Authorization: Bearer sk-prov-<22 chars>` |
| OAuth callback | one-shot state token | `provider_oauth_state` cookie |

### 4.2 Password storage
- **argon2id** (`argon2-cffi`) with parameters tuned to ~250ms on the server.
- Pepper (env var `PROVIDER_AUTH_PEPPER`, 256-bit hex) HMAC-prefixed before hashing.

### 4.3 API keys
- Format: `sk-prov-<22 base62>` (≈ 130 bits entropy).
- DB stores **only** `sha256(key)` and the first 8 chars after the prefix for display: `sk-prov-XYZABC12…`
- Admin UI shows `sk-prov-XYZABC12…` plus *never* the full key. The full key is shown **once** at creation time.
- `last_used_ip` / `last_used_at` updated atomically per request.
- Revocation = set `revoked_at`; the verifier rejects revoked rows.

### 4.4 Two-factor (TOTP)
- RFC-6226 30s windows.
- Secret encrypted with the master key (`PROVIDER_MASTER_KEY`) using XChaCha20-Poly1305 (libsodium / `pynacl`).
- Recovery codes: 10 single-use codes hashed with argon2id, stored alongside.

### 4.5 OAuth/OIDC
- Pluggable providers via `data/oidc.yaml` (start with GitHub + Google + generic OIDC).
- On first login, the user is created with `oidc_subject` set; password auth disabled unless they later set one in *Settings → Security*.

### 4.6 Roles
- `user`: own KB, own API keys, own sessions, own settings.
- `admin`: all of the above + global KB management + user CRUD + masked-key listing + audit log + IP allow-list per key.

### 4.7 Per-key endpoint IP allow-list (admin)
- Optional column on `api_keys` (`ip_allowlist TEXT NULL`) holding CIDR list (CSV).
- Verifier rejects with `403 ip_not_allowed` when the request IP is not within the list.

## 5. Knowledge base scoping (Goal 5)

| Scope | Owner | Who can read | Who can ingest |
|---|---|---|---|
| `kb_user/<id>` | the user | the user, the admin | the user, the admin |
| `kb_global` | admin role | all logged-in users | admin only |

Per-request scope override (RAG options):

```jsonc
{
  "rag": {
    "enabled": true,
    "scope": ["user", "global"],   // default: ["user","global"]
    "top_k": 6
  }
}
```

The gateway opens both Lance tables, performs per-table search, then fuses results with RRF using the same per-scope weights (`global_weight`, `user_weight` configurable in `config_kv`).

## 6. Inference backend (Goal 1, vLLM)

### 6.1 Why Docker-only
vLLM does **not** support Windows natively (it requires Linux + CUDA + xformers). Docker Desktop on Windows/macOS provides an identical runtime to Linux servers with no per-platform branching.

### 6.2 Compose topology

```yaml
# compose.yml (excerpt)
services:
  gateway:    { build: ./provider, ports: ["8088:8088"] }
  vllm-chat:  { image: vllm/vllm-openai:v0.6.0, runtime: nvidia, command: --model … }
  vllm-embed: { image: vllm/vllm-openai:v0.6.0, runtime: nvidia, command: --model Qwen/Qwen3-Embedding-8B --task embed }
  vllm-vision:{ image: vllm/vllm-openai:v0.6.0, runtime: nvidia, command: --model google/gemma-3-4b-it }
  lance:      not a service — embedded in gateway via volume
```

Gateway talks to all three via OpenAI-compatible REST (the same code path used today for llama-server). The `LifecycleManager` is replaced with a thin `BackendRouter` that issues docker-compose `up`/`stop` commands or exposes them via a sidecar control socket — *or* leaves vLLM resident and uses vLLM's **dynamic adapter loading** for hot model swaps.

### 6.3 Migration shim
During the transition, the existing llama-server lifecycle stays as a fallback. A `backend: "llama_cpp" | "vllm"` field in `models.yaml` per-model picks the runtime; the gateway routes upstream calls to whichever process is registered for that model id.

## 7. Phased delivery plan

> Each phase merges to `main` independently and the system stays runnable.

### **Phase A — Multi-user, auth, KB scoping (no backend change)**
Stack stays on llama.cpp + MongoDB; we layer auth + scoping on top.

- [A1] Add `data/control.db` SQLite + Alembic-style schema migrator.
- [A2] `provider/auth.py`: argon2 password hashing, session store, API-key issuance + verify.
- [A3] FastAPI dependency `current_user` + middleware `require_session` / `require_api_key`.
- [A4] `/v1/*` requires `Authorization: Bearer sk-prov-…` (legacy unauthenticated flag for local dev).
- [A5] Web UI login screen, session cookie, password-change, API-key tab (create / list-masked / revoke).
- [A6] Admin pane: user CRUD, request-audit table view, masked-key viewer, per-key IP allow-list.
- [A7] TOTP enrollment (QR via `pyotp` + `qrcode`).
- [A8] OIDC login (GitHub + Google) using `authlib`.
- [A9] KB scoping: `user_id` column on existing Mongo collection; admin-only flag for `global` flag; query path filters; ingest path stamps owner.
- [A10] Audit logging middleware — populate `request_audit` for every `/v1/*` call.
- [A11] Per-user rate-limit (token-bucket in SQLite) — hardening, optional in A.

### **Phase B — LanceDB migration**
- [B1] Add `lancedb` + `pyarrow` to deps. Build `provider/vector_store.py` with `LanceVectorStore` implementing the same interface as `RagService`.
- [B2] Create `kb_global/` and `kb_user/<id>/` lance datasets on first use.
- [B3] One-shot **migration script** `provider/scripts/migrate_mongo_to_lance.py`:
  - Reads every doc from Mongo collection.
  - Writes to `kb_user/<owner_id>/chunks.lance` (or `kb_global` for admin-flagged docs).
  - Verifies counts + spot-checks vector recall.
- [B4] FTS index + IVF_PQ index built once after migration.
- [B5] **Hybrid retrieval** (dense + sparse + RRF) replaces pure dense.
- [B6] **Reasoning prompt search**: register `kb.search(query, scope, k)` as a built-in tool; the model can invoke it mid-reasoning.
- [B7] Mongo becomes optional (`backend: "mongo" | "lance"` config). Existing Mongo path kept until B7 is verified in production.

### **Phase C — vLLM backend**
- [C1] `provider/backends/vllm.py`: thin client treating vLLM `/v1/...` as upstream.
- [C2] Add `backend: vllm` / `backend: llama_cpp` per-model in `models.yaml`.
- [C3] `compose.yml` with `vllm-chat`, `vllm-embed`, `vllm-vision` services.
- [C4] vLLM **continuous batching** = a single resident server can handle many concurrent users; remove the swap-to-load orchestration when `backend: vllm`.
- [C5] Idle-unload watchdog adapted to issue vLLM `/admin/...` (or `docker compose stop`) for VRAM eviction.
- [C6] Per-user concurrency cap (admin-configurable) so one user can't starve others.

### **Phase D — One-click deploy**
- [D1] `scripts/install.sh` (Linux/macOS): checks Docker + nvidia-container-toolkit, copies `.env.example` → `.env`, prompts for first admin password, issues an admin API key, runs `docker compose up -d`.
- [D2] `scripts/install.ps1` (Windows): same flow, checks Docker Desktop + WSL2 GPU support.
- [D3] `scripts/install_macos.sh`: Docker Desktop + Apple-Silicon path uses CPU-only vLLM image (limited but functional) or routes chat to llama.cpp fallback.
- [D4] `Makefile` targets: `make up`, `make down`, `make logs`, `make migrate`, `make backup`, `make rotate-keys`.
- [D5] Healthcheck + readiness endpoint — Compose `depends_on: condition: service_healthy`.
- [D6] Backup script: SQLite `.backup`, Lance dataset `cp -r`, daily cron.

## 8. Backwards compatibility

- Existing `runtime_config.json` is read once on first v2 boot and migrated into `config_kv`. The file is then renamed `runtime_config.json.migrated`.
- The MongoDB connection is opened only when `rag.backend == "mongo"`.
- No config file rewrites are destructive; every migration is idempotent (`schema_version` row in `config_kv`).

## 9. Security checklist

- [x] All passwords argon2id with pepper.
- [x] API keys stored hashed; full plaintext shown once.
- [x] Sessions: HttpOnly, Secure (when behind TLS), SameSite=Lax.
- [x] CSRF: state-changing routes require either `X-CSRF-Token` (matched against session) or `Authorization: Bearer`.
- [x] OAuth state nonce + PKCE.
- [x] Constant-time comparison for key hashes.
- [x] Master key (`PROVIDER_MASTER_KEY`) loaded from env, **never** committed; bootstrap script generates it.
- [x] CORS: only allow the configured origin in production.
- [x] Rate-limit on `/auth/login` (10 / 15min / IP).
- [x] OWASP-A01 (auth) / A02 (crypto) / A03 (injection — SQLite uses parameterized queries only) / A07 (id-and-auth) covered.

## 10. Outstanding decisions (will revisit before each phase starts)

- IVF_PQ vs HNSW for Lance ANN once corpus crosses 1M vectors.
- Whether to bundle a Postgres image instead of SQLite if a user reports lock contention >50 concurrent writers.
- Whether to host TOTP secrets in a separate SQLCipher DB if the master key is rotated frequently.

---

*This document is the source of truth. Each phase's PR description must link back to the corresponding §7 sub-task IDs.*
