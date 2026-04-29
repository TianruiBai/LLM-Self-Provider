"""Smoke test for the model-publish gating + LM Studio discovery.

Boots the gateway over a temp SQLite + temp Lance dir, registers two fake
chat models in `cfg.models`, and verifies:

    * /v1/models hides unpublished models from non-admin viewers.
    * /admin/models lists every model regardless of publish state.
    * /admin/models/<id>/publish flips the visibility.
    * Non-admin POST to /v1/chat/completions on an unpublished model
      returns 403, while published model attempts get past the guard.
    * LM Studio discovery walks `<root>/<pub>/<model>/*.gguf` and registers
      `lmstudio/<pub>/<model>` ids without exposing them by default.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile

WORK = pathlib.Path(tempfile.mkdtemp(prefix="publish_smoke_"))
os.environ["PROVIDER_AUTH_PEPPER"] = "0" * 64
os.environ["PROVIDER_MASTER_KEY"] = "0" * 64
os.environ["PROVIDER_AUTH_DEV_ALLOW"] = "0"
# Point the SQLite control DB at a fresh file.
os.environ["PROVIDER_DATA_DIR"] = str(WORK)


# Re-route db.db_path() at import time.
from provider import db as _db

def _patched_data_dir():
    p = WORK / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p

_db._data_dir = _patched_data_dir  # type: ignore[assignment]
_db._DB_PATH = None

# 1. Discovery: build a fake LM Studio tree.
LM_ROOT = WORK / "lmstudio_models"
(LM_ROOT / "qwen" / "qwen3-9b-instruct").mkdir(parents=True)
(LM_ROOT / "qwen" / "qwen3-9b-instruct" / "qwen3-9b.Q4_K_M.gguf").write_bytes(b"\x00")
(LM_ROOT / "google" / "gemma-3-4b-it").mkdir(parents=True)
(LM_ROOT / "google" / "gemma-3-4b-it" / "gemma-3-4b-it.Q4.gguf").write_bytes(b"\x00")
(LM_ROOT / "google" / "gemma-3-4b-it" / "mmproj-F16.gguf").write_bytes(b"\x00")

from provider.registry import _discover_lmstudio  # noqa: E402

discovered = _discover_lmstudio(str(LM_ROOT))
ids = {m.id for m in discovered}
assert "lmstudio/qwen/qwen3-9b-instruct" in ids, ids
assert "lmstudio/google/gemma-3-4b-it" in ids, ids
gemma = next(m for m in discovered if m.id == "lmstudio/google/gemma-3-4b-it")
assert gemma.kind == "vision", "mmproj-bearing folder should be tagged vision"
assert gemma.mmproj and gemma.mmproj.endswith("mmproj-F16.gguf")
print(f"[1/4] LM Studio discovery: found {len(discovered)} models")

# 2. Publish gating via the FastAPI app.
from fastapi.testclient import TestClient  # noqa: E402

from provider.registry import (  # noqa: E402
    GatewayConfig, ModelConfig, ProviderConfig, RagConfig, ServerConfig,
)
from provider import gateway as _gw  # noqa: E402
from provider import auth as _auth  # noqa: E402

cfg = ProviderConfig(
    server=ServerConfig(llama_server_bin=""),
    gateway=GatewayConfig(),
    rag=RagConfig(backend="lance", lance_dir=str(WORK / "lance"), embedding_dim=8),
    models=[
        ModelConfig(
            id="public-model", kind="chat", path="/dev/null",
            backend="vllm", endpoint="http://fake:8000",
        ),
        ModelConfig(
            id="hidden-model", kind="chat", path="/dev/null",
            backend="vllm", endpoint="http://fake:8000",
        ),
    ],
)
# Bypass real config loading.
_gw.load_config = lambda *a, **kw: cfg  # type: ignore[assignment]

app = _gw.create_app()
client = TestClient(app)

# Bootstrap an admin and a regular user.
admin = _auth.create_user("admin", password="adminpass!", role="admin")
plain_admin, _ = _auth.create_api_key(admin.id, "admin-key")
user = _auth.create_user("alice", password="userpass!", role="user")
plain_user, _ = _auth.create_api_key(user.id, "alice-key")

H_ADMIN = {"Authorization": f"Bearer {plain_admin}"}
H_USER = {"Authorization": f"Bearer {plain_user}"}

# Nothing is published yet — non-admin sees 0, admin sees 2.
r = client.get("/v1/models", headers=H_USER).json()
assert r["data"] == [], f"unpublished models leaked to user: {r}"
r = client.get("/v1/models", headers=H_ADMIN).json()
assert {m["id"] for m in r["data"]} == {"public-model", "hidden-model"}, r
print("[2/4] /v1/models hides unpublished by default")

# Admin publishes one.
r = client.post(
    "/admin/models/public-model/publish",
    json={"label": "GPT-Public"},
    headers=H_ADMIN,
)
assert r.status_code == 200, r.text
assert r.json() == {"model": "public-model", "published": True, "label": "GPT-Public"}

r = client.get("/v1/models", headers=H_USER).json()
assert {m["id"] for m in r["data"]} == {"public-model"}, r
assert r["data"][0]["label"] == "GPT-Public"
print("[3/4] publish makes the model visible to users")

# Non-admin chat against unpublished model -> 403.
r = client.post(
    "/v1/chat/completions",
    json={"model": "hidden-model", "messages": [{"role": "user", "content": "hi"}]},
    headers=H_USER,
)
assert r.status_code == 403, f"expected 403 for hidden, got {r.status_code} {r.text}"
# Non-admin chat against published model -> not 403 (will fail upstream
# because the fake endpoint isn't real, but the publish guard let it through).
r = client.post(
    "/v1/chat/completions",
    json={"model": "public-model", "messages": [{"role": "user", "content": "hi"}]},
    headers=H_USER,
)
assert r.status_code != 403, r.text

# Admin can still call hidden ones.
r = client.post(
    "/v1/chat/completions",
    json={"model": "hidden-model", "messages": [{"role": "user", "content": "hi"}]},
    headers=H_ADMIN,
)
assert r.status_code != 403, r.text
print("[4/4] /v1/chat/completions enforces publish status for users")

# Cleanup.
shutil.rmtree(WORK, ignore_errors=True)
print("ALL GREEN")
