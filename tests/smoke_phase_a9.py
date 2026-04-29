"""Smoke test for A9: KB scoping (per-user + global).

Verifies the gateway plumbs ``viewer_id`` / ``is_admin`` / ``scope`` /
``owner_id`` correctly through to ``RagService`` and that role-based
restrictions on the global KB are enforced.

Mongo is **not** required: ``RagService`` methods are monkey-patched with
in-memory stubs that capture every call.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

tmp = pathlib.Path(tempfile.mkdtemp())
os.environ["PROVIDER_AUTH_PEPPER"] = "a" * 64
os.environ["PROVIDER_MASTER_KEY"] = "b" * 64

import provider.db as db
db._DB_PATH = tmp / "control.db"
db.init()

from provider import rag as ragmod
from provider.rag import RagService, _scope_match, SCOPE_GLOBAL, SCOPE_USER

# ----------------- _scope_match unit checks -----------------

# anonymous viewer -> only global (legacy-or-stamped)
m = _scope_match(None, is_admin=False, scopes=None)
assert m == {"$or": [{"scope": "global"}, {"scope": {"$exists": False}}]}, m

# regular user -> own user docs ∪ global
m = _scope_match(7, is_admin=False, scopes=None)
assert m["$or"][0] == {"$or": [{"scope": "global"}, {"scope": {"$exists": False}}]}
assert m["$or"][1] == {"scope": "user", "owner_id": 7}

# regular user with scope=['user'] -> only own
m = _scope_match(7, is_admin=False, scopes=["user"])
assert m == {"scope": "user", "owner_id": 7}, m

# admin sees everything by default
m = _scope_match(1, is_admin=True, scopes=None)
assert m == {}, m

# admin asking only global
m = _scope_match(1, is_admin=True, scopes=["global"])
assert m == {"$or": [{"scope": "global"}, {"scope": {"$exists": False}}]}, m

print("[1/6] _scope_match OK")

# ----------------- gateway plumbing checks -----------------

# Capture all RagService calls.
calls: list[dict] = []


async def fake_ingest(self, documents, *, source="api", tags=None, owner_id=None, scope=SCOPE_GLOBAL):
    docs = list(documents)
    calls.append({"op": "ingest", "n": len(docs), "source": source, "tags": list(tags or []),
                  "owner_id": owner_id, "scope": scope})
    return {"inserted": len(docs), "scope": scope, "owner_id": owner_id}


async def fake_query(self, text, *, top_k=None, source=None, tags=None,
                     viewer_id=None, is_admin=False, scopes=None):
    calls.append({"op": "query", "text": text, "viewer_id": viewer_id,
                  "is_admin": is_admin, "scopes": scopes})
    return []


async def fake_stats(self, *, viewer_id=None, is_admin=False, scopes=None):
    calls.append({"op": "stats", "viewer_id": viewer_id, "is_admin": is_admin})
    return {"total_chunks": 0, "sources": []}


async def fake_list_documents(self, *, source=None, tag=None, limit=200,
                              viewer_id=None, is_admin=False, scopes=None):
    calls.append({"op": "list", "viewer_id": viewer_id, "is_admin": is_admin, "scopes": scopes})
    return []


async def fake_delete(self, *, chunk_ids=None, source=None, doc_id=None, tag=None,
                      viewer_id=None, is_admin=False, scopes=None):
    calls.append({"op": "delete", "viewer_id": viewer_id, "is_admin": is_admin,
                  "doc_id": doc_id})
    return {"deleted": 0}


async def fake_get_chunks(self, chunk_ids, *, viewer_id=None, is_admin=False, scopes=None):
    calls.append({"op": "chunks", "viewer_id": viewer_id, "is_admin": is_admin,
                  "ids": list(chunk_ids)})
    return []


async def fake_startup(self):  # don't touch Mongo
    return None


async def fake_shutdown(self):
    return None


RagService.ingest = fake_ingest
RagService.query = fake_query
RagService.stats = fake_stats
RagService.list_documents = fake_list_documents
RagService.delete = fake_delete
RagService.get_chunks = fake_get_chunks
RagService.startup = fake_startup
RagService.shutdown = fake_shutdown
# Skip Mongo client construction
RagService.__init__ = lambda self, cfg, embedder_base_url_provider: setattr(self, "cfg", cfg) or None  # type: ignore

from provider.registry import load_config
from provider import gateway

app = gateway.create_app(load_config())

from starlette.testclient import TestClient
c = TestClient(app)

# Bootstrap admin (root) and create a regular user (alice).
r = c.post("/auth/bootstrap", json={"username": "root", "password": "correcthorse"})
assert r.status_code == 200, r.text
admin_id = r.json()["user"]["id"]

r = c.post("/auth/users", json={"username": "alice", "password": "tr0ub4dor3"})
assert r.status_code == 200, r.text
alice_id = r.json()["user"]["id"]

# Fresh client = fresh session for alice.
ca = TestClient(app)
r = ca.post("/auth/login", json={"username": "alice", "password": "tr0ub4dor3"})
assert r.status_code == 200, r.text

print("[2/6] auth ready (admin id=%d, alice id=%d)" % (admin_id, alice_id))

# 1. alice ingest default -> scope='user', owner_id=alice_id
calls.clear()
r = ca.post("/rag/ingest", json={"documents": [{"text": "hello", "metadata": {}}],
                                 "source": "src", "tags": ["t1"]})
assert r.status_code == 200, r.text
assert calls[0]["scope"] == "user", calls[0]
assert calls[0]["owner_id"] == alice_id, calls[0]
print("[3/6] alice default ingest -> scope=user, owner_id=alice OK")

# 2. alice ingest with scope='global' -> 403
r = ca.post("/rag/ingest", json={"documents": [{"text": "x", "metadata": {}}],
                                 "scope": "global"})
assert r.status_code == 403, r.status_code
print("[4/6] alice scope=global -> 403 OK")

# 3. admin ingest with scope='global' -> owner_id=None, scope='global'
calls.clear()
r = c.post("/rag/ingest", json={"documents": [{"text": "policy", "metadata": {}}],
                                "source": "policies", "scope": "global"})
assert r.status_code == 200, r.text
assert calls[0]["scope"] == "global", calls[0]
assert calls[0]["owner_id"] is None, calls[0]
print("[5/6] admin scope=global ingest OK")

# 4. /rag/query passes viewer_id=alice + is_admin=False
calls.clear()
r = ca.post("/rag/query", json={"text": "hello"})
assert r.status_code == 200, r.text
assert calls[0]["viewer_id"] == alice_id and calls[0]["is_admin"] is False, calls[0]

# 5. /rag/stats / list / delete carry viewer + is_admin correctly
calls.clear()
ca.get("/rag/stats")
ca.get("/rag/documents")
ca.request("DELETE", "/rag/documents", json={"doc_id": "x"})
assert all(call["viewer_id"] == alice_id and call["is_admin"] is False for call in calls), calls

calls.clear()
c.get("/rag/stats")
c.get("/rag/documents?scope=global,user")
assert calls[0]["is_admin"] is True, calls[0]
assert calls[1]["scopes"] == ["global", "user"], calls[1]

# 6. Bad scope is rejected
r = ca.post("/rag/ingest", json={"documents": [{"text": "x"}], "scope": "bogus"})
assert r.status_code == 400, r.status_code

print("[6/6] viewer_id / is_admin / scopes plumbing OK")

print("ALL GREEN")
