"""Smoke test for A6+A10: audit middleware writes rows; admin endpoints read them."""
import os
import pathlib
import tempfile

tmp = pathlib.Path(tempfile.mkdtemp())
os.environ["PROVIDER_AUTH_PEPPER"] = "a" * 64
os.environ["PROVIDER_MASTER_KEY"] = "b" * 64

import provider.db as db
db._DB_PATH = tmp / "control.db"
db.init()

from provider.registry import load_config
from provider import gateway, audit
app = gateway.create_app(load_config())

from starlette.testclient import TestClient
c = TestClient(app)

# Bootstrap admin and login.
r = c.post("/auth/bootstrap", json={"username": "root", "password": "correcthorse"})
assert r.status_code == 200, r.text

# Some traffic to audit.
c.get("/auth/me")
c.get("/auth/users")
c.get("/auth/admin/keys")
r = c.post("/auth/keys", json={"name": "smoke"})
assert r.status_code == 200
plain = r.json()["plaintext"]

# /v1/models with bearer (will likely 200 or whatever; we don't care, just audited).
c.get("/v1/models", headers={"Authorization": f"Bearer {plain}"})

# Bad bearer (still expected to be audited as a 401? No — AuthMiddleware short-circuits 401.)
c.get("/v1/models", headers={"Authorization": "Bearer sk-prov-bogus"})

# Read audit.
r = c.get("/auth/admin/audit?limit=50")
assert r.status_code == 200, r.text
j = r.json()
print("rows", len(j["rows"]), "total", j["total"])
assert j["total"] >= 4, f"too few audit rows: {j['total']}"

# Filter by path prefix.
r = c.get("/auth/admin/audit?path_prefix=/auth/keys")
assert r.status_code == 200
keys_rows = r.json()["rows"]
assert any(x["path"].startswith("/auth/keys") for x in keys_rows)
assert all(x["path"].startswith("/auth/keys") for x in keys_rows), "filter leak"

# At least one row should have api_key_id from the bearer call.
r = c.get("/auth/admin/audit?path_prefix=/v1/models")
rows = r.json()["rows"]
print("v1/models rows:", [(x["status"], x.get("key_prefix"), x["ip"]) for x in rows])
assert any(x["api_key_id"] is not None for x in rows), "api_key_id never recorded"

# Summary endpoint.
r = c.get("/auth/admin/audit/summary")
assert r.status_code == 200, r.text
s = r.json()
assert s["by_status"], "summary empty"
print("by_status:", s["by_status"][:5])

# Audit must be admin-only.
c.post("/auth/users", json={"username":"alice","password":"alicepass1","role":"user"})
# Login as alice.
c2 = TestClient(app)
r = c2.post("/auth/login", json={"username":"alice","password":"alicepass1"})
assert r.status_code == 200, r.text
r = c2.get("/auth/admin/audit")
assert r.status_code == 403, f"non-admin should be 403, got {r.status_code}"

# Trim helper.
deleted = audit.trim_audit(retain=5)
print("trimmed", deleted)
left = db.fetchone("SELECT COUNT(*) AS n FROM request_audit")["n"]
assert left <= 5 + 5  # +slack for new audited reads from this very test

print("ALL GREEN")
