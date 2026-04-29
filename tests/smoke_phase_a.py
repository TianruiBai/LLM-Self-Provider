"""Throwaway smoke test for Phase A3+A4. Run from the repo root."""
import os
import pathlib
import tempfile

tmp = pathlib.Path(tempfile.mkdtemp())
os.environ["PROVIDER_AUTH_PEPPER"] = "a" * 64
os.environ["PROVIDER_MASTER_KEY"] = "b" * 64
os.environ["PROVIDER_DISABLE_BOOTSTRAP"] = "0"  # default: bootstrap allowed when 0 users

import provider.db as db
db._DB_PATH = tmp / "control.db"
db.init()

from provider.registry import load_config
from provider import gateway

cfg = load_config()
app = gateway.create_app(cfg)

from starlette.testclient import TestClient

c = TestClient(app)

r = c.get("/health")
print("health", r.status_code)
assert r.status_code == 200, r.text

# Bootstrap mode is active (no users) — protected routes pass through.
r = c.get("/v1/models")
print("v1/models bootstrap", r.status_code)
assert r.status_code != 401, r.text

# Probe me-anonymous
r = c.get("/auth/me-anonymous")
print("me-anon", r.status_code, r.json())
assert r.status_code == 200 and r.json()["bootstrap"] is True

# Create the first admin via /auth/bootstrap.
r = c.post("/auth/bootstrap", json={"username": "root", "password": "correcthorse", "email": "r@x.y"})
print("bootstrap", r.status_code, r.json().get("user", {}).get("username"))
assert r.status_code == 200, r.text

# Now bootstrap is closed.
r = c.post("/auth/bootstrap", json={"username": "x", "password": "correcthorse"})
print("bootstrap2", r.status_code)
assert r.status_code == 403

# Cookie is set on TestClient — /auth/me works.
r = c.get("/auth/me")
print("me", r.status_code, r.json())
assert r.status_code == 200 and r.json()["user"]["role"] == "admin"

# Forced unauth: clear cookies → /v1/* should now 401.
c.cookies.clear()
r = c.get("/v1/models")
print("v1/models anon (post-bootstrap)", r.status_code)
assert r.status_code == 401, r.text

# Login with password.
r = c.post("/auth/login", json={"username": "root", "password": "correcthorse"})
print("login", r.status_code)
assert r.status_code == 200 and r.json()["ok"] is True
r = c.get("/auth/me")
assert r.status_code == 200

# Wrong password → 401.
c2 = TestClient(app)
r = c2.post("/auth/login", json={"username": "root", "password": "WRONG"})
print("login bad", r.status_code)
assert r.status_code == 401

# Issue an API key on the logged-in client.
r = c.post("/auth/keys", json={"name": "cli"})
print("issue key:", r.status_code, r.json()["key"]["masked"])
plain = r.json()["plaintext"]
assert plain.startswith("sk-prov-")

# List keys
r = c.get("/auth/keys")
assert r.status_code == 200 and len(r.json()["keys"]) == 1

# Bearer-only flow: another client with no cookie.
c3 = TestClient(app)
r = c3.get("/v1/models", headers={"Authorization": f"Bearer {plain}"})
print("bearer", r.status_code)
assert r.status_code != 401

r = c3.get("/v1/models", headers={"Authorization": "Bearer sk-prov-totally-bogus-token"})
print("bad bearer", r.status_code)
assert r.status_code == 401

# Admin can list users + sessions
r = c.get("/auth/users")
assert r.status_code == 200 and len(r.json()["users"]) == 1
r = c.get("/auth/sessions")
assert r.status_code == 200

# Non-admin guard: create a regular user, log in as them, deny /auth/users.
r = c.post("/auth/users", json={"username": "alice", "password": "correcthorse2", "role": "user"})
assert r.status_code == 200
c4 = TestClient(app)
r = c4.post("/auth/login", json={"username": "alice", "password": "correcthorse2"})
assert r.status_code == 200
r = c4.get("/auth/users")
print("non-admin -> /auth/users:", r.status_code)
assert r.status_code == 403

# Logout
r = c.post("/auth/logout")
assert r.status_code == 200
r = c.get("/auth/me")
print("after logout:", r.status_code)
assert r.status_code == 401

print("ALL GREEN")
