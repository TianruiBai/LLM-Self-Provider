"""Smoke test for A8: OIDC sign-in (state + callback flow).

Mocks the network round-trips to GitHub/Google so the test runs without
external connectivity.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

tmp = pathlib.Path(tempfile.mkdtemp())
os.environ["PROVIDER_AUTH_PEPPER"] = "a" * 64
os.environ["PROVIDER_MASTER_KEY"] = "b" * 64

# Configure two providers via env vars so load_providers() picks them up.
os.environ["PROVIDER_OIDC_GITHUB_CLIENT_ID"] = "gh-id"
os.environ["PROVIDER_OIDC_GITHUB_CLIENT_SECRET"] = "gh-secret"
os.environ["PROVIDER_OIDC_GOOGLE_CLIENT_ID"] = "go-id"
os.environ["PROVIDER_OIDC_GOOGLE_CLIENT_SECRET"] = "go-secret"

import provider.db as db
db._DB_PATH = tmp / "control.db"
db.init()

from provider import oidc, rag as ragmod
from provider.rag import RagService

# Skip Mongo
RagService.__init__ = lambda self, cfg, embedder_base_url_provider: setattr(self, "cfg", cfg) or None  # type: ignore
async def _noop(self):  # pragma: no cover
    return None
RagService.startup = _noop  # type: ignore
RagService.shutdown = _noop  # type: ignore

oidc.reload_providers()
provs = oidc.load_providers()
assert "github" in provs and "google" in provs, list(provs)
print("[1/6] providers loaded:", list(provs))

# ----------------- _scope of state -----------------

state, verifier, challenge = oidc.create_state("google", "/ui/", with_pkce=True)
assert verifier and challenge and len(state) > 30
row = oidc.consume_state(state)
assert row and row["provider"] == "google" and row["code_verifier"] == verifier and row["redirect_to"] == "/ui/"
# Single-use:
assert oidc.consume_state(state) is None
# Expired state is rejected:
state2, _, _ = oidc.create_state("github", None, with_pkce=False)
db.execute("UPDATE oauth_states SET expires_at=? WHERE state=?", (1, state2))
assert oidc.consume_state(state2) is None
print("[2/6] state lifecycle OK")

# ----------------- gateway / start endpoint -----------------

from provider.registry import load_config
from provider import gateway, oidc_routes
app = gateway.create_app(load_config())
from starlette.testclient import TestClient
c = TestClient(app)

r = c.get("/auth/oidc/providers")
assert r.status_code == 200, r.text
names = sorted(p["name"] for p in r.json()["providers"])
assert names == ["github", "google"], names

# /start should 302 to the provider's authorize URL with state in params.
r = c.get("/auth/oidc/google/start?next=/ui/", follow_redirects=False)
assert r.status_code == 302, r.status_code
loc = r.headers["location"]
assert loc.startswith("https://accounts.google.com/o/oauth2/v2/auth")
assert "state=" in loc and "code_challenge=" in loc
print("[3/6] /start redirects with state + PKCE")

# Unknown provider -> 404
r = c.get("/auth/oidc/nope/start", follow_redirects=False)
assert r.status_code == 404, r.status_code

# ----------------- callback (mocked exchange) -----------------

# Patch exchange_and_fetch to avoid real network.
async def fake_exchange(provider, code, redirect_uri, code_verifier):
    assert provider.name in ("github", "google")
    assert code == "the-code"
    return {
        "subject": f"{provider.name}:42",
        "email": "alice@example.com",
        "name": "Alice",
        "login": "alice",
        "raw": {"login": "alice", "name": "Alice", "email": "alice@example.com"},
    }


oidc_routes.oidc.exchange_and_fetch = fake_exchange  # type: ignore[attr-defined]

# Call /start to get a fresh state, then hand it back to /callback.
r = c.get("/auth/oidc/github/start?next=/ui/somewhere", follow_redirects=False)
loc = r.headers["location"]
from urllib.parse import urlparse, parse_qs
state = parse_qs(urlparse(loc).query)["state"][0]

r = c.get(f"/auth/oidc/github/callback?code=the-code&state={state}", follow_redirects=False)
assert r.status_code == 302, r.text
assert r.headers["location"] == "/ui/somewhere", r.headers["location"]
# Session cookie set:
assert "PROV_SID" in r.cookies, r.cookies
print("[4/6] callback creates session + redirects to safe `next`")

# /me confirms the new user.
me = c.get("/auth/me")
assert me.status_code == 200, me.text
user = me.json()["user"]
assert user["username"] == "alice", user
assert user["oidc_subject"] == "github:42", user
assert user["role"] == "user"
print("[5/6] new OIDC user persisted with oidc_subject")

# Second sign-in with the SAME subject must not create a duplicate user.
c2 = TestClient(app)
r = c2.get("/auth/oidc/github/start", follow_redirects=False)
state2 = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
r = c2.get(f"/auth/oidc/github/callback?code=the-code&state={state2}", follow_redirects=False)
assert r.status_code == 302, r.text
me2 = c2.get("/auth/me").json()["user"]
assert me2["id"] == user["id"], (me2, user)

# Replayed state must fail.
r = c.get(f"/auth/oidc/github/callback?code=the-code&state={state2}", follow_redirects=False)
assert r.status_code == 400, r.status_code

# Wrong-provider state must fail.
state3, _, _ = oidc.create_state("google", None, with_pkce=True)
r = c.get(f"/auth/oidc/github/callback?code=the-code&state={state3}", follow_redirects=False)
assert r.status_code == 400, r.status_code

# Provider error parameter is reported as 400.
r = c.get("/auth/oidc/github/callback?error=access_denied", follow_redirects=False)
assert r.status_code == 400

print("[6/6] reuse, replay, mismatch, and error all handled")
print("ALL GREEN")
