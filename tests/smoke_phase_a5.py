"""Smoke test for A5: static UI assets, login HTML wiring, account.js endpoints."""
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
from provider import gateway
app = gateway.create_app(load_config())

from starlette.testclient import TestClient
c = TestClient(app)

for p in ["/ui/login.html", "/ui/account.js", "/ui/style.css", "/ui/index.html"]:
    r = c.get(p)
    print(p, r.status_code, len(r.content))
    assert r.status_code == 200, p

r = c.get("/auth/me-anonymous")
print("me-anon", r.json())
assert r.status_code == 200 and r.json()["bootstrap"] is True

r = c.post("/auth/bootstrap", json={"username": "root", "password": "correcthorse"})
assert r.status_code == 200
r = c.get("/auth/me")
assert r.status_code == 200 and r.json()["user"]["role"] == "admin"
r = c.get("/auth/admin/keys"); assert r.status_code == 200
r = c.get("/auth/users"); assert r.status_code == 200

html = pathlib.Path("provider/web/login.html").read_text(encoding="utf-8")
for needle in ["/auth/me-anonymous", "/auth/bootstrap", "/auth/login",
               "detectMode", "totp_required"]:
    assert needle in html, f"login.html missing {needle!r}"

js = pathlib.Path("provider/web/account.js").read_text(encoding="utf-8")
for needle in ["/auth/me", "/auth/keys", "/auth/totp/begin", "/auth/totp/finish",
               "/auth/recovery-codes", "/auth/admin/keys", "/auth/sessions",
               "/auth/users", "openAccountModal"]:
    assert needle in js, f"account.js missing {needle!r}"

idx = pathlib.Path("provider/web/index.html").read_text(encoding="utf-8")
assert '/ui/account.js' in idx, "index.html does not load account.js"

print("ALL GREEN")
