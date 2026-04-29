"""Smoke test for A11: per-user + per-IP rate limit."""
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
from provider import gateway, ratelimit
app = gateway.create_app(load_config())

from starlette.testclient import TestClient
c = TestClient(app)

# 1) Bootstrap admin.
r = c.post("/auth/bootstrap", json={"username": "root", "password": "correcthorse"})
assert r.status_code == 200, r.text

# 2) Tighten user.v1 to 3 / 60s, then exhaust.
r = c.put("/auth/admin/ratelimit/user.v1",
          json={"capacity": 3, "refill_per_s": 0.05})
assert r.status_code == 200, r.text

statuses = []
for _ in range(6):
    statuses.append(c.get("/v1/models").status_code)
print("statuses", statuses)
# First 3 should not be 429; 4th onward should be 429.
assert statuses[:3].count(429) == 0, statuses
assert 429 in statuses[3:], statuses

# Verify Retry-After header.
r = c.get("/v1/models")
assert r.status_code == 429
assert int(r.headers.get("retry-after", "0")) >= 1
body = r.json()
assert "retry_after" in body and body["capacity"] == 3

# 3) Reset the bucket via admin endpoint.
r = c.post("/auth/admin/ratelimit/reset", json={"bucket": "user:1:v1"})
assert r.status_code == 200
assert c.get("/v1/models").status_code != 429

# 4) Login throttle (anonymous IP). Tighten ip.login to 2.
r = c.put("/auth/admin/ratelimit/ip.login",
          json={"capacity": 2, "refill_per_s": 0.001})
assert r.status_code == 200
c2 = TestClient(app)
attempts = [c2.post("/auth/login", json={"username": "nope", "password": "x"}).status_code
            for _ in range(5)]
print("login attempts", attempts)
assert 429 in attempts, attempts

# 5) GET /admin/ratelimit lists overrides.
r = c.get("/auth/admin/ratelimit")
assert r.status_code == 200
j = r.json()
assert "user.v1" in j["overrides"] and "ip.login" in j["overrides"]

# 6) Non-admin can't write.
c.post("/auth/admin/ratelimit/reset", json={"bucket": "ip:testclient:login"})
# Loosen ip.login so alice can log in.
c.put("/auth/admin/ratelimit/ip.login", json={"capacity": 100, "refill_per_s": 1})
c.post("/auth/users", json={"username":"alice","password":"alicepass1","role":"user"})
c3 = TestClient(app)
r = c3.post("/auth/login", json={"username":"alice","password":"alicepass1"})
assert r.status_code == 200, f"alice login: {r.status_code} {r.text}"
r = c3.put("/auth/admin/ratelimit/user.v1", json={"capacity":1000,"refill_per_s":1000})
assert r.status_code == 403, r.status_code

print("ALL GREEN")
