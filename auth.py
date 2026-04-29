"""Authentication primitives: passwords, sessions, API keys, TOTP, recovery codes.

This module is **synchronous** and DB-only. HTTP wiring lives in
``auth_deps.py`` (FastAPI dependencies) and the gateway's auth routes.

Storage rules (see ARCHITECTURE.md §4):

* Passwords: argon2id with a 32-byte server-side pepper from
  ``PROVIDER_AUTH_PEPPER`` (HMAC-SHA256 prefix before hashing). Pepper is
  *required* in production; if absent in dev the module logs a warning and
  uses an ephemeral one (sessions and key hashes survive across restarts,
  passwords do not).
* API keys: only the SHA-256 of the plaintext is stored. The plaintext is
  returned **once** at creation time. Lookups query by ``key_hash``.
* Sessions: 256 bits of randomness, opaque to the client. Sliding window
  refresh on each request.
* TOTP secrets: encrypted at rest with libsodium SecretBox using a
  per-install master key from ``PROVIDER_MASTER_KEY``.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass
from typing import Iterable, Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHash

from . import db

log = logging.getLogger("provider.auth")


# --------------------------------------------------------------- constants

PASSWORD_HASHER = PasswordHasher(
    time_cost=3, memory_cost=64 * 1024, parallelism=2, hash_len=32, salt_len=16
)

API_KEY_PREFIX = "sk-prov-"
API_KEY_RANDOM_BYTES = 18                # → 24 chars base64url; total ~32 chars
SESSION_BYTES = 32                       # 256 bits
SESSION_TTL_S = 7 * 24 * 3600
RECOVERY_CODE_COUNT = 10


# ------------------------------------------------------------- master/pepper

def _pepper() -> bytes:
    raw = os.environ.get("PROVIDER_AUTH_PEPPER")
    if raw:
        try:
            return bytes.fromhex(raw) if len(raw) in (32, 64) else raw.encode("utf-8")
        except ValueError:
            return raw.encode("utf-8")
    log.warning(
        "PROVIDER_AUTH_PEPPER is not set — using an in-memory pepper. "
        "Existing password hashes will fail to verify after a restart.",
    )
    if not hasattr(_pepper, "_eph"):
        _pepper._eph = secrets.token_bytes(32)  # type: ignore[attr-defined]
    return _pepper._eph                          # type: ignore[attr-defined,return-value]


def _master_key() -> bytes:
    raw = os.environ.get("PROVIDER_MASTER_KEY")
    if raw:
        if len(raw) == 64:
            return bytes.fromhex(raw)
        return hashlib.sha256(raw.encode("utf-8")).digest()
    log.warning(
        "PROVIDER_MASTER_KEY is not set — TOTP secrets will not survive restarts.",
    )
    if not hasattr(_master_key, "_eph"):
        _master_key._eph = secrets.token_bytes(32)  # type: ignore[attr-defined]
    return _master_key._eph                          # type: ignore[attr-defined,return-value]


# -------------------------------------------------------------- passwords

def _peppered(password: str) -> str:
    """Return password mixed with the server pepper (HMAC-SHA256, hex)."""
    return hmac.new(_pepper(), password.encode("utf-8"), hashlib.sha256).hexdigest()


def hash_password(password: str) -> str:
    if not password or len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    return PASSWORD_HASHER.hash(_peppered(password))


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    try:
        PASSWORD_HASHER.verify(stored_hash, _peppered(password))
    except (VerifyMismatchError, InvalidHash):
        return False
    return True


def needs_rehash(stored_hash: str) -> bool:
    try:
        return PASSWORD_HASHER.check_needs_rehash(stored_hash)
    except Exception:
        return False


# ------------------------------------------------------------------ users

@dataclass
class User:
    id: int
    username: str
    email: Optional[str]
    role: str
    is_active: bool
    totp_enabled: bool
    oidc_subject: Optional[str]

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @classmethod
    def from_row(cls, row) -> "User":
        return cls(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            role=row["role"],
            is_active=bool(row["is_active"]),
            totp_enabled=bool(row["totp_enabled"]),
            oidc_subject=row["oidc_subject"],
        )


def create_user(
    username: str,
    password: Optional[str],
    *,
    email: Optional[str] = None,
    role: str = "user",
    oidc_subject: Optional[str] = None,
) -> User:
    if role not in ("user", "admin"):
        raise ValueError("role must be 'user' or 'admin'")
    if not username or not username.strip():
        raise ValueError("username required")
    pw_hash = hash_password(password) if password else None
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO users (username, email, password_hash, role, oidc_subject, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (username.strip(), email, pw_hash, role, oidc_subject, db.now_ts()),
        )
        uid = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    row = db.fetchone("SELECT * FROM users WHERE id=?", (uid,))
    assert row is not None
    return User.from_row(row)


def get_user_by_id(user_id: int) -> Optional[User]:
    row = db.fetchone("SELECT * FROM users WHERE id=?", (user_id,))
    return User.from_row(row) if row else None


def get_user_by_username(username: str) -> Optional[User]:
    row = db.fetchone(
        "SELECT * FROM users WHERE username=? COLLATE NOCASE", (username.strip(),)
    )
    return User.from_row(row) if row else None


def get_user_by_oidc(subject: str) -> Optional[User]:
    row = db.fetchone("SELECT * FROM users WHERE oidc_subject=?", (subject,))
    return User.from_row(row) if row else None


def authenticate(username: str, password: str) -> Optional[User]:
    row = db.fetchone(
        "SELECT * FROM users WHERE username=? COLLATE NOCASE AND is_active=1",
        (username.strip(),),
    )
    if not row:
        # constant-time-ish: still verify against a dummy hash
        verify_password(password, PASSWORD_HASHER.hash("x" * 16))
        return None
    if not verify_password(password, row["password_hash"] or ""):
        return None
    if needs_rehash(row["password_hash"] or ""):
        new_hash = hash_password(password)
        db.execute("UPDATE users SET password_hash=? WHERE id=?", (new_hash, row["id"]))
    return User.from_row(row)


def set_password(user_id: int, new_password: str) -> None:
    db.execute(
        "UPDATE users SET password_hash=? WHERE id=?",
        (hash_password(new_password), user_id),
    )


def set_role(user_id: int, role: str) -> None:
    if role not in ("user", "admin"):
        raise ValueError("role must be 'user' or 'admin'")
    db.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))


def set_active(user_id: int, active: bool) -> None:
    db.execute("UPDATE users SET is_active=? WHERE id=?", (1 if active else 0, user_id))


def list_users() -> list[User]:
    return [User.from_row(r) for r in db.fetchall("SELECT * FROM users ORDER BY id")]


def admin_count() -> int:
    row = db.fetchone(
        "SELECT COUNT(*) AS n FROM users WHERE role='admin' AND is_active=1"
    )
    return int(row["n"]) if row else 0


def record_login(user_id: int, ip: Optional[str]) -> None:
    db.execute(
        "UPDATE users SET last_login_at=?, last_login_ip=? WHERE id=?",
        (db.now_ts(), ip, user_id),
    )


# --------------------------------------------------------------- sessions

@dataclass
class Session:
    id: str
    user_id: int
    expires_at: int


def create_session(user_id: int, *, ip: Optional[str], user_agent: Optional[str]) -> Session:
    sid = secrets.token_urlsafe(SESSION_BYTES)
    now = db.now_ts()
    exp = now + SESSION_TTL_S
    db.execute(
        "INSERT INTO sessions (id, user_id, created_at, expires_at, ip, user_agent) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (sid, user_id, now, exp, ip, user_agent),
    )
    return Session(id=sid, user_id=user_id, expires_at=exp)


def get_session(sid: str) -> Optional[Session]:
    if not sid:
        return None
    row = db.fetchone(
        "SELECT id, user_id, expires_at FROM sessions WHERE id=?", (sid,)
    )
    if not row:
        return None
    if row["expires_at"] < db.now_ts():
        delete_session(sid)
        return None
    return Session(id=row["id"], user_id=row["user_id"], expires_at=row["expires_at"])


def touch_session(sid: str) -> None:
    """Sliding window: extend expiry on each authenticated request."""
    db.execute(
        "UPDATE sessions SET expires_at=? WHERE id=?",
        (db.now_ts() + SESSION_TTL_S, sid),
    )


def delete_session(sid: str) -> None:
    db.execute("DELETE FROM sessions WHERE id=?", (sid,))


def purge_expired_sessions() -> int:
    cur = db.execute("DELETE FROM sessions WHERE expires_at < ?", (db.now_ts(),))
    return cur.rowcount


# --------------------------------------------------------------- API keys

@dataclass
class ApiKey:
    id: int
    user_id: int
    name: str
    key_prefix: str          # for display, e.g. 'sk-prov-XYZABC12'
    masked: str              # 'sk-prov-XYZABC12…'
    ip_allowlist: Optional[str]
    created_at: int
    last_used_at: Optional[int]
    last_used_ip: Optional[str]
    expires_at: Optional[int]
    revoked_at: Optional[int]


def _hash_key(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def _generate_key() -> tuple[str, str, str]:
    """Return (plaintext, prefix, hash). Prefix is the first 8 chars after the
    fixed ``sk-prov-`` namespace, used for display."""
    rand = base64.urlsafe_b64encode(secrets.token_bytes(API_KEY_RANDOM_BYTES))
    rand_s = rand.decode("ascii").rstrip("=")
    plain = f"{API_KEY_PREFIX}{rand_s}"
    prefix = rand_s[:8]
    return plain, prefix, _hash_key(plain)


def create_api_key(
    user_id: int,
    name: str,
    *,
    ip_allowlist: Optional[str] = None,
    expires_at: Optional[int] = None,
) -> tuple[str, ApiKey]:
    plain, prefix, kh = _generate_key()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO api_keys (user_id, name, key_prefix, key_hash, ip_allowlist, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, name.strip() or "key", prefix, kh, ip_allowlist, db.now_ts(), expires_at),
        )
        kid = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
    row = db.fetchone("SELECT * FROM api_keys WHERE id=?", (kid,))
    assert row is not None
    return plain, _api_key_from_row(row)


def _api_key_from_row(row) -> ApiKey:
    masked = f"{API_KEY_PREFIX}{row['key_prefix']}…"
    return ApiKey(
        id=row["id"],
        user_id=row["user_id"],
        name=row["name"],
        key_prefix=row["key_prefix"],
        masked=masked,
        ip_allowlist=row["ip_allowlist"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
        last_used_ip=row["last_used_ip"],
        expires_at=row["expires_at"],
        revoked_at=row["revoked_at"],
    )


def list_api_keys(user_id: int) -> list[ApiKey]:
    rows = db.fetchall(
        "SELECT * FROM api_keys WHERE user_id=? ORDER BY id DESC", (user_id,)
    )
    return [_api_key_from_row(r) for r in rows]


def list_all_api_keys() -> list[ApiKey]:
    rows = db.fetchall("SELECT * FROM api_keys ORDER BY id DESC")
    return [_api_key_from_row(r) for r in rows]


def revoke_api_key(key_id: int, *, user_id: Optional[int] = None) -> bool:
    """Revoke a key. If ``user_id`` is given, only revoke when it belongs to
    them (admin callers pass ``None`` to bypass the ownership check)."""
    sql = "UPDATE api_keys SET revoked_at=? WHERE id=? AND revoked_at IS NULL"
    params: list = [db.now_ts(), key_id]
    if user_id is not None:
        sql += " AND user_id=?"
        params.append(user_id)
    cur = db.execute(sql, params)
    return cur.rowcount > 0


def update_api_key_ip_allowlist(key_id: int, ip_allowlist: Optional[str]) -> None:
    db.execute(
        "UPDATE api_keys SET ip_allowlist=? WHERE id=?",
        (ip_allowlist, key_id),
    )


def verify_api_key(plain: str) -> Optional[ApiKey]:
    if not plain or not plain.startswith(API_KEY_PREFIX):
        return None
    row = db.fetchone(
        "SELECT * FROM api_keys WHERE key_hash=?", (_hash_key(plain),)
    )
    if not row:
        return None
    if row["revoked_at"]:
        return None
    if row["expires_at"] and row["expires_at"] < db.now_ts():
        return None
    return _api_key_from_row(row)


def touch_api_key(key_id: int, *, ip: Optional[str]) -> None:
    db.execute(
        "UPDATE api_keys SET last_used_at=?, last_used_ip=? WHERE id=?",
        (db.now_ts(), ip, key_id),
    )


# ----------------------------------------------------------------- TOTP

def _encrypt_totp(secret_b32: str) -> str:
    """Encrypt a TOTP secret with the master key. Returns base64(nonce|ct)."""
    try:
        from nacl.secret import SecretBox  # type: ignore
    except ImportError:
        raise RuntimeError(
            "PyNaCl is required for TOTP support. Install with `pip install pynacl`."
        )
    box = SecretBox(_master_key())
    nonce = secrets.token_bytes(SecretBox.NONCE_SIZE)
    ct = box.encrypt(secret_b32.encode("ascii"), nonce).ciphertext
    return base64.b64encode(nonce + ct).decode("ascii")


def _decrypt_totp(blob: str) -> str:
    from nacl.secret import SecretBox  # type: ignore
    raw = base64.b64decode(blob)
    nonce, ct = raw[: SecretBox.NONCE_SIZE], raw[SecretBox.NONCE_SIZE:]
    box = SecretBox(_master_key())
    return box.decrypt(ct, nonce).decode("ascii")


def begin_totp_enrollment(user_id: int) -> tuple[str, str]:
    """Return (otpauth_url, base32_secret) and store the *encrypted* secret
    *without* enabling TOTP yet. The user confirms via :func:`finish_totp_enrollment`.
    """
    import pyotp  # type: ignore
    user = get_user_by_id(user_id)
    if user is None:
        raise ValueError("user not found")
    secret = pyotp.random_base32()
    enc = _encrypt_totp(secret)
    db.execute(
        "UPDATE users SET totp_secret_enc=?, totp_enabled=0 WHERE id=?", (enc, user_id)
    )
    issuer = os.environ.get("PROVIDER_TOTP_ISSUER", "LLM-Self-Provider")
    url = pyotp.totp.TOTP(secret).provisioning_uri(name=user.username, issuer_name=issuer)
    return url, secret


def finish_totp_enrollment(user_id: int, code: str) -> bool:
    import pyotp  # type: ignore
    row = db.fetchone("SELECT totp_secret_enc FROM users WHERE id=?", (user_id,))
    if not row or not row["totp_secret_enc"]:
        return False
    secret = _decrypt_totp(row["totp_secret_enc"])
    if not pyotp.TOTP(secret).verify(code, valid_window=1):
        return False
    db.execute("UPDATE users SET totp_enabled=1 WHERE id=?", (user_id,))
    return True


def verify_totp(user_id: int, code: str) -> bool:
    """Validate a 6-digit TOTP code or a single-use recovery code."""
    if not code:
        return False
    code = code.strip().replace(" ", "")
    row = db.fetchone(
        "SELECT totp_secret_enc, totp_enabled FROM users WHERE id=?", (user_id,)
    )
    if not row or not row["totp_enabled"] or not row["totp_secret_enc"]:
        return False
    if code.isdigit() and len(code) == 6:
        import pyotp  # type: ignore
        secret = _decrypt_totp(row["totp_secret_enc"])
        return bool(pyotp.TOTP(secret).verify(code, valid_window=1))
    # Try recovery code path.
    return consume_recovery_code(user_id, code)


def disable_totp(user_id: int) -> None:
    db.execute(
        "UPDATE users SET totp_enabled=0, totp_secret_enc=NULL WHERE id=?", (user_id,)
    )
    db.execute("DELETE FROM recovery_codes WHERE user_id=?", (user_id,))


# ------------------------------------------------------- recovery codes

def issue_recovery_codes(user_id: int) -> list[str]:
    """Replace any existing recovery codes with ``RECOVERY_CODE_COUNT`` fresh
    ones. The plaintext list is returned **once**; only argon2 hashes are
    persisted."""
    plains = [secrets.token_hex(5) for _ in range(RECOVERY_CODE_COUNT)]
    with db.transaction() as conn:
        conn.execute("DELETE FROM recovery_codes WHERE user_id=?", (user_id,))
        for p in plains:
            conn.execute(
                "INSERT INTO recovery_codes (user_id, code_hash) VALUES (?, ?)",
                (user_id, PASSWORD_HASHER.hash(p)),
            )
    # Format as 5+5 dashes for readability: 'ab12c-def34'
    return [f"{p[:5]}-{p[5:]}" for p in plains]


def consume_recovery_code(user_id: int, code: str) -> bool:
    code = code.replace("-", "").lower().strip()
    if len(code) != 10:
        return False
    rows = db.fetchall(
        "SELECT id, code_hash FROM recovery_codes WHERE user_id=? AND used_at IS NULL",
        (user_id,),
    )
    for r in rows:
        try:
            PASSWORD_HASHER.verify(r["code_hash"], code)
        except (VerifyMismatchError, InvalidHash):
            continue
        db.execute(
            "UPDATE recovery_codes SET used_at=? WHERE id=?", (db.now_ts(), r["id"])
        )
        return True
    return False


# ------------------------------------------------------------- bootstrap

def ensure_initial_admin(username: str, password: str, *, email: Optional[str] = None) -> User:
    """Idempotent: if no admin exists, create one with the given credentials.
    Otherwise return the first active admin row.
    """
    existing = db.fetchone(
        "SELECT * FROM users WHERE role='admin' AND is_active=1 ORDER BY id LIMIT 1"
    )
    if existing:
        return User.from_row(existing)
    return create_user(username, password, email=email, role="admin")
