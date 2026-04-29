-- 0001_init: control-plane schema for auth, sessions, API keys, audit, config.

CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY,
    username        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    email           TEXT             UNIQUE COLLATE NOCASE,
    password_hash   TEXT,
    role            TEXT    NOT NULL DEFAULT 'user' CHECK (role IN ('user','admin')),
    is_active       INTEGER NOT NULL DEFAULT 1,
    totp_secret_enc TEXT,
    totp_enabled    INTEGER NOT NULL DEFAULT 0,
    oidc_subject    TEXT    UNIQUE,
    created_at      INTEGER NOT NULL,
    last_login_at   INTEGER,
    last_login_ip   TEXT
);

CREATE TABLE IF NOT EXISTS api_keys (
    id              INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            TEXT    NOT NULL,
    key_prefix      TEXT    NOT NULL,
    key_hash        TEXT    NOT NULL UNIQUE,
    ip_allowlist    TEXT,                     -- CSV of CIDR; NULL = no restriction
    created_at      INTEGER NOT NULL,
    last_used_at    INTEGER,
    last_used_ip    TEXT,
    expires_at      INTEGER,
    revoked_at      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);

CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT    PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at      INTEGER NOT NULL,
    expires_at      INTEGER NOT NULL,
    ip              TEXT,
    user_agent      TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_exp  ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS recovery_codes (
    id              INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    code_hash       TEXT    NOT NULL,
    used_at         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_recovery_user ON recovery_codes(user_id);

CREATE TABLE IF NOT EXISTS oauth_states (
    state           TEXT PRIMARY KEY,
    provider        TEXT NOT NULL,
    code_verifier   TEXT,
    redirect_to     TEXT,
    expires_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS request_audit (
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
CREATE INDEX IF NOT EXISTS idx_audit_ts ON request_audit(ts);
CREATE INDEX IF NOT EXISTS idx_audit_user ON request_audit(user_id, ts);

CREATE TABLE IF NOT EXISTS config_kv (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    updated_at      INTEGER NOT NULL,
    updated_by      INTEGER REFERENCES users(id) ON DELETE SET NULL
);

-- Per-user rate-limit token buckets (refilled lazily on each request).
CREATE TABLE IF NOT EXISTS rate_buckets (
    key             TEXT PRIMARY KEY,         -- e.g. 'user:42:chat' or 'ip:1.2.3.4:login'
    tokens          REAL NOT NULL,
    updated_at      INTEGER NOT NULL
);
