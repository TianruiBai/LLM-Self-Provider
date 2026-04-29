"""Control-plane SQLite database (users, sessions, API keys, audit, config_kv).

Runs every ``provider/migrations/*.sql`` file in lexical order at startup,
tracking applied migrations in the ``schema_migrations`` table. WAL mode is
enabled for concurrent reads while a write is in flight.

All public APIs are *synchronous* and cheap; FastAPI handlers wrap them in
``asyncio.to_thread`` when calling from the event loop.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger("provider.db")

_DB_PATH: Path | None = None
_LOCK = threading.RLock()
_LOCAL = threading.local()


def _data_dir() -> Path:
    base = Path(__file__).resolve().parent / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base


def db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        _DB_PATH = _data_dir() / "control.db"
    return _DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(db_path()),
        detect_types=sqlite3.PARSE_DECLTYPES,
        isolation_level=None,  # autocommit; transactions wrapped via ``transaction()``
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def get_conn() -> sqlite3.Connection:
    """Return a thread-local connection, opening it on first use."""
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        conn = _connect()
        _LOCAL.conn = conn
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Begin/commit a transaction on the thread-local connection."""
    conn = get_conn()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def execute(sql: str, params: Iterable[Any] | None = None) -> sqlite3.Cursor:
    return get_conn().execute(sql, tuple(params or ()))


def fetchone(sql: str, params: Iterable[Any] | None = None) -> sqlite3.Row | None:
    cur = execute(sql, params)
    return cur.fetchone()


def fetchall(sql: str, params: Iterable[Any] | None = None) -> list[sqlite3.Row]:
    return execute(sql, params).fetchall()


def now_ts() -> int:
    return int(time.time())


# ---------------------------------------------------------------- migrations

def _migrations_dir() -> Path:
    return Path(__file__).resolve().parent / "migrations"


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name        TEXT PRIMARY KEY,
            applied_at  INTEGER NOT NULL
        )
        """
    )


def _applied_set(conn: sqlite3.Connection) -> set[str]:
    return {r["name"] for r in conn.execute("SELECT name FROM schema_migrations")}


def run_migrations() -> list[str]:
    """Apply any pending ``migrations/*.sql`` files in lexical order.

    Each file is executed inside a single transaction. Returns the list of
    migration names that were applied this call.
    """
    with _LOCK:
        conn = get_conn()
        _ensure_migrations_table(conn)
        applied = _applied_set(conn)
        pending: list[Path] = sorted(
            p for p in _migrations_dir().glob("*.sql") if p.name not in applied
        )
        ran: list[str] = []
        for p in pending:
            sql = p.read_text(encoding="utf-8")
            log.info("Applying migration %s", p.name)
            # NOTE: ``executescript`` issues its own COMMIT under autocommit
            # mode, so we cannot wrap it in our own BEGIN/COMMIT. Each
            # migration file is therefore expected to be internally
            # idempotent (we use ``CREATE TABLE IF NOT EXISTS`` etc.).
            try:
                conn.executescript(sql)
                conn.execute(
                    "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)",
                    (p.name, now_ts()),
                )
            except Exception:
                log.exception("Migration %s failed; aborting startup", p.name)
                raise
            ran.append(p.name)
        return ran


def init() -> None:
    """One-time startup hook: open the connection and run migrations."""
    get_conn()
    run_migrations()


def close_all() -> None:
    """Close the thread-local connection (best-effort, used at shutdown)."""
    conn = getattr(_LOCAL, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        _LOCAL.conn = None
