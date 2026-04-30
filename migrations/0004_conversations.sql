-- 0004_conversations: server-side WebUI chat history synced across devices.
--
-- Stores one row per conversation owned by an authenticated user. The full
-- conversation (messages, attachments, metadata) is persisted as a JSON blob
-- in ``data`` so the WebUI client can round-trip its in-memory shape without
-- the server having to understand every field. Lightweight columns are
-- duplicated for cheap listing/sorting.

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT    PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       TEXT    NOT NULL DEFAULT '',
    model       TEXT    NOT NULL DEFAULT '',
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    data        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
    ON conversations (user_id, updated_at DESC);
