-- 0002_models_publish: admin-managed visibility for the model registry.
--
-- The provider auto-discovers models from `models.yaml`, `models_dir`, and
-- the user's LM Studio cache (`~/.lmstudio/models`). Discovery yields *all*
-- the GGUFs the operator happens to have on disk, which is rarely the set
-- they want exposed to end users. This table lets an admin pin a curated
-- list: only ids with `published = 1` show up in `/v1/models` for non-admin
-- callers.

CREATE TABLE IF NOT EXISTS model_publish (
    model_id   TEXT PRIMARY KEY,
    published  INTEGER NOT NULL DEFAULT 0,
    label      TEXT,                            -- optional display label
    updated_at INTEGER NOT NULL,
    updated_by INTEGER REFERENCES users(id) ON DELETE SET NULL
);
