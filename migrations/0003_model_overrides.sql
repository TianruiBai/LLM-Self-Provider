-- 0003_model_overrides: admin-editable per-model runtime parameters.
--
-- Stores tweaks that previously required editing models.yaml on disk and
-- restarting the gateway. Any column may be NULL / empty — when missing,
-- the spawn falls back to the YAML defaults.

ALTER TABLE model_publish ADD COLUMN ctx_size      INTEGER;
ALTER TABLE model_publish ADD COLUMN extra_args    TEXT;     -- JSON array of CLI flags
ALTER TABLE model_publish ADD COLUMN system_prompt TEXT;     -- overrides folder/prompt.md
