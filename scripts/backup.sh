#!/usr/bin/env bash
# D6 — Backup script for the self-provider stack.
#
# Creates a timestamped tarball under ./backups/ containing:
#   - data/control.db (SQLite, copied via .backup so it's WAL-consistent)
#   - data/lance/     (Lance datasets — global + per-user knowledge bases)
#   - .env            (so the same secrets can restore the install)
#
# Designed to be safe to run while the stack is up.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="$REPO_ROOT/backups"
mkdir -p "$OUT_DIR"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# 1. SQLite consistent snapshot (only if the gateway container is up).
if docker compose ps gateway --status running --quiet | grep -q .; then
  echo "==> snapshotting control.db via sqlite3 .backup inside gateway container"
  docker compose exec -T gateway sh -lc \
    'mkdir -p /tmp/snap && sqlite3 /app/data/control.db ".backup /tmp/snap/control.db"'
  docker compose cp gateway:/tmp/snap/control.db "$WORK/control.db"
else
  echo "==> gateway not running; copying control.db from the volume"
  if [[ -f "$REPO_ROOT/data/control.db" ]]; then
    cp "$REPO_ROOT/data/control.db" "$WORK/control.db"
  fi
fi

# 2. Lance datasets — copy whole tree from the docker volume.
echo "==> copying lance datasets"
docker run --rm \
  -v "${REPO_ROOT##*/}_lance:/src:ro" \
  -v "$WORK:/dst" \
  alpine sh -c 'cp -a /src /dst/lance' 2>/dev/null || {
    # Fallback: maybe lance lives on the host directly.
    if [[ -d "$REPO_ROOT/data/lance" ]]; then
      cp -a "$REPO_ROOT/data/lance" "$WORK/lance"
    fi
  }

# 3. .env (without the bootstrap password — strip it).
if [[ -f "$REPO_ROOT/.env" ]]; then
  grep -v '^PROVIDER_BOOTSTRAP_ADMIN_PASSWORD=' "$REPO_ROOT/.env" > "$WORK/.env"
fi

# 4. Tarball.
ARCHIVE="$OUT_DIR/provider-backup-$STAMP.tar.gz"
( cd "$WORK" && tar -czf "$ARCHIVE" . )
echo "==> wrote $ARCHIVE"

ls -lh "$ARCHIVE"
