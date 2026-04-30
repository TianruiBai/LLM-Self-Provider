#!/usr/bin/env bash
set -euo pipefail

PROVIDER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$PROVIDER_ROOT/.." && pwd)"
ENV_FILE="$PROVIDER_ROOT/.env"

CONFIG="$PROVIDER_ROOT/models.yaml"
HOST="0.0.0.0"
PORT="8088"
LOG_LEVEL="info"
PYTHON_EXE=""
PUBLIC_BASE_URL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    --python) PYTHON_EXE="$2"; shift 2 ;;
    --public-base-url) PUBLIC_BASE_URL="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: ./provider/scripts/run-native.sh [options]

  --config PATH            models.yaml to use (default: provider/models.yaml)
  --host HOST              gateway bind host (default: 0.0.0.0)
  --port PORT              gateway bind port (default: 8088)
  --log-level LEVEL        debug|info|warning|error
  --python PATH            python interpreter to use
  --public-base-url URL    override PROVIDER_PUBLIC_BASE_URL for this run
EOF
      exit 0 ;;
    *)
      echo "unknown flag: $1" >&2
      exit 2 ;;
  esac
done

load_dotenv() {
  local path="$1"
  [[ -f "$path" ]] || return 0
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    local line="$raw"
    line="${line#${line%%[![:space:]]*}}"
    line="${line%${line##*[![:space:]]}}"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    local key="${line%%=*}"
    local value="${line#*=}"
    [[ -z "$key" ]] && continue
    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done < "$path"
}

resolve_python() {
  local override="$1"
  if [[ -n "$override" ]]; then
    printf '%s\n' "$override"
    return 0
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    printf '%s\n' "$VIRTUAL_ENV/bin/python"
    return 0
  fi
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$PROJECT_ROOT/.venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  echo "python interpreter not found" >&2
  return 1
}

load_dotenv "$ENV_FILE"

if [[ -n "$PUBLIC_BASE_URL" ]]; then
  export PROVIDER_PUBLIC_BASE_URL="$PUBLIC_BASE_URL"
fi

missing=()
for key in PROVIDER_AUTH_PEPPER PROVIDER_MASTER_KEY PROVIDER_BOOTSTRAP_ADMIN_USER PROVIDER_BOOTSTRAP_ADMIN_PASSWORD; do
  [[ -n "${!key:-}" ]] || missing+=("$key")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "missing required env vars for native run: ${missing[*]}" >&2
  echo "populate provider/.env first" >&2
  exit 1
fi

if [[ "${PROVIDER_PUBLIC_BASE_URL:-}" == *"0.0.0.0"* ]]; then
  echo "WARN: PROVIDER_PUBLIC_BASE_URL points at 0.0.0.0; use --public-base-url for a real external URL." >&2
fi

PYTHON_EXE="$(resolve_python "$PYTHON_EXE")"
cd "$PROJECT_ROOT"

echo "==> running provider natively"
echo "    python: $PYTHON_EXE"
echo "    config: $CONFIG"
echo "    bind  : $HOST:$PORT"

exec "$PYTHON_EXE" -m provider --config "$CONFIG" --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"