#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-your_username}"
REMOTE_HOST="${REMOTE_HOST:-your_host}"
REMOTE_BASE="${REMOTE_BASE:-/path/to/your/remote/base/${REMOTE_USER}}"
REMOTE_REPO="${REMOTE_REPO:-$REMOTE_BASE/Length-Value-Model}"
DRY_RUN="${DRY_RUN:-0}"
DELETE="${DELETE:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

ARGS=(
  -av
  --exclude ".git"
  --exclude ".claude"
  --exclude ".venv"
  --exclude ".venv-eval"
  --exclude ".venv-train"
  --exclude ".venv-infer"
  --exclude "__pycache__"
  --exclude ".pytest_cache"
  --exclude ".mypy_cache"
  --exclude ".ruff_cache"
  --exclude ".DS_Store"
  --exclude "/data/.cache"
  --exclude "/data_generation/LenVM-Data"
  --exclude "/models"
  --exclude "/results"
  --exclude "/logs"
  --exclude "/saves"
  --exclude "/wandb"
  --exclude "/cache"
  --exclude "*.egg-info"
  --exclude "uv.lock"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  ARGS+=(--dry-run)
fi

if [[ "$DELETE" -eq 1 ]]; then
  ARGS+=(--delete)
fi

echo "Syncing repo to remote machine"
echo "  local : $REPO_ROOT/"
echo "  remote: $REMOTE_HOST:$REMOTE_REPO/"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_REPO'"
rsync "${ARGS[@]}" "$REPO_ROOT/" "$REMOTE_HOST:$REMOTE_REPO/"

echo "Sync complete."
