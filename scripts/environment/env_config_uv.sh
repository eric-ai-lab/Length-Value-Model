set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
UV_PYTHON=3.12 uv lock

[ -d ".venv-train" ] || uv venv --python 3.12 .venv-train
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --only-group train

[ -d ".venv-infer" ] || uv venv --python 3.12 .venv-infer
UV_PROJECT_ENVIRONMENT=.venv-infer uv sync --only-group infer

[ -d ".venv-eval" ] || uv venv --python 3.12 .venv-eval
UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --only-group eval
