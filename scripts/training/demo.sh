
set -euo pipefail

source .venv-train/bin/activate

hf download namezz/LenVM-Data dataset_info.json --local-dir ./data_generation/LenVM-Data --repo-type dataset
FORCE_TORCHRUN=1 llamafactory-cli train ./scripts/training/configs/demo.yaml
