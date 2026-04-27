# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and dependencies

This repository is a Python 3.12 `uv` workspace with separate dependency groups for the main LenVM workflows:

```bash
# Create/update all standard environments used by the demo scripts
bash scripts/environment/env_config_uv.sh

# Equivalent targeted setup
UV_PYTHON=3.12 uv lock
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --only-group train
UV_PROJECT_ENVIRONMENT=.venv-infer uv sync --only-group infer
UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --only-group eval
```

The root `pyproject.toml` defines editable workspace sources for the local forks `LlamaFactory-LenVM` and `sglang-LenVM/python`. Most workflow scripts assume they are run from the repository root and activate `.venv-train`, `.venv-infer`, or `.venv-eval` themselves.

## Common commands

```bash
# Download published demo data and model artifacts from Hugging Face
bash scripts/download_data_and_model.sh

# Generate demo training/eval data through an OpenAI-compatible SGLang server
bash scripts/data_generation/demo.sh

# Train a LenVM checkpoint with the demo LlamaFactory config
bash scripts/training/demo.sh

# Evaluate first-token length prediction
bash scripts/inference/demo_length_prediction.sh

# Analyze length-sensitive tokens and generate wordcloud artifacts
bash scripts/inference/demo_length_token.sh

# Run LIFEBench baseline and LenVM-guided evaluations
bash scripts/inference/demo_lifebench.sh

# Run quality/length tradeoff sampling, budget evaluation, and plots
bash scripts/inference/demo_tradeoff.sh

# Generate markdown and hover-HTML LenVM value visualizations
bash scripts/visualization/demo_visual.sh
```

For direct CLI training, use the LlamaFactory entrypoint after activating the train environment:

```bash
source .venv-train/bin/activate
FORCE_TORCHRUN=1 llamafactory-cli train ./scripts/training/configs/demo.yaml
```

For the local LlamaFactory fork:

```bash
cd LlamaFactory-LenVM
make quality              # ruff check + format check
make style                # ruff check --fix + format
make test                 # pytest tests/ tests_v1/
WANDB_DISABLED=true uv run pytest -vv --import-mode=importlib tests/path/to/test_file.py
```

For the local SGLang fork, tests live under `sglang-LenVM/test`; use the local package from the root workspace or activate the inference/dev environment before invoking pytest. SGLang also has `sglang-LenVM/test/pytest.ini` and suite runners `python test/run_suite.py` and `python test/run_suite_nightly.py`.

## Architecture overview

LenVM models remaining generation length as a token-level value estimate. The repository stitches together data generation, LlamaFactory-based value training, SGLang serving/guided decoding, and evaluation scripts.

- `data_generation/data_generator/` builds datasets by sampling multiple completions per prompt from an OpenAI-compatible endpoint. `main.py` handles dataset loading/splitting and CLI arguments, `prompt_builder.py` converts supported datasets into chat requests, `generator.py` performs async OpenAI-compatible generation, and `processor.py` schedules concurrent sampling and batched JSONL writes. Outputs are grouped by `meta_info.lenvm_idx` for LenVM training.
- `LlamaFactory-LenVM/` is a local LlamaFactory fork with the LenVM training stage. Important LenVM-specific paths include `src/llamafactory/train/lenvm/`, `src/llamafactory/data/processor/value_regression.py`, `src/llamafactory/data/collator.py` (`LengthValueDataCollator`), and `src/llamafactory/model/model_utils/valuehead.py`. The demo config uses `stage: lenvm` and writes checkpoints under `saves/`.
- `sglang-LenVM/` is a local SGLang fork extended for LenVM value models and guided sampling. The main LenVM integration is in `python/sglang/srt/lvm/`, server flags are added in `python/sglang/srt/server_args.py`, sampling hooks are in `python/sglang/srt/layers/sampler.py`, and Qwen LenVM runtime wrappers are in `python/sglang/srt/models/qwen2_lvm.py`, `qwen3_lvm.py`, and `qwen2_5_vl_lvm.py`.
- `inference/length_prediction/`, `inference/length_token/`, `inference/tradeoff/`, `inference/LIFEBench/`, and `inference/visualization/` are standalone evaluation/analysis entrypoints. The scripts in `scripts/inference/` and `scripts/visualization/` show the expected server launch flags and result paths.

## Workflow notes

- Run scripts from the repository root; many paths are relative and assume root CWD.
- Demo inference scripts start local SGLang servers on ports `10006`-`10010` and usually wait on `http://127.0.0.1:$PORT/v1/models` before running evaluations.
- LenVM value-model serving uses embedding-style SGLang launches with `--json-model-override-args '{"architectures":["Qwen2ForLengthValueModel"]}'` and `--is-embedding`.
- LenVM-guided decoding launches a base generation model with `--enable-lvm-guided-sampling`, `--lvm-guided-inproc`, `--lvm-guided-inproc-model-path`, and `--lvm-guided-fn sglang.srt.lvm.lvm_guided_sampling:lvm_combined_guidance`.
- Large generated artifacts are expected under `data_generation/LenVM-Data/`, `saves/`, `models/`, `results/`, and `cache/`; avoid committing them unless explicitly requested.
