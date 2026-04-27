# Getting started

This guide sets up the standard LenVM development and demo environments. Run commands from the repository root unless a command explicitly changes directories.

## Requirements

- Python 3.12
- `uv`
- CUDA-capable GPU environment for training and SGLang inference demos
- Hugging Face access for downloading published datasets and model artifacts

The repository is a `uv` workspace. The root `pyproject.toml` defines separate dependency groups for training, inference, and evaluation.

## Set up environments

The standard setup script creates three virtual environments:

```bash
bash scripts/environment/env_config_uv.sh
```

It performs the equivalent of:

```bash
UV_PYTHON=3.12 uv lock
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --only-group train
UV_PROJECT_ENVIRONMENT=.venv-infer uv sync --only-group infer
UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --only-group eval
```

Environment roles:

- `.venv-train`: LlamaFactory and LenVM training dependencies.
- `.venv-infer`: SGLang serving, generation, and inference dependencies.
- `.venv-eval`: evaluation, plotting, and analysis dependencies.

Most workflow scripts activate the environment they need internally.

## Download existing dataset and models

If you want to skip data generation and training, download the existing data and model artifacts:

```bash
bash scripts/download_data_and_model.sh
```

The script downloads LenVM data under `data_generation/LenVM-Data/` and model artifacts under `models/`. These artifacts are provided so you can bypass the expensive data-generation and training stages, but you may need to adjust downstream script parameters such as model paths, checkpoint paths, ports, tensor/pipeline parallelism, or memory fractions before running a specific evaluation.

## Run the demo workflow

The recommended reproducible path is to run the demo scripts. The demo is designed to reproduce most of the paper's results with the files currently included in this repository; the exact full paper-result scripts are not included yet and will be provided later.

Before launching the Slurm demo, copy `scripts/remote_slurm_job/submit_job.sh.example` to `scripts/remote_slurm_job/submit_job.sh`, copy `.env.example` to `.env`, then configure remote/local paths, environment variables, cache locations, and Slurm resources. From a local machine, the wrapper can sync the repository and submit over SSH; when already on the remote server, set `IS_REMOTE=1` to submit directly with `sbatch`. If you are already on a GPU node or a dedicated GPU machine, you can also run `scripts/remote_slurm_job/train.slurm` with `bash`; in that case, set `REMOTE_REPO`, `CACHE_ROOT`, and `SLURM_GPUS_ON_NODE` yourself.

```bash
cp scripts/remote_slurm_job/submit_job.sh.example scripts/remote_slurm_job/submit_job.sh
cp .env.example .env
bash scripts/remote_slurm_job/submit_job.sh
```

On two H100 GPUs, the full demo takes about 1 hour. Results are written under `results/` and include LIFEBench, tradeoff evaluation, length prediction, length-token analysis, and visualization of length-change trends.

You can also run the stages manually:

```bash
bash scripts/data_generation/demo.sh
bash scripts/training/demo.sh
bash scripts/inference/demo_length_prediction.sh
bash scripts/inference/demo_length_token.sh
bash scripts/inference/demo_lifebench.sh
bash scripts/inference/demo_tradeoff.sh
bash scripts/visualization/demo_visual.sh
```

If you downloaded existing artifacts, you can skip `scripts/data_generation/demo.sh` and `scripts/training/demo.sh` after updating the evaluation script parameters to point at the downloaded data/model locations.

## Common pitfalls

- Run scripts from the repository root; many paths are root-relative.
- Ensure Hugging Face credentials and access are configured before downloading private or gated artifacts.
- CUDA, PyTorch, FlashAttention, and SGLang dependencies are hardware-sensitive; use the environment setup script as the first source of truth.
- Training and inference demos can be GPU- and memory-intensive. Adjust script environment variables such as `TP_SIZE`, `DP_SIZE`, `MEM_FRACTION_STATIC`, and `CONTEXT_LENGTH` when needed.

## Next steps

- For the full pipeline, see [workflows.md](workflows.md).
- For component layout and implementation boundaries, see [repository-structure.md](repository-structure.md).
- For advanced data generation commands, see [../data_generation/data_generation.md](../data_generation/data_generation.md).
