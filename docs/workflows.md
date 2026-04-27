# Workflows

This page describes the standard LenVM workflows. Use the root `scripts/` entrypoints whenever possible; they encode the expected environment activation, server flags, paths, and output locations.

Run commands from the repository root. The demo scripts are the main reproducible workflow currently provided: they run a compact end-to-end path that reproduces most of the paper's results. The exact scripts for every paper result are not included yet and will be provided later.

## Pipeline overview

```text
generate data -> train LenVM -> serve/evaluate -> visualize
```

LenVM data generation samples multiple completions per prompt, groups them by `meta_info.lenvm_idx`, and writes JSONL datasets. Training consumes those grouped datasets through the local LlamaFactory fork. Evaluation and guided decoding use the local SGLang fork to serve either the LenVM value model or a base model with LenVM-guided sampling.

## Workflow A: run the demo end to end

This is the recommended reproducible path with the files currently included in the repository. Before launching the Slurm demo, copy `scripts/remote_slurm_job/submit_job.sh.example` to `scripts/remote_slurm_job/submit_job.sh`, copy `.env.example` to `.env`, then configure remote/local paths, environment variables, cache locations, and Slurm resources. The wrapper supports two modes: from a local machine it can sync the repository and submit over SSH; on the remote server it can submit directly with `sbatch` by setting `IS_REMOTE=1`. The default path submits `scripts/remote_slurm_job/train.slurm` through `sbatch`. If you are already on a GPU node or a dedicated GPU machine, you can also run `train.slurm` with `bash`; in that case, set `REMOTE_REPO`, `CACHE_ROOT`, and `SLURM_GPUS_ON_NODE` yourself.

```bash
# Configure paths, environment variables, cache locations, Slurm resources, and IS_REMOTE first.
cp scripts/remote_slurm_job/submit_job.sh.example scripts/remote_slurm_job/submit_job.sh
cp .env.example .env
bash scripts/remote_slurm_job/submit_job.sh
```

On two H100 GPUs, the full demo takes about 1 hour and writes outputs to `results/`. The generated results include LIFEBench, tradeoff evaluation, length prediction, length-token analysis, and visualization of length-change trends.

The same stages can also be run manually:

```bash
bash scripts/environment/env_config_uv.sh
bash scripts/data_generation/demo.sh
bash scripts/training/demo.sh
bash scripts/inference/demo_length_prediction.sh
bash scripts/inference/demo_length_token.sh
bash scripts/inference/demo_lifebench.sh
bash scripts/inference/demo_tradeoff.sh
bash scripts/visualization/demo_visual.sh
```

The demo covers data generation, LenVM training, LIFEBench, tradeoff evaluation, length prediction, length-token analysis, and visualization of length-change trends. It is intended to reproduce most of the paper's results, while full paper-result scripts will be added later.

## Workflow B: use existing data and models

If you want to skip data generation and training, download the existing artifacts:

```bash
bash scripts/environment/env_config_uv.sh
bash scripts/download_data_and_model.sh
```

Then run the evaluation or visualization script you need after updating its parameters to match the downloaded data/model locations:

```bash
bash scripts/inference/demo_length_prediction.sh
bash scripts/inference/demo_length_token.sh
bash scripts/inference/demo_lifebench.sh
bash scripts/inference/demo_tradeoff.sh
bash scripts/visualization/demo_visual.sh
```

Expected major inputs and outputs:

- Data: `data_generation/LenVM-Data/`
- Model artifacts: `models/` or checkpoint paths expected by scripts
- Evaluation and visualization results: `results/`

Downloaded data/model artifacts are a shortcut around expensive stages, not a guarantee that every demo script can run unchanged. Check model paths, checkpoint paths, ports, tensor/pipeline parallel settings, and memory fractions before launching GPU workloads.

## Workflow C: generate LenVM data

Run the demo data generation workflow:

```bash
bash scripts/data_generation/demo.sh
```

The script activates `.venv-infer`, launches an SGLang server, and calls:

```bash
python -m data_generation.data_generator.main
```

Expected outputs are JSONL files under `data_generation/LenVM-Data/<model-name>/...`, including grouped files used by training.

For larger runs, dataset-specific commands, OpenCodeReasoning-2 preparation, and downsampling utilities, see [../data_generation/data_generation.md](../data_generation/data_generation.md).

## Workflow D: train a LenVM checkpoint

Run the demo training workflow:

```bash
bash scripts/training/demo.sh
```

The script activates `.venv-train`, downloads `dataset_info.json` for the demo dataset if needed, and runs:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train ./scripts/training/configs/demo.yaml
```

The demo config uses:

- `stage: lenvm`
- dataset directory: `./data_generation/LenVM-Data`
- output directory: `./saves/a-qwen2.5-3b-instruct-b-qwen2.5-3b-instruct/demo_math_instruction`
- evaluation datasets: `demo_math_test,demo_instruction_test`

## Workflow E: evaluate and analyze

### First-token length prediction

```bash
bash scripts/inference/demo_length_prediction.sh
```

This serves the LenVM checkpoint as an embedding-style value model and writes results under `results/length_prediction/`.

### Length-token analysis

```bash
bash scripts/inference/demo_length_token.sh
```

This analyzes token-level value behavior and can produce wordcloud artifacts under `results/length_token_analysis/`.

### LIFEBench

```bash
bash scripts/inference/demo_lifebench.sh
```

This launches a base SGLang model with in-process LenVM guidance and runs baseline plus LenVM-guided LIFEBench experiments. Benchmark outputs are written under `results/LIFEBench/`.

### Tradeoff evaluation

```bash
bash scripts/inference/demo_tradeoff.sh
```

This runs baseline sampling, LenVM-guided sampling across value scales, budget evaluation, and plotting. Outputs are written under `results/tradeoff/`.

## Workflow F: visualize value behavior

```bash
bash scripts/visualization/demo_visual.sh
```

The visualization script launches the LenVM value-model server, runs the text visualization pipeline, and writes markdown plus hover-HTML artifacts under `results/visualization/`.

## Artifact map

| Stage | Main script | Typical outputs |
| --- | --- | --- |
| Remote Slurm demo | `scripts/remote_slurm_job/submit_job.sh` | full demo outputs under `results/` |
| Setup | `scripts/environment/env_config_uv.sh` | `.venv-train/`, `.venv-infer/`, `.venv-eval/` |
| Download | `scripts/download_data_and_model.sh` | `data_generation/LenVM-Data/`, `models/` |
| Data generation | `scripts/data_generation/demo.sh` | grouped JSONL files under `data_generation/LenVM-Data/` |
| Training | `scripts/training/demo.sh` | checkpoints under `saves/` |
| Inference/eval | `scripts/inference/*.sh` | metrics and JSONL/CSV artifacts under `results/` |
| Visualization | `scripts/visualization/demo_visual.sh` | markdown and HTML files under `results/visualization/` |

## Server patterns

LenVM value-model serving uses SGLang embedding-style launches with architecture override, for example:

```bash
python -m sglang.launch_server \
  --json-model-override-args '{"architectures":["Qwen2ForLengthValueModel"]}' \
  --is-embedding
```

LenVM-guided decoding serves a base model with in-process guidance using flags such as:

```bash
--enable-lvm-guided-sampling
--lvm-guided-inproc
--lvm-guided-inproc-model-path <lenvm-checkpoint>
--lvm-guided-fn sglang.srt.lvm.lvm_guided_sampling:lvm_combined_guidance
```

Use the scripts as the authoritative examples for complete launch commands.
