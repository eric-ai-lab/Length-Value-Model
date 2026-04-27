# Repository structure

This repository integrates data generation, LenVM training, SGLang serving, guided decoding, and evaluation. The local forks under `LlamaFactory-LenVM/` and `sglang-LenVM/` provide implementation support, while the root scripts and docs describe the integrated LenVM workflow.

## Top-level layout

```text
scripts/                 Canonical workflow entrypoints
data_generation/         Data generation pipeline and generated LenVM datasets
inference/               Evaluation, analysis, LIFEBench, tradeoff, visualization
LlamaFactory-LenVM/      Local LlamaFactory fork with LenVM training support
sglang-LenVM/            Local SGLang fork with LenVM serving and guidance support
docs/                    Repo-level user documentation
assets/                  Documentation figures
```

## Data generation

The data generation pipeline lives in `data_generation/data_generator/`.

Important files:

- `data_generation/data_generator/main.py`: command-line entrypoint for dataset loading, splitting, filtering, generation, and grouping.
- `data_generation/data_generator/prompt_builder.py`: converts supported datasets into chat messages for an OpenAI-compatible API.
- `data_generation/data_generator/generator.py`: performs async generation through an OpenAI-compatible client.
- `data_generation/data_generator/processor.py`: schedules concurrent sampling and batched JSONL writes.
- `data_generation/data_generation.md`: advanced command reference for larger or dataset-specific generation runs.

Generated samples are grouped by `meta_info.lenvm_idx` so training can compare multiple completions for the same prompt.

## Training with LlamaFactory-LenVM

`LlamaFactory-LenVM/` is a local LlamaFactory fork extended for LenVM training.

Important LenVM-specific paths:

- `LlamaFactory-LenVM/src/llamafactory/train/lenvm/`: LenVM training workflow, trainer, and metrics.
- `LlamaFactory-LenVM/src/llamafactory/data/processor/value_regression.py`: dataset processor for length-value regression.
- `LlamaFactory-LenVM/src/llamafactory/data/collator.py`: includes `LengthValueDataCollator`.
- `LlamaFactory-LenVM/src/llamafactory/model/model_utils/valuehead.py`: value-head support used by LenVM training.

The root demo training script is `scripts/training/demo.sh`, and its demo config is `scripts/training/configs/demo.yaml`.

## Serving and guided decoding with sglang-LenVM

`sglang-LenVM/` is a local SGLang fork extended for LenVM value models and guided sampling.

Important LenVM-specific paths:

- `sglang-LenVM/python/sglang/srt/lvm/`: LenVM guided sampling, in-process runner, and value utilities.
- `sglang-LenVM/python/sglang/srt/server_args.py`: server flags such as `--enable-lvm-guided-sampling`, `--lvm-guided-inproc`, and related LenVM options.
- `sglang-LenVM/python/sglang/srt/layers/sampler.py`: sampling hook that applies LenVM guidance.
- `sglang-LenVM/python/sglang/srt/models/qwen2_lvm.py`: Qwen2 LenVM runtime wrapper.
- `sglang-LenVM/python/sglang/srt/models/qwen3_lvm.py`: Qwen3 LenVM runtime wrapper.
- `sglang-LenVM/python/sglang/srt/models/qwen2_5_vl_lvm.py`: Qwen2.5-VL LenVM runtime wrapper.

Inference scripts under `scripts/inference/` show complete launch commands for value-model serving and guided decoding.

## Evaluation and visualization

The `inference/` directory contains standalone evaluation and analysis entrypoints:

- `inference/length_prediction/`: first-token length prediction evaluation.
- `inference/length_token/`: token-level value analysis and wordcloud utilities.
- `inference/LIFEBench/`: length-instruction-following benchmark integration.
- `inference/tradeoff/`: quality/length budget and tradeoff evaluation.
- `inference/visualization/`: value visualization scripts and sample inputs.

Root scripts in `scripts/inference/` and `scripts/visualization/` are the recommended way to run these components.

## How components interact

1. `scripts/data_generation/demo.sh` launches SGLang and writes grouped JSONL data under `data_generation/LenVM-Data/`.
2. `scripts/training/demo.sh` trains a LenVM checkpoint through `llamafactory-cli` using `scripts/training/configs/demo.yaml`.
3. Inference scripts launch SGLang either as a LenVM value-model server or as a base model with in-process LenVM guidance.
4. Evaluation and visualization scripts read datasets/checkpoints and write outputs under `results/`.

## Documentation boundaries

- Root `README.md` and `docs/` explain the integrated LenVM workflow.
- `LlamaFactory-LenVM/README.md` documents the local LlamaFactory fork and upstream LLaMA-Factory behavior.
- `sglang-LenVM/README.md` documents the local SGLang fork and upstream SGLang behavior.

When the same task can be done through a root script or a subproject command, prefer the root script for this repository's standard workflow.
