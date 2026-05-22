# LenVM inference timing analysis

End-to-end and per-decoding-step latency decomposition for the LenVM-guided
sampling path, compared against an otherwise-identical baseline SGLang server.

This exists to answer the question raised in the LenVM paper review: how much
wall-clock overhead does LenVM-guided decoding add on top of vanilla decoding,
and where does that overhead live inside each decoding step?

## What gets measured

Two server lifecycles drive the comparison, so per-step instrumentation only
captures the configuration under test:

1. **baseline** — SGLang with `--enable-lvm-guided-sampling` off. Vanilla
   chat-completion sampling against `Qwen/Qwen2.5-7B-Instruct`.
2. **lenvm** — Same base model, plus the in-process LenVM value model
   (`./models/namezz/lvm-math-0402-a-qwen2.5-7b-instruct-b-qwen2.5-1.5b-instruct`
   by default), with `--enable-lvm-guided-sampling --lvm-guided-inproc
   --lvm-guided-fn lvm_combined_guidance`.

Both servers point `SGLANG_LVM_TIMING_LOG` at their own JSONL file. The client
hits each server in turn with the same GSM8K prompt set (50 questions × 16
samples by default), recording end-to-end wall clock per run.

The instrumentation lives in `sglang-LenVM/python/sglang/srt/lvm/timing.py` and
hooks `Sampler.forward` plus `LvmGuidedSampler.apply`. Per decoding step it
emits one JSONL line with:

- `t_sampler_total_ms` — total time inside `Sampler.forward`
- `t_pre_lvm_ms` — preprocess + temperature scaling + softmax
- `t_lvm_apply_outer_ms` — full `LvmGuidedSampler.apply` call
  - `t_lvm_build_pending_ms` — gather candidates & request state
  - `t_lvm_forward_ms` — LenVM extend + launch + collect
  - `t_lvm_apply_guidance_ms` — apply value-based adjustment to probs
- `t_sample_ms` — sampling kernel
- `lvm_active`, `batch_size`, `is_greedy`

Theoretical FLOPs are computed by `analyze.py` at the layer level. `flops.py`
loads each model's `config.json` (HF cache or local dir; falls back to
hardcoded Qwen2.5 dims if missing) and counts:

- per-layer linear matmuls: Q / K / V / O projections (GQA-aware) + SwiGLU MLP
  (gate + up + down)
- per-layer attention compute: `2 * H_q * head_dim * seq_len` for each of
  Q@K^T and attn@V (so attention contribution scales with position)
- `lm_head`: `2 * hidden * vocab`

A baseline run is split into prefill (charged once per unique prompt, since
SGLang prefix caching is on by default) and decode (per sample). A LenVM run
adds, per generated token, one `tree_value_extend` forward plus `k`
candidate forwards through the value model. The analyzer reports both the
total PFLOPs and a per-component split so the linear / attention / lm_head
shares of the baseline are visible alongside the LenVM-specific overhead.
Contrasting the theoretical FLOPs ratio with the measured wall-clock ratio
shows how much of the slowdown is raw compute increase vs GPU
underutilization.

## Running it

```bash
# from repository root, with .venv-infer and .venv-eval already built
bash scripts/inference/lenvm_timing.sh
```

Overridable knobs (env vars; see top of the script for defaults):
`BASE_MODEL`, `LENVM_MODEL`, `MAX_QUESTIONS`, `N_SAMPLES`, `MAX_TOKENS`,
`MAX_CONCURRENCY`, `LENVM_TOP_K`, `LENVM_VALUE_SCALE`, `RESULTS_DIR`.

The script chains three stages:

1. Start baseline server → run `inference.timing.run_timing` → kill server
2. Start LenVM server → run `inference.timing.run_timing` → kill server
3. `inference.timing.analyze` reads both JSONL streams + meta files and writes
   a `summary.csv`, `summary.json`, and two plots into `RESULTS_DIR`.

## Outputs (under `RESULTS_DIR`)

- `baseline.timing.jsonl`, `lenvm.timing.jsonl` — per-step records
- `baseline.meta.json`, `lenvm.meta.json` — wall-clock, token counts, cmdline
- `baseline.gpu_samples.csv`, `lenvm.gpu_samples.csv` — `nvidia-smi` 1 Hz log
- `summary.csv`, `summary.json` — aggregated table (incl. theoretical FLOPs, achieved TFLOPs/s, ratio row)
- `per_step_breakdown.png` — stacked bar of sampler-side decomposition
- `lvm_apply_breakdown.png` — LenVM `apply()` internal breakdown
- `flops_breakdown.png` — stacked bar of theoretical FLOPs by component
  (base linear / attention / lm_head + LenVM extend / candidates / prefill)

## Caveats

- The per-step timer adds Python-level `time.perf_counter()` calls on the
  decoding hot path. They are no-ops when `SGLANG_LVM_TIMING_LOG` is unset.
- `t_lvm_apply_outer_ms` covers a few short helpers (e.g.
  `_get_inproc_provider`, `clean_stale_requests`) not separately broken out;
  at production batch sizes the residual is small but visible at low load.
- Single-rank only. For TP/DP > 1 the timer writes from one rank; extend the
  log filename with the rank suffix if you need per-worker traces.
- Wall-clock comparisons assume both servers see the same prompts at the same
  concurrency. Run them back-to-back on a quiet GPU.
