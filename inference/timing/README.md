# LenVM inference timing analysis

End-to-end and per-decoding-step latency decomposition for the LenVM-guided
sampling path, compared against an otherwise-identical baseline SGLang server.

This exists to answer the question raised in the LenVM paper review: how much
wall-clock overhead does LenVM-guided decoding add on top of vanilla decoding,
and where does that overhead live inside each decoding step?

## TL;DR results

Reference configuration: 1× H100 SXM, Qwen2.5-7B-Instruct base + the
`lvm-math-0402-a-qwen2.5-7b-instruct-b-qwen2.5-1.5b-instruct` value model,
GSM8K 50 questions × 16 samples, `max_tokens=6000`, T=1.0, top-p=1.0,
baseline `top_k=-1`, LenVM `top_k=5`, value-scale 0 (paper config).

| metric | baseline | LenVM | ratio |
| --- | ---: | ---: | ---: |
| end-to-end wall clock (s) | 19.22 | 87.44 | **4.55× slower** |
| throughput (output tok/s) | 12,366 | 2,719 | **4.55× drop** |
| mean `Sampler.forward` (ms / step) | 0.19 | 50.31 | **268×** |
| theoretical GFLOPs / output token | 14.24 | 30.24 | **2.12×** |
| achieved throughput (TFLOPs/s) | 179.9 | 83.2 | 0.46× |
| H100 bf16 peak utilization | ~18% | ~9% | — |

`Sampler.forward` breakdown (mean ms / step): pre-LVM 0.07 → 0.14,
`LvmGuidedSampler.apply` 0.00 → 50.03 (build_pending 12.3, LenVM forward 36.6,
apply_guidance 5.3), sample kernel 0.12 → 0.17. The LenVM `apply()` call is
~99% of the per-step latency delta; pre-LVM and sample-kernel costs are
unchanged.

FLOPs breakdown (PFLOPs over the run): baseline.linear 3.17, base.attention
0.02, base.lm_head 0.26 — same for both runs. LenVM adds extend 0.63 +
candidates 3.17 + prefill 0.01 = 3.82 PFLOPs extra. Of the extra cost,
~83% is the `top_k=5` candidate forwards through the 1.5B value model.

**Conclusion**: LenVM nearly doubles the theoretical compute (2.10× total
PFLOPs) and slows wall clock by 4.55×. Half of the slowdown is genuine
extra compute; the other half is GPU underutilization (CPU candidate prep
stalls, separate-stream sync, smaller effective batch on the value model).

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
- LenVM-extra FLOPs assume **independent single-token forwards** for each of
  the `k` candidates per generated token. This matches the current
  sglang-LenVM in-proc path (`tree_value_extend` extends KV once, then
  each candidate is a separate single-token forward attending to the shared
  prefix cache). If a future implementation batches all `k` candidates into
  one forward and amortizes some MLP/attention cost, pass
  `candidate_cost_multiplier < 1.0` to `lvm_extra_flops` to scale that term.
- LenVM head cost uses the small `MLP2SiLUValueHead` (`d*d + d*out_dim`) not
  the base model's `lm_head` (`d * vocab_size`). `ModelConfig.load(...)`
  auto-detects this by checking for `value_head.safetensors` next to
  `config.json` or `LengthValueModel`/`ValueModel`/`ValueHead` in the
  config's `architectures`. Force the choice via `head_type=...` if needed.
- Wall-clock comparisons assume both servers see the same prompts at the same
  concurrency. Run them back-to-back on a quiet GPU.
