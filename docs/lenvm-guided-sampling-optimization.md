# LenVM guided sampling optimization summary

Latency-oriented changes for the in-process LenVM guided decoding path in the
SGLang fork. This follows the timing question from PR #2: how much overhead
does LenVM add, and where does that overhead remain after optimizing the hot
path?

## TL;DR results

Reference benchmark: 1x H100 SXM, `Qwen/Qwen2.5-7B-Instruct` base model plus
`namezz/lvm-math-0402-a-qwen2.5-7b-instruct-b-qwen2.5-1.5b-instruct`,
GSM8K 50 questions x 16 samples, `max_tokens=6000`, temperature 1.0,
top-p 1.0, min-p 0.01. Baseline uses `top_k=-1`; LenVM uses `top_k=5`,
`value_mode=centered_exp`, `value_scale=0.001`, `gamma=0.997`.

`value_scale=0.001` is used as a near-neutral active-LenVM configuration. In
the optimized path, `centered_exp` with scale 0 is a true no-op and skips LenVM
entirely, so scale 0 is not useful for measuring active LenVM overhead.

| run | wall clock | avg completion tokens / choice | acc_first | acc_any |
| --- | ---: | ---: | ---: | ---: |
| baseline | 19.84 s | 295.96 | 0.88 | 1.00 |
| optimized LenVM | 80.02 s | 298.36 | 0.90 | 0.96 |
| ratio | 4.03x slower | +0.81% | - | - |

Compared with the original PR #2 reference result, 19.22 s -> 87.44 s
(4.55x slower), this reduces the guided run wall clock by 8.5%, the slowdown
ratio by 11.4%, and the incremental LenVM overhead by 11.8%.

The model memory reported by SGLang on the benchmark node was 14.30 GB for the
bf16 7B base model and 3.03 GB for the bf16 1.5B LenVM value model.

## Time breakdown

Small profiling run: same 7B/1.5B model pair, GSM8K 10 questions x 4 samples,
`top_k=5`, scale 0.001, with `SGLANG_LVM_TIMING=1`.

Baseline `Sampler.forward` is about 0.70-0.77 ms per call; most of that is the
FlashInfer sample backend. With LenVM active, `Sampler.forward` rises to about
13-15 ms after warmup, and `LvmGuidedSampler.apply` accounts for roughly 97% of
that time.

The final timing log at 800 guided calls reports:

| component | average time |
| --- | ---: |
| `Sampler.forward` | 13.22 ms |
| `LvmGuidedSampler.apply` | 12.85 ms |
| `build_pending` | 1.31 ms |
| fused extend+candidate path | 1.10 ms |
| fallback prefix extend | 9.23 ms |
| fallback candidate launch | 8.84 ms |
| GPU guidance application | 1.08 ms |

The branch-specific fallback timings are not additive with `apply_total`, but
they show the remaining bottleneck clearly: most residual latency comes from
requests that still fall back to two LVM forwards, one to extend the LenVM KV
cache and one to score candidate tokens.

## What changed

- Add a request-level fast precheck before initializing the in-process LenVM
  provider. Batches with no active value-guidance request return immediately,
  which avoids polluting baseline runs.
- Treat neutral guidance settings as no-ops: `centered_exp` / `value_bias`
  scale 0, `mul` scale <= 0 or scale 1, and other expectation modes at scale 1.
- Keep compacted candidate ids, probabilities, and masks on GPU for the common
  expectation-guidance path. Only row metadata is copied to CPU.
- Apply guidance in place and return only rows that changed, preserving
  top-k/top-p/min-p filtering for unmodified rows without cloning the full
  `[batch, vocab]` probability tensor.
- Add a fused in-process LenVM path that extends a tiny prefix delta and scores
  candidates in one forward when the request layout supports it.
- Add FlashInfer prefill argument generation for tree-value attention masks and
  vectorize the candidate self-attention diagonal.
- Avoid `.tolist()` GPU synchronization in Qwen2/Qwen3/Qwen2.5-VL LenVM value
  slicing by carrying prefix and candidate lengths in the tree-value spec.
- Cache request EOS ids for repeated candidate filtering.
- Add opt-in timing logs controlled by `SGLANG_LVM_TIMING`,
  `SGLANG_LVM_TIMING_INTERVAL`, and `SGLANG_LVM_TIMING_SKIP_CALLS`.

## Validation

Focused unit coverage was added for:

- GPU candidate compaction and CPU fallback candidate lists.
- No-op scale skipping.
- Baseline batches not initializing the in-process LenVM runner.
- Mixed value-guidance modes on the GPU path.
- Fused tree-value prefix/candidate mask construction.
- FlashInfer prefill argument generation.

Commands run:

```bash
python -m compileall -q \
  sglang-LenVM/python/sglang/srt/layers/sampler.py \
  sglang-LenVM/python/sglang/srt/lvm/lvm_guided_sampling.py \
  sglang-LenVM/python/sglang/srt/lvm/lvm_inproc_runner.py \
  sglang-LenVM/python/sglang/srt/lvm/lvm_value_utils.py \
  sglang-LenVM/python/sglang/srt/lvm/tree_value_spec.py \
  sglang-LenVM/python/sglang/srt/models/qwen2_lvm.py \
  sglang-LenVM/python/sglang/srt/models/qwen3_lvm.py \
  sglang-LenVM/python/sglang/srt/models/qwen2_5_vl_lvm.py

git diff --check
```

The focused tests were also executed directly through `.venv-infer/bin/python`
because the workspace does not currently have `pytest` installed.

Slurm benchmark artifacts:

- `results/timing/pr7b_q50n16_s001_84065_20260523_211347/`
- `results/timing/pr7b_prof_s001_84067_20260523_212103/`

## Remaining optimization target

The fused path is still all-or-nothing at the batch level in many situations.
The profiling run reached only 63 fused calls out of 800 guided calls. The next
high-value optimization is to split each batch into fusible and fallback rows:
run the fused path for eligible rows and use the two-phase path only for the
rest. That should directly attack the remaining prefix-extend and
candidate-launch overhead.
