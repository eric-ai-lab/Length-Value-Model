#!/usr/bin/env bash
# Compare end-to-end and per-step decoding cost: baseline vs LenVM-guided.
#
# Two server lifecycles (so per-step timer only captures the configuration
# under test):
#   1. SGLang with --enable-lvm-guided-sampling OFF -> baseline timing.
#   2. SGLang with LenVM enabled + 7B base + 1.5B LenVM -> guided timing.
# Both servers point SGLANG_LVM_TIMING_LOG at distinct JSONL files. The client
# replays the same GSM8K prompt set against each, then analyze.py emits a
# CSV/JSON table and per-step decomposition plot.
#
# Run from repository root.

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10020}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.4}"
LENVM_MEM_FRACTION_STATIC="${LENVM_MEM_FRACTION_STATIC:-0.4}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LENVM_MODEL="${LENVM_MODEL:-./models/namezz/lvm-math-0402-a-qwen2.5-7b-instruct-b-qwen2.5-1.5b-instruct}"
DATASET="${DATASET:-gsm8k}"
MAX_QUESTIONS="${MAX_QUESTIONS:-50}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-50}"
N_SAMPLES="${N_SAMPLES:-16}"
MAX_TOKENS="${MAX_TOKENS:-6000}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MIN_P="${MIN_P:-0.01}"
LENVM_TOP_K="${LENVM_TOP_K:-5}"
LENVM_VALUE_SCALE="${LENVM_VALUE_SCALE:-0}"
LENVM_VALUE_MODE="${LENVM_VALUE_MODE:-centered_exp}"
LENVM_VALUE_GAMMA="${LENVM_VALUE_GAMMA:-0.997}"

RESULTS_DIR="${RESULTS_DIR:-./results/timing/$(basename "$BASE_MODEL")_vs_$(basename "$LENVM_MODEL")}"
mkdir -p "$RESULTS_DIR"

# ---- helpers ---------------------------------------------------------------

wait_for_server() {
  local port="$1"
  for _ in $(seq 1 1200); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then return 0; fi
    sleep 2
  done
  echo "Server on port ${port} failed to become ready" >&2
  return 1
}

kill_server() {
  local pid="$1"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || return 0
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

# ---- stage 1: baseline (LenVM disabled) ------------------------------------

echo "==> Stage 1: baseline server (no LenVM)"
source .venv-infer/bin/activate

BASELINE_TIMING_LOG="$RESULTS_DIR/baseline.timing.jsonl"
: > "$BASELINE_TIMING_LOG"

SGLANG_LVM_TIMING_LOG="$BASELINE_TIMING_LOG" \
python -m sglang.launch_server \
  --model-path "$BASE_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --dp-size "$DP_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" &
SERVER_PID=$!
trap 'kill_server "$SERVER_PID"' EXIT

wait_for_server "$PORT"
echo "Baseline server ready"

source .venv-eval/bin/activate
python -m inference.timing.run_timing \
  --tag baseline \
  --server-url "http://127.0.0.1:$PORT" \
  --dataset-name "$DATASET" \
  --max-questions "$MAX_QUESTIONS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --n "$N_SAMPLES" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k -1 \
  --min-p "$MIN_P" \
  --output-dir "$RESULTS_DIR"

kill_server "$SERVER_PID"
trap - EXIT
SERVER_PID=""

# ---- stage 2: LenVM (in-proc guidance) -------------------------------------

echo "==> Stage 2: LenVM server (7B base + LenVM in-proc)"
source .venv-infer/bin/activate

LENVM_TIMING_LOG="$RESULTS_DIR/lenvm.timing.jsonl"
: > "$LENVM_TIMING_LOG"

SGLANG_LVM_TIMING_LOG="$LENVM_TIMING_LOG" \
python -m sglang.launch_server \
  --model-path "$BASE_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size "$TP_SIZE" \
  --dp-size "$DP_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --enable-lvm-guided-sampling \
  --lvm-guided-inproc \
  --lvm-guided-inproc-model-path "$LENVM_MODEL" \
  --lvm-guided-inproc-json-model-override-args '{"architectures":["Qwen2ForLengthValueModel"]}' \
  --disable-overlap-schedule \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --lvm-guided-inproc-mem-fraction-static "$LENVM_MEM_FRACTION_STATIC" \
  --lvm-guided-fn sglang.srt.lvm.lvm_guided_sampling:lvm_combined_guidance &
SERVER_PID=$!
trap 'kill_server "$SERVER_PID"' EXIT

wait_for_server "$PORT"
echo "LenVM server ready"

source .venv-eval/bin/activate
python -m inference.timing.run_timing \
  --tag lenvm \
  --server-url "http://127.0.0.1:$PORT" \
  --dataset-name "$DATASET" \
  --max-questions "$MAX_QUESTIONS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --n "$N_SAMPLES" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --top-k "$LENVM_TOP_K" \
  --min-p "$MIN_P" \
  --value-scale "$LENVM_VALUE_SCALE" \
  --value-mode "$LENVM_VALUE_MODE" \
  --value-gamma "$LENVM_VALUE_GAMMA" \
  --output-dir "$RESULTS_DIR"

kill_server "$SERVER_PID"
trap - EXIT
SERVER_PID=""

# ---- stage 3: analyze ------------------------------------------------------

echo "==> Stage 3: analyze"
python -m inference.timing.analyze \
  --results-dir "$RESULTS_DIR" \
  --base-model "$BASE_MODEL" \
  --lvm-model "$LENVM_MODEL"

echo "Done. Results in $RESULTS_DIR"
