#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10010}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
MODEL_DIR="${MODEL_DIR:-./saves/a-qwen2.5-3b-instruct-b-qwen2.5-3b-instruct/demo_math_instruction}"


RESULTS_DIR="${RESULTS_DIR:-./results/visualization}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_LOG="${OUTPUT_LOG:-${RESULTS_DIR}/visualization_${TIMESTAMP}.md}"
OUTPUT_HTML="${OUTPUT_HTML:-${RESULTS_DIR}/visualization_${TIMESTAMP}_hover.html}"

source .venv-infer/bin/activate

mkdir -p "$RESULTS_DIR"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo "SGLang server is terminated"
  fi
}
trap cleanup EXIT

echo "Begin LenVM Visualization Demo"

python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --json-model-override-args '{"architectures":["Qwen2ForLengthValueModel"]}' \
  --is-embedding \
  --tp-size "$TP_SIZE" \
  --dp-size "$DP_SIZE" \
  --attention-backend triton \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --context-length "$CONTEXT_LENGTH" \
  --host "$HOST" \
  --port "$PORT" \
  --tokenizer-state-cleanup-delay-s 1 &
SERVER_PID=$!

for _ in {1..1200}; do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    break
  fi
  sleep 2
done

sleep 30

echo "SGLang server is ready"

python ./inference/visualization/real_sample_tree_value_visualization.py \
  --model-dir "$MODEL_DIR" \
  --server-url "http://127.0.0.1:$PORT" \
  --sample-file ./inference/visualization/default_text_sample.json \
  --lenvm-label "a-qwen2.5-3b-instruct-b-qwen2.5-3b-instruct" \
  --base-generator-label "Qwen3-30B-A3B-Instruct-2507" \
  --mode encode \
  --gamma 0.997 \
  --ignore-last-token \
  --max-steps -1 \
  --token-text-col-width 20 \
  --viz-rel-err \
  --viz-width 20 \
  --viz-max-pct 100 \
  --viz-clip-marker '▲' \
  --output-format markdown | tee "$OUTPUT_LOG"

python ./inference/visualization/length_curver_hover_demo.py \
  "$OUTPUT_LOG" \
  -o "$OUTPUT_HTML"

echo "Visualization log: $OUTPUT_LOG"
echo "Hover demo HTML: $OUTPUT_HTML"
