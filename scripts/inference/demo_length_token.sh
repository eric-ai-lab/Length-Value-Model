set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10009}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.4}"
LENVM_MEM_FRACTION_STATIC="${LENVM_MEM_FRACTION_STATIC:-0.4}"

source .venv-infer/bin/activate

echo "Begin Length Prediction Evaluation"

# Length Prediction

## server for length prediction
python -m sglang.launch_server \
  --model-path ./saves/a-qwen2.5-3b-instruct-b-qwen2.5-3b-instruct/demo_math_instruction \
  --json-model-override-args '{"architectures":["Qwen2ForLengthValueModel"]}' \
  --is-embedding \
  --tp-size $TP_SIZE \
  --dp-size $DP_SIZE \
  --attention-backend triton \
  --mem-fraction-static $MEM_FRACTION_STATIC \
  --context-length $CONTEXT_LENGTH \
  --host $HOST \
  --port $PORT \
  --tokenizer-state-cleanup-delay-s 1 &
SERVER_PID=$!

for _ in {1..1200}; do
  if curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null; then
    break
  fi
  sleep 2
done

sleep 30

echo "SGLang server is ready"

source .venv-eval/bin/activate

# analysis
python ./inference/length_token/analyze_length_token.py \
  --server-url http://127.0.0.1:$PORT \
  --model-dir ./saves/a-qwen2.5-3b-instruct-b-qwen2.5-3b-instruct/demo_math_instruction \
  --data ./data_generation/LenVM-Data/Qwen2.5-3B-Instruct/deepmath-103k/demo_test_s128_n4_t1.0_p1.0_m3000.grouped.jsonl \
  --output-dir ./results/length_token_analysis \
  --score value_gamma_td \
  --upper-threshold 0.01 \
  --lower-threshold -0.01 \
  --max-samples 10000 \
  --batch-size 1 \
  --next-token-window 16 \
  --max-concurrency 1 \
  --wordcloud-top-k 80 \
  --wordcloud

kill $SERVER_PID
echo "SGLang server is terminated"