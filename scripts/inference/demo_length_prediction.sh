# https://huggingface.co/collections/namezz/length-value-model

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10008}"
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

### Math
python ./inference/length_prediction/eval_first_token_prediction.py \
  --server-url http://127.0.0.1:$PORT \
  --model-dir ./saves/a-qwen2.5-3b-instruct-b-qwen2.5-3b-instruct/demo_math_instruction \
  --data ./data_generation/LenVM-Data/Qwen2.5-3B-Instruct/deepmath-103k/demo_test_s128_n4_t1.0_p1.0_m3000.grouped.jsonl \
  --output-dir ./results/length_prediction/math \
  --dataset-name deepmath_103k \
  --gamma 0.997 \
  --expected-samples-per-question 4 \
  --method tree_value \
  --batch-size 16 \
  --max-concurrency 32 \
  --fail-fast \
  --max-prompt-tokens 3000


### Instruction-Following
python ./inference/length_prediction/eval_first_token_prediction.py \
  --server-url http://127.0.0.1:$PORT \
  --model-dir ./saves/a-qwen2.5-3b-instruct-b-qwen2.5-3b-instruct/demo_math_instruction \
  --data ./data_generation/LenVM-Data/Qwen2.5-3B-Instruct/wildchat/demo_test_s128_n4_t1.0_p1.0_m3000.grouped.jsonl \
  --output-dir ./results/length_prediction/wildchat \
  --dataset-name wildchat \
  --gamma 0.997 \
  --expected-samples-per-question 4 \
  --method tree_value \
  --batch-size 16 \
  --max-concurrency 32 \
  --fail-fast \
  --max-prompt-tokens 3000

kill $SERVER_PID
echo "SGLang server is terminated"