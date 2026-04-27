# https://huggingface.co/collections/namezz/length-value-model

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10007}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.4}"
LENVM_MEM_FRACTION_STATIC="${LENVM_MEM_FRACTION_STATIC:-0.4}"


source .venv-infer/bin/activate

echo "Begin Tradeoff Evaluation"

# Launch the SGLang server
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --host $HOST \
  --port $PORT \
  --tp-size $TP_SIZE \
  --dp-size $DP_SIZE \
  --context-length $CONTEXT_LENGTH \
  --enable-lvm-guided-sampling \
  --lvm-guided-inproc \
  --lvm-guided-inproc-model-path ./saves/a-qwen2.5-3b-instruct-b-qwen2.5-3b-instruct/demo_math_instruction \
  --lvm-guided-inproc-json-model-override-args '{"architectures":["Qwen2ForLengthValueModel"]}' \
  --disable-overlap-schedule \
  --mem-fraction-static $MEM_FRACTION_STATIC \
  --lvm-guided-inproc-mem-fraction-static $LENVM_MEM_FRACTION_STATIC \
  --lvm-guided-fn sglang.srt.lvm.lvm_guided_sampling:lvm_combined_guidance &
SERVER_PID=$!


# Wait for the SGLang server to be ready
for _ in {1..1200}; do
  if curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null; then
    break
  fi
  sleep 2
done

sleep 30

echo "SGLang server is ready"

## Tradeoff Evaluation
source .venv-eval/bin/activate

python ./inference/tradeoff/sample_eval.py \
  --dataset-name gsm8k \
  --server-url http://127.0.0.1:$PORT \
  --output-dir ./results/tradeoff/Qwen2.5-3B-Instruct/gsm8k \
  --tag baseline_q50_n16_p1.0_topk-1_minp0.01 \
  --stage all \
  --max-questions 50 \
  --max-concurrency 50 \
  --request-timeout 600000 \
  --max-tokens 6000 \
  --temperature 1.0 \
  --top-p 1.0 \
  --n 16 \
  --http-backend aiohttp \
  --top-k -1 \
  --min-p 0.01

echo "Tradeoff evaluation baseline is complete"

SCALES=(-100 -10 -5 -2 0 2 5 10)
for VALUE_SCALE in "${SCALES[@]}"; do
  SCALE_TAG="${VALUE_SCALE//./_}"
  echo "Running LenVM evaluation with scale $VALUE_SCALE"
  python ./inference/tradeoff/sample_eval.py \
    --dataset-name gsm8k \
    --server-url http://127.0.0.1:$PORT \
    --output-dir ./results/tradeoff/Qwen2.5-3B-Instruct/gsm8k \
    --tag "lenvm_q50_n16_p1.0_topk5_minp0.01_gamma0.997_centered_exp_${SCALE_TAG}" \
    --stage all \
    --max-questions 50 \
    --max-concurrency 50 \
    --request-timeout 600000 \
    --max-tokens 6000 \
    --temperature 1.0 \
    --top-p 1.0 \
    --n 16 \
    --http-backend aiohttp \
    --top-k 5 \
    --min-p 0.01 \
    --value-scale "$VALUE_SCALE" \
    --value-mode centered_exp \
    --value-gamma 0.997
done

echo "Tradeoff evaluation LenVM is complete"

python ./inference/tradeoff/budget_eval.py \
    --responses ./results/tradeoff/Qwen2.5-3B-Instruct/gsm8k/gsm8k.baseline_q50_n16_p1.0_topk-1_minp0.01.responses.jsonl \
    --tokenizer Qwen/Qwen2.5-3B-Instruct \
    --output-dir ./results/tradeoff/Qwen2.5-3B-Instruct/gsm8k \
    --budgets 350 400 500 600 700 800 900 1000 1500 2000 5000 10000

echo "Budget evaluation is complete"

python3 ./inference/tradeoff/plot_baseline_vs_lenvm.py \
  --baseline-dir ./results/tradeoff/Qwen2.5-3B-Instruct/gsm8k \
  --centered-dir ./results/tradeoff/Qwen2.5-3B-Instruct/gsm8k \
  --output-dir ./results/tradeoff/Qwen2.5-3B-Instruct/gsm8k/tradeoff_plot

echo "Tradeoff evaluation plot is complete"

kill "$SERVER_PID"
echo "SGLang server is terminated"