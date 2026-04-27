# https://huggingface.co/collections/namezz/length-value-model

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10006}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.4}"
LENVM_MEM_FRACTION_STATIC="${LENVM_MEM_FRACTION_STATIC:-0.4}"


source .venv-infer/bin/activate

echo "Begin LIFEBench Evaluation"

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


# Run the inference
## LIFEBench Evaluation
source .venv-eval/bin/activate

cd ./inference/LIFEBench/

python run_exp.py \
  --model_type SGLang_Local_No_Length_Control \
  --output_file_dir ../../results/LIFEBench/token/demo-qwen2.5-3b-instruct-baseline_token \
  --length_metric token \
  --length_constraints 128 \
  --meta_data_path ./data/data_lite_no_summary.jsonl \
  --control_methods "equal to" "at most" "at least" \
  --max_concurrency 32

echo "LIFEBench baseline evaluation is complete"

python run_exp.py \
  --model_type SGLang_Local \
  --output_file_dir ../../results/LIFEBench/token/demo-a-qwen2.5-3b-instruct-b-qwen2.5-3b-instruct_token \
  --length_metric token \
  --length_constraints 128 \
  --meta_data_path data/data_lite_no_summary.jsonl \
  --control_methods "equal to" "at most" "at least" \
  --max_concurrency 32

echo "LIFEBench LenVM evaluation is complete"

python evaluate.py  \
  --data_dir ../../results/LIFEBench/token \
  --length_metric token \
  --output_csv ../../results/LIFEBench/token/token.csv

echo "LIFEBench evaluation is complete"

cd ../../

kill "$SERVER_PID"
echo "SGLang server is terminated"
