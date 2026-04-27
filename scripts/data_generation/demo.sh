set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-30000}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10005}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1024}"
TEST_SAMPLES="${TEST_SAMPLES:-128}"
TRAIN_SAMPLES_PER_QUESTION="${TRAIN_SAMPLES_PER_QUESTION:-4}"
TEST_SAMPLES_PER_QUESTION="${TEST_SAMPLES_PER_QUESTION:-4}"
MAX_TOKENS="${MAX_TOKENS:-3000}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

source .venv-infer/bin/activate

# Launch the SGLang server
python -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --tp-size $TP_SIZE \
  --dp-size $DP_SIZE \
  --context-length $CONTEXT_LENGTH \
  --host $HOST \
  --port $PORT &
SERVER_PID=$!

# Wait for the SGLang server to be ready
for _ in {1..1200}; do
  if curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null; then
    break
  fi
  sleep 2
done


# Generate the datasets
MODEL_DIR_NAME=$(basename $MODEL_PATH)

## Generate the DeepMath-103K dataset
python -m data_generation.data_generator.main \
  --dataset-name zwhe99/DeepMath-103K \
  --dataset-split train \
  --train-samples $TRAIN_SAMPLES \
  --test-samples $TEST_SAMPLES \
  --train-samples-per-question $TRAIN_SAMPLES_PER_QUESTION \
  --test-samples-per-question $TEST_SAMPLES_PER_QUESTION \
  --train-raw-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/deepmath-103k/demo_train_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.jsonl \
  --train-group-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/deepmath-103k/demo_train_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --test-raw-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/deepmath-103k/demo_test_s${TEST_SAMPLES}_n${TEST_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.jsonl \
  --test-group-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/deepmath-103k/demo_test_s${TEST_SAMPLES}_n${TEST_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --keep-columns final_answer,difficulty,topic \
  --model-name $MODEL_PATH \
  --temperature $TEMPERATURE \
  --top-p $TOP_P \
  --max-tokens $MAX_TOKENS \
  --openai-base-url http://127.0.0.1:$PORT/v1 \
  --openai-api-key "empty" \
  --max-concurrency 5000 \
  --max-connections 5000 \
  --max-keepalive-connections 5000 \
  --group-by-index \
  --max-retries 2 \
  --retry-initial-delay 1.0 \
  --retry-max-delay 20.0 \
  --request-timeout 6000.0 \
  --save-batch-size 5000

## Generate the WildChat dataset
python -m data_generation.data_generator.main \
  --dataset-name allenai/WildChat \
  --dataset-split train \
  --train-samples $TRAIN_SAMPLES \
  --test-samples $TEST_SAMPLES \
  --train-samples-per-question $TRAIN_SAMPLES_PER_QUESTION \
  --test-samples-per-question $TEST_SAMPLES_PER_QUESTION \
  --train-raw-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/wildchat/demo_train_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.jsonl \
  --train-group-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/wildchat/demo_train_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --test-raw-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/wildchat/demo_test_s${TEST_SAMPLES}_n${TEST_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.jsonl \
  --test-group-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/wildchat/demo_test_s${TEST_SAMPLES}_n${TEST_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --keep-columns conversation_id,language,toxic,redacted \
  --model-name $MODEL_PATH \
  --temperature $TEMPERATURE \
  --top-p $TOP_P \
  --max-tokens $MAX_TOKENS \
  --openai-base-url http://127.0.0.1:$PORT/v1 \
  --openai-api-key None \
  --max-concurrency 5000 \
  --max-connections 5000 \
  --max-keepalive-connections 5000 \
  --group-by-index \
  --max-retries 2 \
  --retry-initial-delay 1.0 \
  --retry-max-delay 20.0 \
  --request-timeout 6000.0 \
  --save-batch-size 5000

kill "$SERVER_PID"
