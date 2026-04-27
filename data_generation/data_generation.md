# Advanced data generation reference

This page collects detailed data-generation commands for full or dataset-specific LenVM runs. For the standard demo path, start from the root [README](../README.md) or run:

```bash
bash scripts/data_generation/demo.sh
```

For the full end-to-end pipeline, see [docs/workflows.md](../docs/workflows.md).

# 1. Download and save the OpenCodeReasoning-2 dataset
```bash
python data_generation/utils/download_save_opencodereasoning2.py \
  --languages python \
  --output_dir data_generation/LenVM-Data/OpenCodeReasoning-2-updated
```

# 2. Launch the SGLang server
```bash
MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507
TP_SIZE=1
DP_SIZE=2
CONTEXT_LENGTH=30000
HOST=0.0.0.0
PORT=10005
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --tp-size $TP_SIZE \
  --dp-size $DP_SIZE \
  --context-length $CONTEXT_LENGTH \
  --host $HOST \
  --port $PORT &
```

# 3. Data generation settings
Select one of the configs below.

## 3.1 Qwen3-30B-A3B-Instruct-2507
```bash
MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507
SERVER_PORT=10005
TRAIN_SAMPLES=-1
TEST_SAMPLES=8192
TRAIN_SAMPLES_PER_QUESTION=16
TEST_SAMPLES_PER_QUESTION=4
MAX_TOKENS=16000
TEMPERATURE=1.0
TOP_P=1.0
MODEL_DIR_NAME=$(basename $MODEL_PATH)
```


## 3.2 Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct and Qwen2.5-VL-7B-Instruct
```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
SERVER_PORT=10005
TRAIN_SAMPLES=-1
TEST_SAMPLES=8192
TRAIN_SAMPLES_PER_QUESTION=16
TEST_SAMPLES_PER_QUESTION=4
MAX_TOKENS=5000
TEMPERATURE=1.0
TOP_P=1.0
MODEL_DIR_NAME=$(basename $MODEL_PATH)
```


# 4. Generate the datasets
## 4.1 Generate the DeepMath-103K dataset
```bash
python -m data_generation.data_generator.main \
  --dataset-name zwhe99/DeepMath-103K \
  --dataset-split train \
  --train-samples $TRAIN_SAMPLES \
  --test-samples $TEST_SAMPLES \
  --train-samples-per-question $TRAIN_SAMPLES_PER_QUESTION \
  --test-samples-per-question $TEST_SAMPLES_PER_QUESTION \
  --train-raw-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/deepmath-103k/train_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.jsonl \
  --train-group-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/deepmath-103k/train_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --test-raw-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/deepmath-103k/test_s${TEST_SAMPLES}_n${TEST_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.jsonl \
  --test-group-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/deepmath-103k/test_s${TEST_SAMPLES}_n${TEST_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --keep-columns final_answer,difficulty,topic \
  --model-name $MODEL_PATH \
  --temperature $TEMPERATURE \
  --top-p $TOP_P \
  --max-tokens $MAX_TOKENS \
  --openai-base-url http://127.0.0.1:$SERVER_PORT/v1 \
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
```

## 4.2 Generate the WildChat dataset
```bash
python -m data_generation.data_generator.main \
  --dataset-name allenai/WildChat \
  --dataset-split train \
  --train-samples $TRAIN_SAMPLES \
  --test-samples $TEST_SAMPLES \
  --train-samples-per-question $TRAIN_SAMPLES_PER_QUESTION \
  --test-samples-per-question $TEST_SAMPLES_PER_QUESTION \
  --train-raw-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/wildchat/train_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.jsonl \
  --train-group-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/wildchat/train_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --test-raw-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/wildchat/test_s${TEST_SAMPLES}_n${TEST_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.jsonl \
  --test-group-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/wildchat/test_s${TEST_SAMPLES}_n${TEST_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --keep-columns conversation_id,language,toxic,redacted \
  --model-name $MODEL_PATH \
  --temperature $TEMPERATURE \
  --top-p $TOP_P \
  --max-tokens $MAX_TOKENS \
  --openai-base-url http://127.0.0.1:$SERVER_PORT/v1 \
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
```

## 4.3 Generate the OpenCodeReasoning-2 dataset
```bash
python -m data_generation.data_generator.main \
  --dataset-name nvidia/OpenCodeReasoning-2 \
  --dataset-split python \
  --load-from-local \
  --local-dataset-path ./data_generation/LenVM-Data/OpenCodeReasoning-2-updated \
  --train-samples $TRAIN_SAMPLES \
  --test-samples $TEST_SAMPLES \
  --train-samples-per-question $TRAIN_SAMPLES_PER_QUESTION \
  --test-samples-per-question $TEST_SAMPLES_PER_QUESTION \
  --train-raw-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/open_code_reasoning_2/train_python_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.jsonl \
  --train-group-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/open_code_reasoning_2/train_python_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --test-raw-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/open_code_reasoning_2/test_python_s${TEST_SAMPLES}_n${TEST_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.jsonl \
  --test-group-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/open_code_reasoning_2/test_python_s${TEST_SAMPLES}_n${TEST_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --keep-columns pass_rate,source,dataset,split,difficulty,index,id,question_id \
  --model-name $MODEL_PATH \
  --temperature $TEMPERATURE \
  --top-p $TOP_P \
  --max-tokens $MAX_TOKENS \
  --openai-base-url http://127.0.0.1:$SERVER_PORT/v1 \
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
```

## 4.4 Generate the R1-Onevision dataset (VL model)
```bash
python -m data_generation.data_generator.main \
  --dataset-name Fancy-MLLM/R1-Onevision \
  --dataset-split train \
  --train-samples $TRAIN_SAMPLES \
  --test-samples $TEST_SAMPLES \
  --train-samples-per-question $TRAIN_SAMPLES_PER_QUESTION \
  --test-samples-per-question $TEST_SAMPLES_PER_QUESTION \
  --train-raw-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/r1_onevision/train_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.jsonl \
  --train-group-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/r1_onevision/train_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --test-raw-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/r1_onevision/test_s${TEST_SAMPLES}_n${TEST_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.jsonl \
  --test-group-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/r1_onevision/test_s${TEST_SAMPLES}_n${TEST_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --image-dir ./data_generation/LenVM-Data/$MODEL_DIR_NAME/r1_onevision/images \
  --keep-columns id \
  --model-name $MODEL_PATH \
  --temperature $TEMPERATURE \
  --top-p $TOP_P \
  --max-tokens $MAX_TOKENS \
  --openai-base-url http://127.0.0.1:$SERVER_PORT/v1 \
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
```

# 5. Analyze the length distribution of the generated datasets
```bash
DATASET_NAME=deepmath-103k
python data_generation/utils/distribution_analysis.py \
  --data-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/$DATASET_NAME/train_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl  \
  --output-path ./data_generation/LenVM-Data/$MODEL_DIR_NAME/$DATASET_NAME/distribution_analysis.png
```

# 6. Downsample the generated data to reduce the number of samples per question
```bash
DATASET_NAME=deepmath-103k
ORIGINAL_SAMPLES_PER_QUESTION=2
DOWN_SAMPLES_PER_QUESTION=1
python data_generation/utils/downsampler.py \
  --input ./data_generation/LenVM-Data/$MODEL_DIR_NAME/$DATASET_NAME/train_s${TRAIN_SAMPLES}_n${TRAIN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --output ./data_generation/LenVM-Data/$MODEL_DIR_NAME/$DATASET_NAME/train_s${TRAIN_SAMPLES}_n${DOWN_SAMPLES_PER_QUESTION}_t${TEMPERATURE}_p${TOP_P}_m${MAX_TOKENS}.grouped.jsonl \
  --num-questions -1 \
  --num-samples $DOWN_SAMPLES_PER_QUESTION \
  --group-size $ORIGINAL_SAMPLES_PER_QUESTION \
  --group-mode random \
  --within-mode random \
  --seed 42 
```
