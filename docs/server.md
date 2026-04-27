# 1. Setup server with different models
## 1.1 Qwen3-30B-A3B-Instruct-2507
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 30011 \
  --tp-size 4 \
  --dp-size 1 \
  --context-length 30000 \
  --enable-lvm-guided-sampling \
  --lvm-guided-inproc \
  --lvm-guided-inproc-model-path ./models/namezz/lvm-math-0408-a-qwen3-30b-a3b-instruct-b-qwen3-1.7b-base \
  --lvm-guided-inproc-json-model-override-args '{"architectures":["Qwen3ForLengthValueModel"]}' \
  --disable-overlap-schedule \
  --mem-fraction-static 0.4 \
  --lvm-guided-inproc-mem-fraction-static 0.4 \
  --lvm-guided-fn sglang.srt.lvm.lvm_guided_sampling:lvm_combined_guidance
```

## 1.2 Qwen2.5-3B-Instruct
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --host 0.0.0.0 \
  --port 30011 \
  --tp-size 1 \
  --dp-size 4 \
  --context-length 30000 \
  --enable-lvm-guided-sampling \
  --lvm-guided-inproc \
  --lvm-guided-inproc-model-path ./models/namezz/lvm-rel-a-qwen2.5-3b-instruct-b-qwen2.5-3b-instruct \
  --lvm-guided-inproc-json-model-override-args '{"architectures":["Qwen2ForLengthValueModel"]}' \
  --disable-overlap-schedule \
  --mem-fraction-static 0.4 \
  --lvm-guided-inproc-mem-fraction-static 0.4 \
  --lvm-guided-fn sglang.srt.lvm.lvm_guided_sampling:lvm_combined_guidance
```

## 1.3 Qwen2.5-7B-Instruct
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 30011 \
  --tp-size 1 \
  --dp-size 4 \
  --context-length 30000 \
  --enable-lvm-guided-sampling \
  --lvm-guided-inproc \
  --lvm-guided-inproc-model-path ./models/namezz/lvm-instruct-0327-a-qwen2.5-7b-instruct-b-qwen2.5-1.5b-instruct \
  --lvm-guided-inproc-json-model-override-args '{"architectures":["Qwen2ForLengthValueModel"]}' \
  --disable-overlap-schedule \
  --mem-fraction-static 0.4 \
  --lvm-guided-inproc-mem-fraction-static 0.4 \
  --lvm-guided-fn sglang.srt.lvm.lvm_guided_sampling:lvm_combined_guidance
```

# 1.4 Qwen2.5-VL-7B-Instruct
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 30011 \
  --tp-size 1 \
  --dp-size 4 \
  --context-length 30000 \
  --enable-lvm-guided-sampling \
  --lvm-guided-inproc \
  --lvm-guided-inproc-model-path ./models/namezz/lvm-a-qwen2.5-vl-7b-instruct-b-qwen2.5-vl-3b-instruct \
  --lvm-guided-inproc-json-model-override-args '{"architectures":["Qwen2_5_VLForLengthValueModel"]}' \
  --disable-overlap-schedule \
  --mem-fraction-static 0.4 \
  --lvm-guided-inproc-mem-fraction-static 0.4 \
  --lvm-guided-fn sglang.srt.lvm.lvm_guided_sampling.lvm_combined_guidance
```

# 2. Test Server
## 2.1 w/o LenVM
```bash
curl -sS http://127.0.0.1:30011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"default",
    "messages":[
      {"role":"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
      {"role":"user","content":"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n11*11=?"}
    ],
    "max_tokens":1500,
    "temperature":1.0,
    "top_p":1.0,
    "min_p":0.01,
    "top_k":10,
    "n": 1
  }'
echo "\n"
```

## 2.2 w/ LenVM
```bash
curl -sS http://127.0.0.1:30011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"default",
    "messages":[
      {"role":"system","content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
      {"role":"user","content":"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n11*11=?"}
    ],
    "max_tokens":1500,
    "temperature":1.0,
    "top_p":1.0,
    "top_k":10,
    "min_p":0.01,
    "custom_params": {
      "value_scale": 1.0,
      "value_mode": "centered_exp"
    },
    "n": 1
  }'
echo "\n"
```