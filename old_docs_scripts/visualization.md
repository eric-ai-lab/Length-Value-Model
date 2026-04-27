# Visualization

This page documents the scripts under `inference/visualization/`.

## 1. One-command demo

`scripts/visualization/demo.sh` runs the text-only visualization, writes the markdown log to `results/`, and then builds a hover HTML demo from that log.

```bash
./scripts/visualization/demo.sh
```

Default outputs:
- `results/visualization_<timestamp>.md`
- `results/visualization_<timestamp>_hover.html`

You can override common settings with environment variables:

```bash
MODEL_DIR=Qwen/Qwen3-30B-A3B-Instruct-2507 \
SERVER_URL=http://127.0.0.1:30010 \
GAMMA=0.9998 \
MODE=encode \
./scripts/visualization/demo.sh
```

## 2. Text-only value visualization

Use this script when you want to inspect the length-value outputs for a normal LLM sample.

```bash
python ./inference/visualization/real_sample_tree_value_visualization.py \
  --model-dir Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --server-url http://127.0.0.1:30010 \
  --mode encode \
  --gamma 0.9998 \
  --ignore-last-token \
  --max-steps -1 \
  --token-text-col-width 10 \
  --viz-rel-err \
  --viz-width 20 \
  --viz-max-pct 100 \
  --viz-clip-marker '▲' \
  --output-format markdown
```

What it does:
- loads a fixed text conversation sample
- tokenizes it with the model chat template
- queries the server for tokenwise length-value predictions
- prints a table with token id, token text, value prediction, converted length, true remaining length, and relative error

Main modes:
- `--mode encode`: one `/encode` call over the full sequence
- `--mode tree_value`: batched `/tree_value` calls with prefix + next-token candidates

Useful options:
- `--include-prompt-last`: also print the token before the assistant answer
- `--ignore-last-token`: skip the final assistant token after EOS stripping
- `--viz-rel-err`: add an ASCII error bar column

## 3. VLM value visualization with SGLang server

Use this script when you want to inspect a multimodal sample through the SGLang embedding server.

```bash
python ./inference/visualization/real_sample_tree_value_visualization_vl.py \
  --model-dir Qwen/Qwen2.5-VL-7B-Instruct \
  --server-url http://127.0.0.1:30010 \
  --image-path ./inference/visualization/mathvista_idx63.png \
  --gamma 0.997 \
  --ignore-last-token \
  --max-steps -1 \
  --token-text-col-width 10 \
  --viz-rel-err \
  --viz-width 20 \
  --viz-max-pct 100 \
  --viz-clip-marker '▲' \
  --output-format markdown
```

What it does:
- loads a fixed VLM sample from MathVista idx=63
- includes both image and text in the chat template
- uses the server-side VL tokenizer and embedding endpoint
- prints the same per-token length-value table as the text-only script

Important detail:
- `--model-dir` should point to the tokenizer/processor-compatible base model
- the script sends prompt-only and full-conversation requests to handle server-side image token expansion correctly

## 4. VLM value visualization with Hugging Face

Use this script when you want a fully local check without SGLang.

```bash
CUDA_VISIBLE_DEVICES=2 python ./inference/visualization/real_sample_tree_value_visualization_vl_hf.py \
  --model-path models/namezz/lvm-a-qwen2.5-vl-7b-instruct-b-qwen2.5-vl-3b-instruct \
  --image-path ./inference/visualization/mathvista_idx63.png \
  --gamma 0.997 \
  --ignore-last-token \
  --viz-rel-err \
  --output-format markdown
```

What it does:
- loads the VLM backbone and `value_head.safetensors` locally
- runs one forward pass on the full conversation
- applies the value head to the final hidden states
- prints the per-token length-value table

When to use it:
- debugging the checkpoint directly
- checking the value head behavior without server-side caching or tokenizer differences
- comparing HF output against the SGLang server output

## 5. Hover demo HTML generator

Use this script when you want an interactive HTML page from a text log.

```bash
python ./inference/visualization/length_curver_hover_demo.py \
  --input-file output.txt \
  --output-file output_hover_demo.html
```

What it does:
- parses a visualization log
- extracts the token rows and the question text
- builds a standalone HTML page with:
  - predicted curve
  - ground-truth curve
  - token text panel
  - hover/click linking between chart and tokens

Typical workflow:
1. run one of the visualization scripts above
2. save the terminal output to a text file
3. pass that file to this demo builder
4. open the generated HTML in a browser

## 6. Practical differences between the three visualization scripts

- `real_sample_tree_value_visualization.py` is for **text-only** LLM samples.
- `real_sample_tree_value_visualization_vl.py` is for **VLM samples through SGLang**.
- `real_sample_tree_value_visualization_vl_hf.py` is for **VLM samples through local Hugging Face loading**.
- `length_curver_hover_demo.py` is for **interactive viewing** of the printed log.

## 7. Common notes

- The scripts all use the LenVM length conversion formula from value prediction to remaining length.
- The printed `true` column is the remaining answer length used for comparison.
- The last assistant token often has larger error because it may be excluded from training by `value_mask`.
