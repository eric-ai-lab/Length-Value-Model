# https://huggingface.co/collections/namezz/length-value-model

source .venv-train/bin/activate

## Data
hf download namezz/LenVM-Data --local-dir ./data_generation/LenVM-Data --repo-type dataset

## Model
### Base generation model: Qwen2.5-7B-Instruct
### Length Value Model base model: Qwen2.5-0.5B-Instruct
### Example:
### hf download <model_name> \
###   --repo-type model \
###   --local-dir <model_path>
HF_MODEL_ID="namezz/lvm-a-qwen2.5-7b-instruct-b-qwen2.5-0.5b-instruct"
hf download $HF_MODEL_ID --local-dir ./models/namezz/lvm-a-qwen2.5-7b-instruct-b-qwen2.5-0.5b-instruct --repo-type model

