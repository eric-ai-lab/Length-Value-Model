import json
import re

import yaml

from exp.universe_api.AnthropicModel import AnthropicModel
from exp.universe_api.AzureOpenaiModel import AzureOpenaiModel
from exp.universe_api.GenAiModel import GenAiModel
from exp.universe_api.HTTPBaseModel import HTTPBaseModel
from exp.universe_api.OpenAiBaseModel import OpenAiBaseModel
from exp.universe_api.OpenAiStreamModel import OpenAiStreamModel
from exp.universe_api.PipelineModel import PipelineModel
from exp.universe_api.SuriModel import SuriModel
from exp.universe_api.LongWriterModel import LongWriterModel


def _soften_equal_to_wording_for_sglang(task: str, *, lang: str) -> str:
    """Rewrite strict 'equal to' length wording into 'approximately' for SGLang runs."""
    if not isinstance(task, str) or not task:
        return task
    if lang == "en":
        # Common patterns in this benchmark:
        # - "must be equal to 16 tokens long."
        # - "must be equal to 32 tokens long."
        task = task.replace("must be equal to ", "should be approximately ")
        task = task.replace("must equal ", "should be approximately ")
        # Only replace explicit word-count requirements, avoid unrelated "word" mentions.
        task = re.sub(r"(\d+)\s+words?\s+long", r"\1 tokens long", task)
        task = re.sub(r"(\d+)\s+words?\b", r"\1 tokens", task)
        return task
    if lang == "cn":
        # Common patterns:
        # - "长度 必须等于16token"
        # - "长度必须等于16token"
        task = task.replace("必须等于", "应约为")
        # Only replace explicit count units, avoid unrelated "字" mentions.
        task = re.sub(r"(\d+)\s*字", r"\1 token", task)
        return task
    return task


def _convert_length_wording(task: str, *, lang: str, length_metric: str) -> str:
    if not isinstance(task, str) or not task:
        return task

    metric = str(length_metric or "word").strip().lower()
    if metric == "word":
        return task
    if metric != "token":
        raise ValueError(f"Unsupported length_metric: {length_metric}. Use 'word' or 'token'.")

    if lang == "en":
        task = re.sub(r"(\d+)\s+words?\s+long", r"\1 tokens long", task)
        task = re.sub(r"(\d+)\s+words?\b", r"\1 tokens", task)
        task = re.sub(r"(\d+)\s+tokens?\s+long\b", r"\1 tokens long", task)
        # task = re.sub(r"(\d+)\s+tokens\b", r"\1 tokens (not words)", task)
        # if "A token is the model's internal text unit" not in task:
        #     task += "\n\nA token is the model's internal text unit, not a word. One word may contain multiple tokens, and punctuation also counts."
        return task
    if lang == "cn":
        task = re.sub(r"(\d+)\s*字", r"\1 token", task)
        task = task.replace("字数", "token数")
        task = re.sub(r"(\d+)\s*token\b", r"\1", task)
        # if "token 是模型内部的文本单位" not in task:
        #     task += "\n\n说明：token 是模型内部的文本单位，不等于单词或字。一个单词可能对应多个 token，标点也计入 token。"
        # return task
    return task


def try_api_call(model, meta_data, output_file_dir, control_method, length_constraint, max_concurrency, length_metric):
    model.prepare_dir(output_file_dir, control_method, length_constraint)
    model.set_length_metric(length_metric)
    model.get_cache_data(meta_data, max_concurrency=max_concurrency)


def select_model(api_type, key_config_file, param_config_file):
    with open(param_config_file, 'r', encoding='utf-8') as file:
        model_params = yaml.safe_load(file).get(api_type, {})
    with open(key_config_file, 'r', encoding='utf-8') as file:
        api_params = yaml.safe_load(file).get(api_type, {})

    openai_base_api_type = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-5.4", "gpt-5.4-thinking", "o1-mini", "o4-mini", "deepseek-v3", "deepseek-r1", "sglang-local","sglang-local-1", "sglang-local-no-length-control","o3-mini", "Qwen3-235B-A22B-Instruct-2507"]
    openai_stream_api_type = ["Qwen3-235B-A22B", "Qwen3-235B-A22B-Thinking", "Qwen3-32B", "Qwen3-32B-Thinking"]
    http_base_api_type = ["doubao-1.5-pro", "doubao-1.5-thinking-pro"]
    azure_base_api_type = []
    genai_base_api_type = ["gemini-2.0-flash", "gemini-2.0-flash-thinking", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-3.1-flash-lite-preview", "gemini-3.1-pro-preview"]
    anthropic_base_api_type = ["claude-3.5-sonnet", "claude-3.7-sonnet", "claude-3.7-sonnet-thinking", "claude-sonnet-4-6", "claude-sonnet-4-6-thinking", "claude-opus-4-6", "claude-opus-4-6-thinking", "claude-sonnet-4", "claude-opus-4"]
    pipeline_base_api_type = ["Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct",
                              "Qwen2.5-72B-Instruct", "glm-4-9b",
                              "Mistral-7B-Instruct"]
    long_writer_api_type = ["LongWriter-glm4-9b", "LongWriter-llama3.1-8b"]
    suri_api_type = ["suri-i-orpo"]
    if api_type in openai_base_api_type:
        model = OpenAiBaseModel(api_type, model_params, api_params)
    elif api_type in openai_stream_api_type:
        model = OpenAiStreamModel(api_type, model_params, api_params)
    elif api_type in pipeline_base_api_type:
        model = PipelineModel(api_type, model_params)
    elif api_type in azure_base_api_type:
        model = AzureOpenaiModel(api_type, model_params, api_params)
    elif api_type in genai_base_api_type:
        model = GenAiModel(api_type, model_params, api_params)
    elif api_type in anthropic_base_api_type:
        model = AnthropicModel(api_type, model_params, api_params)
    elif api_type in http_base_api_type:
        model = HTTPBaseModel(api_type, model_params, api_params)
    elif api_type in long_writer_api_type:
        model = LongWriterModel(api_type, model_params)
    elif api_type in suri_api_type:
        model = SuriModel(api_type, model_params)
    else:
        raise Exception(f"No api type names: {api_type}")
    return model


def exp_request(meta_data_path, output_file_dir, control_methods, length_constraints, api_type, key_config_file,
                param_config_file, max_concurrency=1, length_metric="word"):
    lang_adapt_dict = {
        'equal to': '等于',
        'at least': '至少有',
        'at most': '至多有'
    }
    model = select_model(api_type, key_config_file, param_config_file)
    params = model.model_params
    soften_equal = bool(params.get("soften_equal_to_wording", False)) if isinstance(params, dict) else False
    threshold = next(
        (params.get(k) for k in
         ["max_new_tokens", "max_completion_tokens", "max_output_tokens", "max_tokens", "max_length"]
         if params.get(k) is not None),
        0
    )
    for control_method in control_methods:
        for length_constraint in length_constraints:
            if int(length_constraint) > threshold:
                continue
            meta_data = []
            with open(meta_data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if "task" in data:
                        original_task = data["task"]
                        control_method_lang_adapt = control_method
                        if data['lang'] == 'cn':
                            control_method_lang_adapt = lang_adapt_dict[control_method]
                        modified_task = original_task.replace("{word_count_type}", control_method_lang_adapt)
                        modified_task = modified_task.replace("{word_count}", str(length_constraint))
                        modified_task = _convert_length_wording(
                            modified_task,
                            lang=data.get("lang", ""),
                            length_metric=length_metric,
                        )
                        # For SGLang-local length control, soften 'equal to' phrasing to 'approximately'.
                        if soften_equal and control_method == "equal to" and str(length_metric).strip().lower() == "token":
                            modified_task = _soften_equal_to_wording_for_sglang(
                                modified_task, lang=data.get("lang", "")
                            )
                        meta_data.append({'prompt': modified_task,
                                          'type': data['type'],
                                          'category': data['category'],
                                          'lang': data['lang']}, )
                print(f"Processing control method: {control_method}, length constraint: {length_constraint}")
                try_api_call(
                    model,
                    meta_data,
                    output_file_dir,
                    control_method,
                    length_constraint,
                    max_concurrency,
                    length_metric,
                )
    model.clear()
