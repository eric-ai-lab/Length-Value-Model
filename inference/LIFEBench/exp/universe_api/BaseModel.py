import asyncio
import json
import os
import time
from abc import ABC, abstractmethod

from exp.cache import load_cache, save_cache, count_words


class BaseModel(ABC):

    def __init__(self, api_type, model_params):
        self.meta_data = None
        self.length_constraint = None
        self.control_method = None
        self.api_type = api_type
        self.model_params = model_params
        self.output_file = ""
        self.cache_file = ""
        self.length_metric = "word"
        self._printed_tokenizer_name = False

    def prepare_dir(self, output_file_dir, control_method, length_constraint):
        self.control_method = control_method
        self.length_constraint = length_constraint
        output_dir = os.path.join(output_file_dir, f"{self.control_method}")
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, f"{self.length_constraint}.jsonl")
        cache_dir = os.path.join(output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, f"{self.length_constraint}_cache.json")

    def set_length_metric(self, length_metric):
        metric = str(length_metric or "word").strip().lower()
        if metric not in {"word", "token"}:
            raise ValueError(f"Unsupported length metric: {length_metric}. Use 'word' or 'token'.")
        self.length_metric = metric

    def _build_call_args(self) -> dict:
        """Build per-call args for the current (control_method, length_constraint).

        This is the hook where we inject LVM-guided decoding params for SGLang.

        If `model_params["enable_lvm_length_control"]` is truthy, we will attach:
          extra_body.custom_params = {
            value_constraint: eq/ge/le  (mapped from control_method),
            target_length: int(length_constraint),
            target_length_mode: "total" (default) or overridden via model_params["lvm_target_length_mode"],
            gamma: optional (model_params["lvm_gamma"]),
          }

        Note: We put these under `extra_body` so the OpenAI Python SDK can forward them
        as extra JSON fields without rejecting unknown top-level parameters.
        """
        args = dict(self.model_params) if isinstance(self.model_params, dict) else {}

        # Internal-only knobs; remove them from the API args.
        # Retry is handled in BaseModel.get_cache_data(); it must not be forwarded to the API.
        args.pop("max_retries", None)
        # Used to construct `messages` in model wrappers; must not be forwarded as a kwarg.
        args.pop("system_prompt", None)
        # Benchmark prompt rewriting toggle; must not be forwarded as a kwarg.
        args.pop("soften_equal_to_wording", None)
        # Evaluation-only toggle; must not be forwarded as a kwarg.
        args.pop("length_metric", None)
        # Optional official count-only model name; used internally by Anthropic token counting.
        args.pop("count_model", None)
        # Optional tokenizer source used only for local token counting.
        args.pop("tokenizer_model", None)
        # Optional user prompt prefix; must not be forwarded as a kwarg.
        args.pop("user_prompt_prefix", None)
        args.pop("enable_user_prompt_prefix", None)
        enable = bool(args.pop("enable_lvm_length_control", False))
        length_mode = args.pop("lvm_target_length_mode", "total")
        gamma = args.pop("lvm_gamma", None)
        extra_custom = args.pop("lvm_custom_params", None)
        lvm_value_constraint = args.pop("lvm_value_constraint", None)

        if not enable:
            return args

        control_to_constraint = {
            "equal to": "eq",
            "at most": "le",
            "at least": "ge",
        }
        value_constraint = lvm_value_constraint or control_to_constraint.get(self.control_method)
        if value_constraint is None:
            return args

        # When measuring in tokens, add 1 to account for the EOS token which is
        # counted by the model but not included in the decoded response text.
        eos_offset = 1 if str(self.length_metric).strip().lower() == "token" else 0
        custom_params = {
            "value_constraint": value_constraint,
            "target_length": int(self.length_constraint) + eos_offset,
            "target_length_mode": str(length_mode),
        }
        if gamma is not None:
            try:
                custom_params["gamma"] = float(gamma)
            except (TypeError, ValueError):
                pass
        if isinstance(extra_custom, dict):
            # Allow users to pass through arbitrary extra keys.
            custom_params.update(extra_custom)

        extra_body = args.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        else:
            extra_body = dict(extra_body)
        extra_body["custom_params"] = custom_params
        args["extra_body"] = extra_body
        return args

    def _get_max_retries(self) -> int:
        max_retries = 3
        if isinstance(self.model_params, dict):
            try:
                max_retries = int(self.model_params.get("max_retries", max_retries))
            except Exception:
                max_retries = 3
        return max(1, max_retries)

    def _extract_response_completion_tokens(self, completion_tokens, reasoning_tokens=None):
        if completion_tokens is None:
            return None
        try:
            total = int(completion_tokens)
        except Exception:
            return None

        if reasoning_tokens is None:
            return total

        try:
            reasoning = int(reasoning_tokens)
        except Exception:
            return total

        return max(0, total - reasoning)

    def _compute_gpt_token_count(self, response):
        model_name = None
        if isinstance(self.model_params, dict):
            model_name = self.model_params.get("model")

        try:
            import tiktoken
        except Exception as exc:
            raise RuntimeError(
                "Token mode for GPT models requires the `tiktoken` package to be installed."
            ) from exc

        enc = None
        if isinstance(model_name, str) and model_name.strip():
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                enc = None
        tokenizer_name = getattr(enc, "name", None) if enc is not None else None
        if enc is None:
            enc = tiktoken.get_encoding("o200k_base")
            tokenizer_name = "o200k_base"

        if not self._printed_tokenizer_name:
            resolved_model = model_name if isinstance(model_name, str) and model_name.strip() else self.api_type
            print(f"[tokenizer] api_type={self.api_type} model={resolved_model} tokenizer={tokenizer_name}")
            self._printed_tokenizer_name = True

        return len(enc.encode(str(response)))

    def _compute_transformers_token_count(self, response):
        model_name = None
        if isinstance(self.model_params, dict):
            model_name = self.model_params.get("tokenizer_model") or self.model_params.get("model")
        if not isinstance(model_name, str) or not model_name.strip():
            raise RuntimeError(
                f"Token mode requires a valid model name/path in model_params for api_type='{self.api_type}'."
            )

        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Token mode for this model requires the `transformers` package to be installed."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False,
            )
        tokenizer_name = getattr(tokenizer, "name_or_path", None) or model_name
        if not self._printed_tokenizer_name:
            print(f"[tokenizer] api_type={self.api_type} model={model_name} tokenizer={tokenizer_name}")
            self._printed_tokenizer_name = True

        return len(tokenizer.encode(str(response), add_special_tokens=False))

    def _compute_length_metrics(self, response, completion_tokens):
        word_count = count_words(response)
        token_count = None

        if self.length_metric == "token":
            if str(self.api_type).startswith("gpt-"):
                token_count = self._compute_gpt_token_count(response)
            elif str(self.api_type).startswith("Qwen3-"):
                token_count = self._compute_transformers_token_count(response)
            elif str(self.api_type).startswith("claude-"):
                if not hasattr(self, "count_visible_text_tokens"):
                    raise RuntimeError(
                        f"Claude token mode requires count_visible_text_tokens(), but api_type='{self.api_type}' does not provide it."
                    )
                token_count = self.count_visible_text_tokens(str(response))
            elif str(self.api_type).startswith("gemini-"):
                token_count = completion_tokens
                if token_count is None:
                    raise RuntimeError(
                        f"Gemini token mode requires completion_tokens from usage metadata, but api_type='{self.api_type}' did not return it."
                    )
            elif completion_tokens is not None:
                token_count = completion_tokens
            elif hasattr(self, "count_visible_text_tokens"):
                token_count = self.count_visible_text_tokens(str(response))
            else:
                raise RuntimeError(
                    f"Token mode is currently only supported for GPT, Claude, Gemini, and models that provide count_visible_text_tokens(). "
                    f"Got api_type='{self.api_type}'."
                )

        return word_count, token_count

    async def _process_single_index(self, idx, cached_responses, cache_lock):
        max_retries = self._get_max_retries()
        prompt = self.meta_data[idx]["prompt"]

        for attempt in range(max_retries):
            try:
                begin_time = time.time()
                response = await asyncio.to_thread(self._call_llm, prompt, self._build_call_args())
                thinking = ""
                completion_tokens = None
                if isinstance(response, dict):
                    thinking = response["thinking"]
                    completion_tokens = response.get("completion_tokens")
                    response = response["response"]

                end_time = time.time()
                word_count, token_count = self._compute_length_metrics(response, completion_tokens)
                record = {
                    "response": response,
                    "thinking": thinking,
                    "word_count": word_count,
                    "token_count": token_count,
                    "time": end_time - begin_time,
                    **self.meta_data[idx]
                }

                async with cache_lock:
                    cached_responses[idx] = record
                    await asyncio.to_thread(save_cache, self.cache_file, cached_responses)

                print(f"Index {idx} processed, cache updated.")
                return True
            except Exception as e:
                error_msg = f"Error occurred while processing index {idx} (attempt {attempt + 1}/{max_retries}): {e}"
                e_str = str(e)
                if (
                    "context length" in e_str.lower()
                    or "longer than the model's context length" in e_str.lower()
                    or "is longer than the model" in e_str.lower()
                ):
                    print(error_msg + ", not retrying (context length overflow)")
                    break
                if attempt < max_retries - 1:
                    print(error_msg + ", retrying in 3 seconds...")
                    await asyncio.sleep(3)
                else:
                    print(error_msg + ", giving up retry")

        print(f"Index {idx} failed after {max_retries} attempts, skipping.")
        return False

    async def _get_cache_data_async(self, meta_data, max_concurrency):
        self.meta_data = meta_data
        cached_responses = load_cache(self.cache_file)
        existing_indices = set(cached_responses.keys())
        total_indices = set(range(len(self.meta_data)))
        remaining_indices = sorted(total_indices - existing_indices)
        if not remaining_indices:
            print("All prompts have been processed.")
            self._store_cache(cached_responses)
            return

        concurrency = max(1, int(max_concurrency))
        semaphore = asyncio.Semaphore(concurrency)
        cache_lock = asyncio.Lock()

        async def bounded_process(idx):
            async with semaphore:
                return await self._process_single_index(idx, cached_responses, cache_lock)

        await asyncio.gather(*(bounded_process(idx) for idx in remaining_indices))
        self._store_cache(cached_responses)

    def get_cache_data(self, meta_data, max_concurrency=1):
        asyncio.run(self._get_cache_data_async(meta_data, max_concurrency))

    def _store_cache(self, cached_responses):
        with open(self.output_file, "w", encoding="utf-8") as f:  # Overwrite mode
            for idx in sorted(cached_responses.keys()):
                data = cached_responses[idx]
                # Remove prompt from cache structure directly
                data.pop("prompt", None)
                f.write(json.dumps({
                    "index": idx,
                    "control_method": self.control_method,
                    "length_constraint": self.length_constraint,
                    "api_type": self.api_type,
                    "length_metric": self.length_metric,
                    **data,
                }, ensure_ascii=False) + "\n")

    @abstractmethod
    def _call_llm(self, prompt, args):
        pass

    @abstractmethod
    def clear(self):
        pass
