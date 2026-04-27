from openai import OpenAI

from exp.universe_api.BaseModel import BaseModel
from exp.universe_api.OpenAiBaseModel import OpenAiBaseModel


class OpenAiStreamModel(OpenAiBaseModel):
    def __init__(self, api_type, model_params, api_params):
        super().__init__(api_type, model_params, api_params)

    def _call_llm(self, prompt, args):
        # Defensive: never forward benchmark-internal knobs to the OpenAI SDK.
        if isinstance(args, dict):
            args = dict(args)
            args.pop("max_retries", None)
            args.pop("system_prompt", None)
        # Optional system prompt (configured per model in model_param_config.yaml).
        sys_prompt = None
        try:
            if isinstance(self.model_params, dict):
                sys_prompt = self.model_params.get("system_prompt")
        except Exception:
            sys_prompt = None
        # Optional user prompt prefix.
        prefix = ""
        try:
            if isinstance(self.model_params, dict) and self.model_params.get(
                "enable_user_prompt_prefix", False
            ):
                prefix = str(self.model_params.get("user_prompt_prefix") or "")
        except Exception:
            prefix = ""
        messages = []
        if isinstance(sys_prompt, str) and sys_prompt.strip():
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prefix + prompt})
        response = self.client.chat.completions.create(
            messages=messages,
            **args
        )
        answer_content = ""
        reasoning_content = ""
        completion_tokens = None
        reasoning_tokens = None
        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                reasoning_content += delta.reasoning_content
            if hasattr(delta, "content") and delta.content:
                answer_content += delta.content
            try:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    completion_tokens = getattr(usage, "completion_tokens", completion_tokens)
                    details = getattr(usage, "completion_tokens_details", None)
                    if details is not None:
                        reasoning_tokens = getattr(details, "reasoning_tokens", reasoning_tokens)
            except Exception:
                pass
        completion_tokens = self._extract_response_completion_tokens(completion_tokens, reasoning_tokens)
        return {"response": answer_content, "thinking": reasoning_content, "completion_tokens": completion_tokens}

    def clear(self):
        self.client = None
