from openai import OpenAI

from exp.universe_api.BaseModel import BaseModel


class OpenAiBaseModel(BaseModel):
    def __init__(self, api_type, model_params, api_params):
        super().__init__(api_type, model_params)
        normalized_api_params = dict(api_params) if isinstance(api_params, dict) else {}
        base_url = normalized_api_params.get("base_url")
        if isinstance(base_url, str) and base_url.strip():
            normalized_base_url = base_url.rstrip("/")
            if not normalized_base_url.endswith("/v1"):
                normalized_base_url = f"{normalized_base_url}/v1"
            normalized_api_params["base_url"] = normalized_base_url
        self.client = OpenAI(**normalized_api_params)

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
        content = response.choices[0].message.content
        completion_tokens = None
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                completion_tokens = getattr(usage, "completion_tokens", None)
                details = getattr(usage, "completion_tokens_details", None)
                reasoning_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None
                completion_tokens = self._extract_response_completion_tokens(completion_tokens, reasoning_tokens)
        except Exception:
            completion_tokens = None
        if completion_tokens is not None:
            return {"response": content, "thinking": "", "completion_tokens": completion_tokens}
        return content

    def clear(self):
        self.client = None
