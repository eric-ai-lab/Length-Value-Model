import requests
from openai import OpenAI

from exp.universe_api.BaseModel import BaseModel


class HTTPBaseModel(BaseModel):
    def __init__(self, api_type, model_params, api_params):
        super().__init__(api_type, model_params)
        self.api_key = api_params["api_key"]
        self.base_url = api_params["base_url"]

    def _call_llm(self, prompt, args):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        sys_prompt = None
        try:
            if isinstance(self.model_params, dict):
                sys_prompt = self.model_params.get("system_prompt")
        except Exception:
            sys_prompt = None
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
        if isinstance(args, dict):
            args = dict(args)
            args.pop("system_prompt", None)
            args.pop("max_retries", None)
            args.pop("user_prompt_prefix", None)
            args.pop("enable_user_prompt_prefix", None)
        payload = {"messages": messages, **args}
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=2000)
        response.raise_for_status()
        data = response.json()
        msg = data["choices"][0]["message"]
        completion_tokens = None
        reasoning_tokens = None
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens")
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict):
                reasoning_tokens = details.get("reasoning_tokens")
            completion_tokens = self._extract_response_completion_tokens(completion_tokens, reasoning_tokens)
        if "reasoning_content" in msg:
            return {"response": msg["content"], "thinking": msg["reasoning_content"], "completion_tokens": completion_tokens}
        else:
            if completion_tokens is not None:
                return {"response": msg["content"], "thinking": "", "completion_tokens": completion_tokens}
            return msg["content"]

    def clear(self):
        pass
