from google import genai
from google.genai import types
from openai import OpenAI

from exp.universe_api.BaseModel import BaseModel


class GenAiModel(BaseModel):
    def __init__(self, api_type, model_params, api_params):
        super().__init__(api_type, model_params)
        self.model = model_params["model"]
        self.base_url = api_params.get("base_url")
        self._is_openai_compatible = bool(self.base_url)

        if self._is_openai_compatible:
            openai_params = {"api_key": api_params["api_key"]}
            normalized_base_url = str(self.base_url).rstrip("/")
            if not normalized_base_url.endswith("/v1"):
                normalized_base_url = f"{normalized_base_url}/v1"
            openai_params["base_url"] = normalized_base_url
            self.client = OpenAI(**openai_params)
        else:
            official_params = dict(api_params)
            official_params.pop("base_url", None)
            self.client = genai.Client(**official_params)

    @staticmethod
    def _get_usage_value(usage_metadata, snake_name, camel_name):
        if usage_metadata is None:
            return None
        value = getattr(usage_metadata, snake_name, None)
        if value is None:
            value = getattr(usage_metadata, camel_name, None)
        return value

    @staticmethod
    def _normalize_openai_compatible_args(args):
        normalized_args = dict(args) if isinstance(args, dict) else {}
        max_output_tokens = normalized_args.pop("max_output_tokens", None)
        if max_output_tokens is not None and normalized_args.get("max_tokens") is None:
            normalized_args["max_tokens"] = max_output_tokens

        # Gemini-native config keys are not accepted by OpenAI-compatible chat completions APIs.
        normalized_args.pop("response_mime_type", None)
        normalized_args.pop("response_schema", None)
        normalized_args.pop("candidate_count", None)
        normalized_args.pop("seed", None)
        return normalized_args

    def _call_llm(self, prompt, args):
        args.pop("model", None)
        if self._is_openai_compatible:
            args = self._normalize_openai_compatible_args(args)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
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
                    completion_tokens = self._extract_response_completion_tokens(
                        completion_tokens, reasoning_tokens
                    )
            except Exception:
                completion_tokens = None
        else:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    **args
                )
            )
            content = response.text
            usage_metadata = getattr(response, "usage_metadata", None)
            candidate_tokens = self._get_usage_value(
                usage_metadata, "candidates_token_count", "candidatesTokenCount"
            )
            thoughts_tokens = self._get_usage_value(
                usage_metadata, "thoughts_token_count", "thoughtsTokenCount"
            )

            completion_tokens = None
            if candidate_tokens is not None or thoughts_tokens is not None:
                total_output_tokens = 0
                if candidate_tokens is not None:
                    total_output_tokens += int(candidate_tokens)
                if thoughts_tokens is not None:
                    total_output_tokens += int(thoughts_tokens)
                completion_tokens = self._extract_response_completion_tokens(
                    total_output_tokens, thoughts_tokens
                )

        if completion_tokens is not None:
            return {"response": content, "thinking": "", "completion_tokens": completion_tokens}
        return content

    def clear(self):
        self.client = None
