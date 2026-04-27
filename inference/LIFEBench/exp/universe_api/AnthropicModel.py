import json
import re
import urllib.error
import urllib.request

import anthropic

from exp.universe_api.BaseModel import BaseModel


def _strip_thinking_tags(text: str) -> str:
    """Remove <thinking>...</thinking> blocks. If a tag is unclosed, strip everything from it onward."""
    text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL).lstrip()
    # Remove unclosed <thinking> and everything after it
    text = re.sub(r"<thinking>.*$", "", text, flags=re.DOTALL).rstrip()
    return text


def _extract_thinking_from_text(text: str):
    """Extract <thinking>...</thinking> content from text, return (thinking, response).
    If a <thinking> tag is opened but never closed, everything before the unclosed tag
    (inclusive) is treated as thinking and the rest as response.
    """
    # First handle complete <thinking>...</thinking> blocks
    thinking_parts = re.findall(r"<thinking>(.*?)</thinking>", text, flags=re.DOTALL)
    response_text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL).lstrip()

    # Handle unclosed <thinking> tag: treat everything up to and including it as thinking
    unclosed = re.search(r"^(.*)<thinking>(.*)", response_text, flags=re.DOTALL)
    if unclosed:
        thinking_parts.append(unclosed.group(1) + unclosed.group(2))
        response_text = ""

    thinking = "\n".join(thinking_parts)
    return thinking, response_text


class AnthropicModel(BaseModel):
    def __init__(self, api_type, model_params, api_params):
        super().__init__(api_type, model_params)
        generation_api_key = api_params.get("api_key")
        count_api_key = api_params.get("count_api_key", generation_api_key)
        if not generation_api_key:
            raise ValueError(f"{api_type} requires api_key for generation.")
        if not count_api_key:
            raise ValueError(
                f"{api_type} requires count_api_key for official Anthropic count_tokens."
            )

        count_params = {"api_key": count_api_key}

        base_url = api_params.get("base_url")
        self.generation_api_key = generation_api_key
        self.generation_base_url = None
        if base_url:
            normalized_base_url = str(base_url).rstrip("/")
            if normalized_base_url.endswith("/v1"):
                normalized_base_url = normalized_base_url[: -len("/v1")]
            self.generation_base_url = normalized_base_url
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=generation_api_key)
        self.count_client = anthropic.Anthropic(**count_params)
        self.model = model_params["model"]
        self.count_model = model_params.get("count_model", self.model)

    def _call_custom_generation_endpoint(self, prompt, args):
        payload_args = dict(args) if isinstance(args, dict) else {}
        # Bedrock-backed Claude endpoints commonly reject requests that set both temperature and top_p.
        if payload_args.get("temperature") is not None and payload_args.get("top_p") is not None:
            payload_args.pop("top_p", None)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **payload_args,
        }
        req = urllib.request.Request(
            f"{self.generation_base_url}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.generation_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2000) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

        content_blocks = body.get("content") or []
        thinking = ""
        response_text = ""
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "thinking":
                thinking += block.get("thinking", "")
            elif block_type == "text":
                response_text += block.get("text", "")

        # Some proxy APIs embed thinking in the text block; extract it if present.
        if not thinking and "<thinking>" in response_text:
            thinking, response_text = _extract_thinking_from_text(response_text)
        elif thinking:
            response_text = _strip_thinking_tags(response_text)
        if thinking:
            return {"thinking": thinking, "response": response_text}
        return response_text

    def _call_llm(self, prompt, args):
        args.pop("model", None)
        if self.generation_base_url:
            return self._call_custom_generation_endpoint(prompt, args)

        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **args,
            timeout=2000
        )
        if len(response.content) > 1:
            return {"thinking": response.content[0].thinking, "response": response.content[1].text}
        else:
            text = response.content[0].text
            if "<thinking>" in text:
                thinking, response_text = _extract_thinking_from_text(text)
                return {"thinking": thinking, "response": response_text}
            return text

    def count_visible_text_tokens(self, text: str) -> int:
        response = self.count_client.messages.count_tokens(
            model=self.count_model,
            messages=[{"role": "assistant", "content": text}],
        )
        input_tokens = getattr(response, "input_tokens", None)
        if input_tokens is None:
            raise RuntimeError("Anthropic count_tokens did not return input_tokens.")
        visible_tokens = int(input_tokens) - 16
        if visible_tokens < 0:
            raise RuntimeError(
                f"Anthropic count_tokens returned input_tokens={input_tokens}, which is smaller than the expected assistant-message baseline 16."
            )
        return visible_tokens

    def clear(self):
        self.client = None
        self.count_client = None
