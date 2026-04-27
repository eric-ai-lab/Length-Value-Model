import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import openai

from .config import GenerationConfig
from .prompt_builder import PromptBuilder
from .utils import calculate_backoff_delay, safe_get, RETRYABLE_STATUS_CODES

logger = logging.getLogger(__name__)

class DataGenerator:
    """Handles async generation of completions using OpenAI-compatible API."""

    def __init__(
        self,
        config: GenerationConfig,
        max_concurrency: int,
        prompt_builder: PromptBuilder,
    ):
        self.config = config
        self.prompt_builder = prompt_builder
        self.max_concurrency = max(1, max_concurrency)
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

        timeout = httpx.Timeout(config.request_timeout) if config.request_timeout > 0 else None
        limits = httpx.Limits(
            max_connections=config.max_connections,
            max_keepalive_connections=config.max_keepalive_connections,
        )
        http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
        self.client = openai.AsyncClient(
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            http_client=http_client,
        )

    async def close(self):
        """Closes the underlying HTTP client."""
        await self.client.close()

    def _should_retry(self, exc: Exception) -> bool:
        """Determines if the exception is retryable."""
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, httpx.ReadTimeout)):
            return True

        status_code = getattr(exc, "status_code", None)
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
        
        if status_code in RETRYABLE_STATUS_CODES:
            return True
        return 500 <= (status_code or 0) < 600

    async def process_sample(
        self,
        sample: Dict[str, Any],
        split_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Generates a completion for a single sample with retry logic."""
        # Semaphore is acquired per-attempt so the slot is released during backoff sleep,
        # allowing other coroutines to proceed instead of blocking on a sleeping retry.
        for attempt in range(self.config.max_retries + 1):
            async with self.semaphore:
                try:
                    messages = self.prompt_builder.build_messages(sample)
                    request_payload = {
                        "model": self.config.model_name,
                        "messages": messages,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                        "top_p": self.config.top_p,
                    }
                    if self.config.top_k is not None:
                        request_payload["top_k"] = self.config.top_k

                    response = await self.client.chat.completions.create(**request_payload)

                    answer = response.choices[0].message.content
                    completion_tokens = getattr(response.usage, "completion_tokens", None) if response.usage else None

                    # Extract kept fields and lenvm_idx directly from sample
                    kept_fields = {col: sample[col] for col in self.config.keep_columns if col in sample}

                    lenvm_idx = sample.get("lenvm_idx")
                    assert lenvm_idx is not None, f"lenvm_idx is not found in sample: {sample}"

                    conversations: List[Dict[str, str]] = []
                    if self.prompt_builder.is_multimodal():
                        # For multimodal: extract text-only question for conversations
                        for msg in messages:
                            role = msg.get("role")
                            if role == "user":
                                # Content is a list of parts; extract text
                                content_parts = msg.get("content", [])
                                text_parts = [p["text"] for p in content_parts if p.get("type") == "text"]
                                question_text = " ".join(text_parts)
                                # Prepend <image> placeholder for LlamaFactory compatibility
                                conversations.append({"from": "human", "value": f"<image>{question_text}"})
                            elif role == "system":
                                conversations.append({"from": "system", "value": msg.get("content", "")})
                            elif role == "assistant":
                                conversations.append({"from": "gpt", "value": msg.get("content", "")})
                    else:
                        for msg in messages:
                            role = msg.get("role")
                            if role == "system":
                                conversations.append({"from": "system", "value": msg.get("content", "")})
                            elif role == "user":
                                conversations.append({"from": "human", "value": msg.get("content", "")})
                            elif role == "assistant":
                                conversations.append({"from": "gpt", "value": msg.get("content", "")})

                    conversations.append({"from": "gpt", "value": answer})

                    result = {
                        "conversations": conversations,
                        "meta_info": {
                            "answer_token_length": completion_tokens,
                            "lenvm_idx": lenvm_idx,
                            "split": split_name,
                            "dataset_name": self.prompt_builder.dataset_name,
                            "model_name": self.config.model_name,
                            "temperature": self.config.temperature,
                            "top_p": self.config.top_p,
                            "top_k": self.config.top_k,
                            "max_tokens": self.config.max_tokens,
                        },
                        "source_fields": kept_fields,
                    }

                    # Save image and add images field for multimodal datasets
                    if self.prompt_builder.is_multimodal() and self.config.image_dir:
                        image_b64 = sample.get("image")
                        if image_b64:
                            image_dir = Path(self.config.image_dir)
                            image_dir.mkdir(parents=True, exist_ok=True)
                            image_filename = f"{lenvm_idx:05d}.png"
                            image_path = image_dir / image_filename
                            image_path.write_bytes(base64.b64decode(image_b64))
                            result["images"] = [str(image_path)]

                    return result
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if attempt >= self.config.max_retries or not self._should_retry(exc):
                        logger.error(f"Failed to process sample after {attempt+1} attempts: {exc}")
                        return None

                    delay = calculate_backoff_delay(attempt + 1, self.config.retry_initial_delay, self.config.retry_max_delay)
                    logger.warning(f"Retry {attempt+1}/{self.config.max_retries} in {delay:.2f}s due to {type(exc).__name__}")
                    await asyncio.sleep(delay)
        return None
