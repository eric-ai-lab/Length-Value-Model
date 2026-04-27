from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class GenerationConfig:
    """Configuration for text generation and API requests."""
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float
    top_k: Optional[int]
    openai_base_url: str
    openai_api_key: Optional[str]
    max_retries: int
    retry_initial_delay: float
    retry_max_delay: float
    request_timeout: float
    max_connections: int
    max_keepalive_connections: int
    keep_columns: List[str] = field(default_factory=list)
    image_dir: Optional[str] = None
