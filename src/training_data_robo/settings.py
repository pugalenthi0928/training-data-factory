from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Settings:
    """Global configuration for the Training Data Robo factory."""

    # Concurrency / performance knobs
    # Can override with: export TDR_MAX_CONCURRENT="20"
    max_concurrent_requests: int = int(os.getenv("TDR_MAX_CONCURRENT", "10"))

    # Default LLM model used by the factory
    # Can override with: export TDR_DEFAULT_MODEL="gpt-4.1"
    default_model: str = os.getenv("TDR_DEFAULT_MODEL", "gpt-4.1-mini")

    # Default chunking behaviour (used as sensible global defaults)
    # Can override with:
    #   export TDR_MAX_CHARS="1000"
    #   export TDR_OVERLAP="150"
    default_max_chars: int = int(os.getenv("TDR_MAX_CHARS", "800"))
    default_overlap: int = int(os.getenv("TDR_OVERLAP", "100"))

    # For future: external web-scraping service
    decoder_api_key: Optional[str] = os.getenv("DECODER_API_KEY")

    # OpenAI API key (if set, we use a real LLM client)
    # Set with: export OPENAI_API_KEY="sk-..."
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    @classmethod
    def from_env(cls, **overrides: object) -> "Settings":
        """
        Build Settings from environment variables, with optional overrides.

        Example:
            settings = Settings.from_env(default_model="gpt-4.1-mini")
        """
        settings = cls()
        for key, value in overrides.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        return settings
