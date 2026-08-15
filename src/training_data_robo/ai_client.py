from __future__ import annotations

from typing import Optional, Protocol

from openai import OpenAI


class BaseLLMClient(Protocol):
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = 0.2,
        top_p: float = 1.0,
    ) -> str: ...


class DummyLLMClient:
    """
    Simple stand-in for a real LLM.

    It just echoes back the first part of the user prompt so that
    we can test the pipeline without paying for tokens.
    """

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = 0.2,
        top_p: float = 1.0,
    ) -> str:
        trimmed = user_prompt[:max_tokens]
        return f"[DUMMY RESPONSE]\n{trimmed}"


class OpenAILLMClient:
    """
    Thin wrapper around the OpenAI Responses API.
    """

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()

        self.model = model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = 0.2,
        top_p: float = 1.0,
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return str(response.output_text)
