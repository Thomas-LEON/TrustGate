"""
TrustGate — Google Gemini LLM Provider (Legacy).

NOTE: This module is kept for direct SDK usage and backward compatibility.
The primary LLM routing path is: routes.py → LLMGateway (core/gateway.py) → LiteLLM.
GeminiProvider is NOT called in the main API flow.
"""
from __future__ import annotations

import os

from google import genai
from google.genai import types

from providers.base import BaseLLMProvider


# Model fallback cascade — kept in sync with core/config.py defaults
DEFAULT_MODELS = [
    "gemini/gemini-3.8-flash",
    "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-3.1-flash-lite",
]


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider with model cascade fallback."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in .env or pass it explicitly."
            )
        self._client = genai.Client(api_key=self._api_key)
        self._model = model  # None = use cascade
        self._active_model = model or DEFAULT_MODELS[0]

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate with automatic model fallback."""
        models_to_try = [self._model] if self._model else DEFAULT_MODELS

        last_error = None
        for model_id in models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                    ),
                )
                self._active_model = model_id
                return response.text
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(
            f"All models failed. Last error: {last_error}"
        )

    def get_model_name(self) -> str:
        return self._active_model
