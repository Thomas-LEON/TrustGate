"""
TrustGate — Abstract LLM Provider Interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Send a prompt to the LLM and return the raw text response.
        
        Args:
            prompt: The full prompt to send.
            temperature: Sampling temperature.
            
        Returns:
            Raw text response from the LLM.
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier string."""
        ...
