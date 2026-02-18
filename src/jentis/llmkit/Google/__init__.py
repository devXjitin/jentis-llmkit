"""
Jentis LLMKit - Google Gemini Provider
========================================

Google Gemini LLM integration for Jentis Agentic Kit.

Example:
    >>> from jentis.llmkit.Google import GoogleLLM, google_llm
    >>> 
    >>> # Class-based
    >>> llm = GoogleLLM(model="gemini-1.5-pro", api_key="your-key")
    >>> response = llm.generate_response("What is AI?")
    >>> 
    >>> # Function-based
    >>> response = google_llm(
    ...     prompt="What is AI?",
    ...     model="gemini-1.5-pro",
    ...     api_key="your-key"
    ... )

Author: Jentis Developer
Version: 1.0.0
"""

from .base import (
    google_llm,
    google_llm_stream,
    GoogleLLM,
    GoogleLLMError,
    GoogleLLMAPIError,
    GoogleLLMImportError,
    GoogleLLMResponseError,
)

__all__ = [
    "google_llm",
    "google_llm_stream",
    "GoogleLLM",
    "GoogleLLMError",
    "GoogleLLMAPIError",
    "GoogleLLMImportError",
    "GoogleLLMResponseError",
]
