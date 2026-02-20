"""
xAI Grok LLM Provider

Production-ready wrapper around xAI's Grok client (OpenAI-compatible API).

Author: Jentis Developer
Version: 1.0.0
"""

from typing import Optional, Dict, Any
import os
import time


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================
class GrokLLMError(Exception):
    """Base exception class for all Grok LLM-related errors."""


class GrokLLMImportError(GrokLLMError):
    """Raised when the OpenAI client library cannot be imported or initialized."""


class GrokLLMAPIError(GrokLLMError):
    """Raised when the API request fails after all retry attempts."""


class GrokLLMResponseError(GrokLLMError):
    """Raised when the API response cannot be interpreted or is malformed."""


def grok_llm(
    prompt: str,
    model: str = "grok-beta",
    api_key: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    max_retries: int = 3,
    timeout: Optional[float] = 30.0,
    backoff_factor: float = 0.5,
) -> dict:
    """Call xAI Grok API and return structured response with metadata.

    Args:
        prompt: The prompt / input text to send to the model. Must be non-empty.
        model: Model identifier (default: "grok-beta").
        api_key: API key to use. If omitted, uses XAI_API_KEY env var.
        temperature: Controls randomness (0.0-2.0). Higher = more random.
        top_p: Nucleus sampling threshold (0.0-1.0).
        max_tokens: Maximum tokens to generate.
        frequency_penalty: Penalty for token frequency (-2.0 to 2.0).
        presence_penalty: Penalty for token presence (-2.0 to 2.0).
        max_retries: Number of attempts to make on transient failures.
        timeout: Optional timeout (seconds).
        backoff_factor: Base factor for exponential backoff between retries.

    Returns:
        Dictionary containing:
            - content: The generated text from the model
            - model: The model identifier used
            - usage: Dictionary with token usage information
                - input_tokens: Number of tokens in the input prompt
                - output_tokens: Number of tokens in the output completion
                - total_tokens: Total number of tokens used

    Raises:
        ValueError: If required arguments are missing or invalid.
        GrokLLMImportError: If the OpenAI client is not installed.
        GrokLLMAPIError: If all retry attempts fail.
        GrokLLMResponseError: If response is invalid.
    """
    # Input Validation
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if not isinstance(max_retries, int) or max_retries < 1:
        raise ValueError("max_retries must be an integer >= 1")
    
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")
    if top_p is not None and not (0.0 <= top_p <= 1.0):
        raise ValueError("top_p must be between 0.0 and 1.0")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if frequency_penalty is not None and not (-2.0 <= frequency_penalty <= 2.0):
        raise ValueError("frequency_penalty must be between -2.0 and 2.0")
    if presence_penalty is not None and not (-2.0 <= presence_penalty <= 2.0):
        raise ValueError("presence_penalty must be between -2.0 and 2.0")

    # API Key Configuration
    api_key = api_key or os.environ.get("XAI_API_KEY")
    if not api_key:
        raise GrokLLMImportError(
            "No API key provided and environment variable XAI_API_KEY is not set"
        )

    # Import OpenAI client (Grok uses OpenAI-compatible API)
    try:
        from openai import OpenAI
    except ImportError:
        raise GrokLLMImportError(
            "openai package not installed. Install with: pip install openai"
        )

    # Build kwargs
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty

    # Retry Loop
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1",
                timeout=timeout
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            # Extract content
            content = ""
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if choice.message and choice.message.content:
                    content = choice.message.content
            
            if not content:
                raise GrokLLMResponseError("No content in response")
            
            # Extract usage
            usage_data = {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }
            if response.usage:
                usage_data["input_tokens"] = response.usage.prompt_tokens
                usage_data["output_tokens"] = response.usage.completion_tokens
                usage_data["total_tokens"] = response.usage.total_tokens
            
            return {
                "content": content,
                "model": model,
                "usage": usage_data,
            }

        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                raise GrokLLMAPIError(
                    f"Grok LLM request failed after {max_retries} attempts: {exc}"
                ) from exc
            time.sleep(backoff_factor * (2 ** (attempt - 1)))

    raise GrokLLMAPIError("Grok LLM request failed") from last_exc


def grok_llm_stream(
    prompt: str,
    model: str = "grok-beta",
    api_key: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    timeout: Optional[float] = 30.0,
):
    """Call xAI Grok API and stream the generated text chunks.

    Args:
        prompt: The prompt / input text to send to the model.
        model: Model identifier (default: "grok-beta").
        api_key: API key to use. If omitted, uses XAI_API_KEY env var.
        temperature: Controls randomness (0.0-2.0).
        top_p: Nucleus sampling threshold (0.0-1.0).
        max_tokens: Maximum tokens to generate.
        frequency_penalty: Penalty for token frequency (-2.0 to 2.0).
        presence_penalty: Penalty for token presence (-2.0 to 2.0).
        timeout: Optional timeout (seconds).

    Yields:
        Text chunks as they are generated.

    Raises:
        ValueError: If required arguments are invalid.
        GrokLLMImportError: If client not installed.
        GrokLLMAPIError: If streaming request fails.
    """
    # Input Validation
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")
    if top_p is not None and not (0.0 <= top_p <= 1.0):
        raise ValueError("top_p must be between 0.0 and 1.0")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if frequency_penalty is not None and not (-2.0 <= frequency_penalty <= 2.0):
        raise ValueError("frequency_penalty must be between -2.0 and 2.0")
    if presence_penalty is not None and not (-2.0 <= presence_penalty <= 2.0):
        raise ValueError("presence_penalty must be between -2.0 and 2.0")

    # API Key
    api_key = api_key or os.environ.get("XAI_API_KEY")
    if not api_key:
        raise GrokLLMImportError(
            "No API key provided and environment variable XAI_API_KEY is not set"
        )

    # Import
    try:
        from openai import OpenAI
    except ImportError:
        raise GrokLLMImportError(
            "openai package not installed. Install with: pip install openai"
        )

    # Build kwargs
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=timeout
        )
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

    except Exception as exc:
        raise GrokLLMAPIError(f"Grok streaming failed: {exc}") from exc


class GrokLLM:
    """
    Class-based wrapper for xAI Grok LLM.
    
    Example:
        >>> llm = GrokLLM(model="grok-beta", api_key="your-key")
        >>> response = llm.generate_response("What is Python?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: str = "grok-beta",
        api_key: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        backoff_factor: float = 0.5,
    ):
        """Initialize xAI Grok LLM wrapper."""
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response from xAI Grok.
        
        Returns only the text content for backward compatibility.
        Use grok_llm() function directly for full metadata.
        """
        result = grok_llm(
            prompt=prompt,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            max_retries=self.max_retries,
            timeout=self.timeout,
            backoff_factor=self.backoff_factor,
        )
        return result["content"]
    
    def generate_response_stream(self, prompt: str):
        """Generate a streaming response from xAI Grok.
        
        Yields text chunks as they are generated.
        
        Example:
            >>> llm = GrokLLM(model="grok-beta")
            >>> for chunk in llm.generate_response_stream("Write a story"):
            ...     print(chunk, end='', flush=True)
        """
        return grok_llm_stream(
            prompt=prompt,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            timeout=self.timeout,
        )


__all__ = [
    "grok_llm",
    "grok_llm_stream",
    "GrokLLM",
    "GrokLLMError",
    "GrokLLMAPIError",
    "GrokLLMImportError",
    "GrokLLMResponseError",
]
