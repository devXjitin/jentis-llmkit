"""
Anthropic Claude LLM Provider

Production-ready wrapper around Anthropic's client.

Author: Jentis Developer
Version: 1.0.0
"""

from typing import Optional, Dict, Any
import os
import time


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================
class AnthropicLLMError(Exception):
    """Base exception class for all Anthropic Claude LLM-related errors."""


class AnthropicLLMImportError(AnthropicLLMError):
    """Raised when the Anthropic client library cannot be imported or initialized."""


class AnthropicLLMAPIError(AnthropicLLMError):
    """Raised when the API request fails after all retry attempts."""


class AnthropicLLMResponseError(AnthropicLLMError):
    """Raised when the API response cannot be interpreted or is malformed."""


def anthropic_llm(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: int = 1024,
    max_retries: int = 3,
    timeout: Optional[float] = 30.0,
    backoff_factor: float = 0.5,
) -> dict:
    """Call Anthropic Claude API and return structured response with metadata.

    Args:
        prompt: The prompt / input text to send to the model. Must be non-empty.
        model: Model identifier (e.g. "claude-3-5-sonnet-20241022").
        api_key: API key to use. If omitted, uses ANTHROPIC_API_KEY env var.
        temperature: Controls randomness (0.0-1.0). Higher = more random.
        top_p: Nucleus sampling threshold (0.0-1.0).
        top_k: Top-k sampling. Limits to k most likely tokens.
        max_tokens: Maximum tokens to generate. Required by Anthropic API.
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
        AnthropicLLMImportError: If the Anthropic client is not installed.
        AnthropicLLMAPIError: If all retry attempts fail.
        AnthropicLLMResponseError: If response is invalid.
    """
    # Input Validation
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if not isinstance(max_retries, int) or max_retries < 1:
        raise ValueError("max_retries must be an integer >= 1")
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be an integer >= 1")
    
    if temperature is not None and not (0.0 <= temperature <= 1.0):
        raise ValueError("temperature must be between 0.0 and 1.0")
    if top_p is not None and not (0.0 <= top_p <= 1.0):
        raise ValueError("top_p must be between 0.0 and 1.0")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")

    # API Key Configuration
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AnthropicLLMImportError(
            "No API key provided and environment variable ANTHROPIC_API_KEY is not set"
        )

    # Import Anthropic client
    try:
        import anthropic
    except ImportError:
        raise AnthropicLLMImportError(
            "anthropic package not installed. Install with: pip install anthropic"
        )

    # Retry Loop
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
            
            # Build message parameters
            create_params: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                create_params["temperature"] = temperature
            if top_p is not None:
                create_params["top_p"] = top_p
            if top_k is not None:
                create_params["top_k"] = top_k
            
            response = client.messages.create(**create_params)
            
            # Extract content
            content = ""
            if hasattr(response, "content") and response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text
            
            if not content:
                raise AnthropicLLMResponseError("No content in response")
            
            # Extract usage
            usage_data = {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }
            if hasattr(response, "usage"):
                usage = response.usage
                input_tok = getattr(usage, "input_tokens", None)
                output_tok = getattr(usage, "output_tokens", None)
                usage_data["input_tokens"] = input_tok
                usage_data["output_tokens"] = output_tok
                if input_tok is not None and output_tok is not None:
                    usage_data["total_tokens"] = input_tok + output_tok
            
            return {
                "content": content,
                "model": model,
                "usage": usage_data,
            }

        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                raise AnthropicLLMAPIError(
                    f"Anthropic LLM request failed after {max_retries} attempts: {exc}"
                ) from exc
            time.sleep(backoff_factor * (2 ** (attempt - 1)))

    raise AnthropicLLMAPIError("Anthropic LLM request failed") from last_exc


def anthropic_llm_stream(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: int = 1024,
    timeout: Optional[float] = 30.0,
):
    """Call Anthropic Claude API and stream the generated text chunks.

    Args:
        prompt: The prompt / input text to send to the model.
        model: Model identifier (e.g. "claude-3-5-sonnet-20241022").
        api_key: API key to use. If omitted, uses ANTHROPIC_API_KEY env var.
        temperature: Controls randomness (0.0-1.0).
        top_p: Nucleus sampling threshold (0.0-1.0).
        top_k: Top-k sampling.
        max_tokens: Maximum tokens to generate.
        timeout: Optional timeout (seconds).

    Yields:
        Text chunks as they are generated.

    Raises:
        ValueError: If required arguments are invalid.
        AnthropicLLMImportError: If client not installed.
        AnthropicLLMAPIError: If streaming request fails.
    """
    # Input Validation
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be an integer >= 1")
    
    if temperature is not None and not (0.0 <= temperature <= 1.0):
        raise ValueError("temperature must be between 0.0 and 1.0")
    if top_p is not None and not (0.0 <= top_p <= 1.0):
        raise ValueError("top_p must be between 0.0 and 1.0")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")

    # API Key
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AnthropicLLMImportError(
            "No API key provided and environment variable ANTHROPIC_API_KEY is not set"
        )

    # Import
    try:
        import anthropic
    except ImportError:
        raise AnthropicLLMImportError(
            "anthropic package not installed. Install with: pip install anthropic"
        )

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        
        # Build stream parameters
        stream_params: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            stream_params["temperature"] = temperature
        if top_p is not None:
            stream_params["top_p"] = top_p
        if top_k is not None:
            stream_params["top_k"] = top_k
        
        with client.messages.stream(**stream_params) as stream:
            for text in stream.text_stream:
                yield text

    except Exception as exc:
        raise AnthropicLLMAPIError(f"Anthropic streaming failed: {exc}") from exc


class AnthropicLLM:
    """
    Class-based wrapper for Anthropic Claude LLM.
    
    Example:
        >>> llm = AnthropicLLM(model="claude-3-5-sonnet-20241022", api_key="your-key")
        >>> response = llm.generate_response("What is Python?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: int = 1024,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        backoff_factor: float = 0.5,
    ):
        """Initialize Anthropic Claude LLM wrapper."""
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response from Anthropic Claude.
        
        Returns only the text content for backward compatibility.
        Use anthropic_llm() function directly for full metadata.
        """
        result = anthropic_llm(
            prompt=prompt,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            timeout=self.timeout,
            backoff_factor=self.backoff_factor,
        )
        return result["content"]
    
    def generate_response_stream(self, prompt: str):
        """Generate a streaming response from Anthropic Claude.
        
        Yields text chunks as they are generated.
        
        Example:
            >>> llm = AnthropicLLM(model="claude-3-5-sonnet-20241022")
            >>> for chunk in llm.generate_response_stream("Write a story"):
            ...     print(chunk, end='', flush=True)
        """
        return anthropic_llm_stream(
            prompt=prompt,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )


__all__ = [
    "anthropic_llm",
    "anthropic_llm_stream",
    "AnthropicLLM",
    "AnthropicLLMError",
    "AnthropicLLMAPIError",
    "AnthropicLLMImportError",
    "AnthropicLLMResponseError",
]
