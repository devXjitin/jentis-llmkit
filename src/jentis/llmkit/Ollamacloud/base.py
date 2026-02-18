"""
Ollama Cloud LLM Provider

Production-ready wrapper around Ollama Cloud client.

Author: Jentis Developer
Version: 1.0.0
"""

from typing import Optional
import os
import time


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================
class OllamaCloudLLMError(Exception):
    """Base exception class for all Ollama Cloud LLM-related errors."""


class OllamaCloudLLMImportError(OllamaCloudLLMError):
    """Raised when the Ollama client library cannot be imported or initialized."""


class OllamaCloudLLMAPIError(OllamaCloudLLMError):
    """Raised when the API request fails after all retry attempts."""


class OllamaCloudLLMResponseError(OllamaCloudLLMError):
    """Raised when the API response cannot be interpreted or is malformed."""


def ollama_cloud_llm(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    host: str = "https://ollama.com",
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    timeout: Optional[float] = 30.0,
    backoff_factor: float = 0.5,
) -> dict:
    """Call Ollama Cloud API and return structured response with metadata.

    Args:
        prompt: The prompt / input text to send to the model. Must be non-empty.
        model: Model identifier (e.g. "llama2", "mistral", "codellama").
        api_key: API key to use. If omitted, uses OLLAMA_API_KEY env var.
        host: Host URL for Ollama Cloud API (default: https://ollama.com).
        temperature: Controls randomness (0.0-2.0). Higher = more random.
        top_p: Nucleus sampling threshold (0.0-1.0).
        max_tokens: Maximum tokens to generate.
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
        OllamaCloudLLMImportError: If the Ollama client is not installed.
        OllamaCloudLLMAPIError: If all retry attempts fail.
        OllamaCloudLLMResponseError: If response is invalid.
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

    # API Key Configuration
    api_key = api_key or os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise OllamaCloudLLMImportError(
            "No API key provided and environment variable OLLAMA_API_KEY is not set"
        )

    # Import Ollama client
    try:
        from ollama import Client
    except ImportError:
        raise OllamaCloudLLMImportError(
            "ollama package not installed. Install with: pip install ollama"
        )

    # Build options
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    # Retry Loop
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            client = Client(
                host=host,
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=timeout
            )
            
            response = client.chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }],
                options=options if options else None,
                stream=False
            )
            
            # Extract content
            content = ""
            if isinstance(response, dict) and 'message' in response:
                message = response['message']
                if isinstance(message, dict) and 'content' in message:
                    content = message['content']
            
            if not content:
                raise OllamaCloudLLMResponseError("No content in response")
            
            # Extract usage
            usage_data = {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            }
            
            if isinstance(response, dict):
                # Try to get token counts from response
                if 'prompt_eval_count' in response:
                    usage_data["input_tokens"] = response['prompt_eval_count']
                if 'eval_count' in response:
                    usage_data["output_tokens"] = response['eval_count']
                
                if usage_data["input_tokens"] and usage_data["output_tokens"]:
                    usage_data["total_tokens"] = usage_data["input_tokens"] + usage_data["output_tokens"]
            
            return {
                "content": content,
                "model": model,
                "usage": usage_data,
            }

        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                raise OllamaCloudLLMAPIError(
                    f"Ollama Cloud LLM request failed after {max_retries} attempts: {exc}"
                ) from exc
            time.sleep(backoff_factor * (2 ** (attempt - 1)))

    raise OllamaCloudLLMAPIError("Ollama Cloud LLM request failed") from last_exc


def ollama_cloud_llm_stream(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    host: str = "https://ollama.com",
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = 30.0,
):
    """Call Ollama Cloud API and stream the generated text chunks.

    Args:
        prompt: The prompt / input text to send to the model.
        model: Model identifier (e.g. "llama2", "mistral", "codellama").
        api_key: API key to use. If omitted, uses OLLAMA_API_KEY env var.
        host: Host URL for Ollama Cloud API (default: https://ollama.com).
        temperature: Controls randomness (0.0-2.0).
        top_p: Nucleus sampling threshold (0.0-1.0).
        max_tokens: Maximum tokens to generate.
        timeout: Optional timeout (seconds).

    Yields:
        Text chunks as they are generated.

    Raises:
        ValueError: If required arguments are invalid.
        OllamaCloudLLMImportError: If client not installed.
        OllamaCloudLLMAPIError: If streaming request fails.
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

    # API Key
    api_key = api_key or os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise OllamaCloudLLMImportError(
            "No API key provided and environment variable OLLAMA_API_KEY is not set"
        )

    # Import
    try:
        from ollama import Client
    except ImportError:
        raise OllamaCloudLLMImportError(
            "ollama package not installed. Install with: pip install ollama"
        )

    # Build options
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    try:
        client = Client(
            host=host,
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=timeout
        )
        
        stream = client.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
            }],
            options=options if options else None,
            stream=True
        )
        
        for chunk in stream:
            if isinstance(chunk, dict) and 'message' in chunk:
                message = chunk['message']
                if isinstance(message, dict) and 'content' in message:
                    content = message['content']
                    if content:
                        yield content

    except Exception as exc:
        raise OllamaCloudLLMAPIError(f"Ollama Cloud streaming failed: {exc}") from exc


class OllamaCloudLLM:
    """
    Class-based wrapper for Ollama Cloud LLM.
    
    Example:
        >>> llm = OllamaCloudLLM(model="llama2", api_key="your-key")
        >>> response = llm.generate_response("What is Python?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        host: str = "https://ollama.com",
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        backoff_factor: float = 0.5,
    ):
        """Initialize Ollama Cloud LLM wrapper."""
        self.model = model
        self.api_key = api_key
        self.host = host
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response from Ollama Cloud.
        
        Returns only the text content for backward compatibility.
        Use ollama_cloud_llm() function directly for full metadata.
        """
        result = ollama_cloud_llm(
            prompt=prompt,
            model=self.model,
            api_key=self.api_key,
            host=self.host,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            timeout=self.timeout,
            backoff_factor=self.backoff_factor,
        )
        return result["content"]
    
    def generate_response_stream(self, prompt: str):
        """Generate a streaming response from Ollama Cloud.
        
        Yields text chunks as they are generated.
        
        Example:
            >>> llm = OllamaCloudLLM(model="llama2")
            >>> for chunk in llm.generate_response_stream("Write a story"):
            ...     print(chunk, end='', flush=True)
        """
        return ollama_cloud_llm_stream(
            prompt=prompt,
            model=self.model,
            api_key=self.api_key,
            host=self.host,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )


__all__ = [
    "ollama_cloud_llm",
    "ollama_cloud_llm_stream",
    "OllamaCloudLLM",
    "OllamaCloudLLMError",
    "OllamaCloudLLMAPIError",
    "OllamaCloudLLMImportError",
    "OllamaCloudLLMResponseError",
]
