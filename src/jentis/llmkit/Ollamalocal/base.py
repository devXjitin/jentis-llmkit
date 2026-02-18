"""
Ollama Local LLM Provider

Production-ready wrapper for local Ollama instance using native ollama client.

Author: Jentis Developer
Version: 1.0.0
"""

from typing import Optional
import time


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================
class OllamaLocalLLMError(Exception):
    """Base exception class for all Ollama Local LLM-related errors."""


class OllamaLocalLLMImportError(OllamaLocalLLMError):
    """Raised when the ollama client library cannot be imported or initialized."""


class OllamaLocalLLMAPIError(OllamaLocalLLMError):
    """Raised when the API request fails after all retry attempts."""


class OllamaLocalLLMResponseError(OllamaLocalLLMError):
    """Raised when the API response cannot be interpreted or is malformed."""


def ollama_local_llm(
    prompt: str,
    model: str,
    host: str = "http://localhost:11434",
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    timeout: Optional[float] = 30.0,
    backoff_factor: float = 0.5,
) -> dict:
    """Call local Ollama API and return structured response with metadata.

    Args:
        prompt: The prompt / input text to send to the model. Must be non-empty.
        model: Model identifier (e.g. "llama2", "mistral", "codellama").
        host: Host URL for local Ollama instance (default: http://localhost:11434).
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
        OllamaLocalLLMImportError: If the ollama client is not installed.
        OllamaLocalLLMAPIError: If all retry attempts fail.
        OllamaLocalLLMResponseError: If response is invalid.
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

    # Import ollama client
    try:
        from ollama import Client
    except ImportError:
        raise OllamaLocalLLMImportError(
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
            content = response.get('message', {}).get('content', '')
            
            if not content:
                raise OllamaLocalLLMResponseError("No content in response")
            
            # Extract usage (optional in Ollama)
            input_tokens = response.get('prompt_eval_count')
            output_tokens = response.get('eval_count')
            total_tokens = None
            if input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
            
            usage_data = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
            
            return {
                "content": content,
                "model": model,
                "usage": usage_data,
            }

        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                raise OllamaLocalLLMAPIError(
                    f"Ollama Local LLM request failed after {max_retries} attempts: {exc}"
                ) from exc
            time.sleep(backoff_factor * (2 ** (attempt - 1)))

    raise OllamaLocalLLMAPIError("Ollama Local LLM request failed") from last_exc


def ollama_local_llm_stream(
    prompt: str,
    model: str,
    host: str = "http://localhost:11434",
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = 30.0,
):
    """Call local Ollama API and stream the generated text chunks.

    Args:
        prompt: The prompt / input text to send to the model.
        model: Model identifier (e.g. "llama2", "mistral", "codellama").
        host: Host URL for local Ollama instance (default: http://localhost:11434).
        temperature: Controls randomness (0.0-2.0).
        top_p: Nucleus sampling threshold (0.0-1.0).
        max_tokens: Maximum tokens to generate.
        timeout: Optional timeout (seconds).

    Yields:
        Text chunks as they are generated.

    Raises:
        ValueError: If required arguments are invalid.
        OllamaLocalLLMImportError: If client not installed.
        OllamaLocalLLMAPIError: If streaming request fails.
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

    # Import
    try:
        from ollama import Client
    except ImportError:
        raise OllamaLocalLLMImportError(
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
        raise OllamaLocalLLMAPIError(f"Ollama Local streaming failed: {exc}") from exc


class OllamaLocalLLM:
    """
    Class-based wrapper for local Ollama LLM.
    
    Example:
        >>> llm = OllamaLocalLLM(model="llama2")
        >>> response = llm.generate_response("What is Python?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        backoff_factor: float = 0.5,
    ):
        """Initialize local Ollama LLM wrapper."""
        self.model = model
        self.host = host
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response from local Ollama.
        
        Returns only the text content for backward compatibility.
        Use ollama_local_llm() function directly for full metadata.
        """
        result = ollama_local_llm(
            prompt=prompt,
            model=self.model,
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
        """Generate a streaming response from local Ollama.
        
        Yields text chunks as they are generated.
        
        Example:
            >>> llm = OllamaLocalLLM(model="llama2")
            >>> for chunk in llm.generate_response_stream("Write a story"):
            ...     print(chunk, end='', flush=True)
        """
        return ollama_local_llm_stream(
            prompt=prompt,
            model=self.model,
            host=self.host,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )


__all__ = [
    "ollama_local_llm",
    "ollama_local_llm_stream",
    "OllamaLocalLLM",
    "OllamaLocalLLMError",
    "OllamaLocalLLMAPIError",
    "OllamaLocalLLMImportError",
    "OllamaLocalLLMResponseError",
]
