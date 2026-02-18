"""
Google Gemini LLM Provider

Production-ready wrapper around Google's Generative AI client.

Author: Jentis Developer
Version: 1.0.0
"""

from typing import Optional, Any
import os
import time
import warnings
import sys
from contextlib import contextmanager

# ============================================================================
# Environment Configuration
# ============================================================================
# Suppress verbose logging from gRPC and underlying libraries
# Google's generativeai uses gRPC which can produce verbose ALTS warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'          # Suppress gRPC verbose logs
os.environ['GLOG_minloglevel'] = '2'            # Suppress Google logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # Suppress TensorFlow logs

# Suppress Python warnings from gRPC modules
warnings.filterwarnings('ignore', category=UserWarning, module='.*grpc.*')

@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output using low-level file descriptor redirection."""
    import io
    
    original_stderr = sys.stderr
    original_stderr_fd = None
    
    try:
        # Save and redirect file descriptor
        try:
            original_stderr_fd = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
        except Exception:
            pass
        
        # Redirect Python's sys.stderr
        sys.stderr = io.StringIO()
        yield
    finally:
        # Restore stderr
        if original_stderr_fd is not None:
            try:
                os.dup2(original_stderr_fd, 2)
                os.close(original_stderr_fd)
            except Exception:
                pass
        sys.stderr = original_stderr

# ============================================================================
# Module-Level Client Import
# ============================================================================
# Import Google generativeai client at module level for better performance
# Handle multiple packaging strategies (google.generativeai vs google.genai)
_GOOGLE_GENAI_AVAILABLE = False
genai_module = None
try:
    with suppress_stderr():
        try:
            # Preferred: standard packaging
            import google.generativeai as genai_module  # type: ignore
            _GOOGLE_GENAI_AVAILABLE = True
        except Exception:
            # Fallback: alternate or older packaging structure
            from google import genai as genai_module  # type: ignore
            _GOOGLE_GENAI_AVAILABLE = True
except Exception:
    _GOOGLE_GENAI_AVAILABLE = False
    genai_module = None


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================
class GoogleLLMError(Exception):
    """
    Base exception class for all Google Gemini LLM-related errors.
    
    All custom exceptions in this module inherit from this class,
    allowing users to catch any module-specific error with a single
    except clause.
    """


class GoogleLLMImportError(GoogleLLMError):
    """
    Raised when the Google generative AI client library cannot be imported or initialized.
    
    Common causes:
        - google-generativeai package not installed (pip install google-generativeai)
        - Missing or invalid API key
        - Client initialization failure
        - Incompatible SDK version
    """


class GoogleLLMAPIError(GoogleLLMError):
    """
    Raised when the API request fails after all retry attempts.
    
    Common causes:
        - Network connectivity issues
        - API service unavailable
        - Rate limiting or quota exceeded
        - Invalid model name or parameters
        - Authentication failures
    """


class GoogleLLMResponseError(GoogleLLMError):
    """
    Raised when the API response cannot be interpreted or is malformed.
    
    Common causes:
        - Empty response from API
        - Missing expected fields in response structure
        - Unexpected response format from SDK version
        - Content safety filters blocked the response
    """


def _extract_text_from_response(resp: Any) -> Optional[str]:
    """
    Extract generated text from Google API response using defensive parsing.
    
    Optimized with early returns and minimal overhead for common response formats.
    Handles multiple SDK versions and response structures for maximum compatibility.
    
    Args:
        resp: Response object from Google Gemini API
    
    Returns:
        Extracted text string if found, None if extraction fails
    """
    if resp is None:
        return None

    # Strategy 1: Direct .text attribute (most common)
    try:
        text = getattr(resp, "text", None)
        if text is not None:
            if callable(text):
                text = text()
            if isinstance(text, str) and text.strip():
                return text
    except Exception:
        pass

    # Strategy 2: Structured candidates format
    candidates = getattr(resp, "candidates", None)
    if candidates and isinstance(candidates, (list, tuple)) and candidates:
        first = candidates[0]
        
        # Try candidate.content.parts[0].text
        content = getattr(first, "content", None)
        if content:
            parts = getattr(content, "parts", None)
            if parts and isinstance(parts, (list, tuple)) and parts:
                part_text = getattr(parts[0], "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    return part_text
            
            # Try content.text
            content_text = getattr(content, "text", None)
            if isinstance(content_text, str) and content_text.strip():
                return content_text
        
        # Try candidate.text
        candidate_text = getattr(first, "text", None)
        if isinstance(candidate_text, str) and candidate_text.strip():
            return candidate_text

    # Strategy 3: Dictionary-based responses
    if isinstance(resp, dict):
        candidates = resp.get("candidates")
        if candidates and isinstance(candidates, (list, tuple)) and candidates:
            first = candidates[0]
            if isinstance(first, dict):
                content = first.get("content")
                if isinstance(content, dict):
                    parts = content.get("parts")
                    if isinstance(parts, (list, tuple)) and parts:
                        part = parts[0]
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str) and text.strip():
                                return text
                
                # Try simpler structures
                text = first.get("content") or first.get("text")
                if isinstance(text, str) and text.strip():
                    return text
        
        # Try top-level text field
        text = resp.get("text")
        if isinstance(text, str) and text.strip():
            return text

    return None


def _extract_usage_from_response(resp: Any) -> dict:
    """
    Extract usage metadata (token counts) from Google API response.
    
    Args:
        resp: Response object from Google Gemini API
    
    Returns:
        Dictionary with token usage information
    """
    usage_data = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    
    if resp is None:
        return usage_data
    
    # Strategy 1: Direct usage_metadata attribute
    try:
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            usage_data["input_tokens"] = getattr(usage, "prompt_token_count", None)
            usage_data["output_tokens"] = getattr(usage, "candidates_token_count", None)
            usage_data["total_tokens"] = getattr(usage, "total_token_count", None)
            return usage_data
    except Exception:
        pass
    
    # Strategy 2: Dictionary-based usage metadata
    if isinstance(resp, dict):
        usage = resp.get("usage_metadata") or resp.get("usage")
        if isinstance(usage, dict):
            usage_data["input_tokens"] = usage.get("prompt_token_count") or usage.get("promptTokens")
            usage_data["output_tokens"] = usage.get("candidates_token_count") or usage.get("completionTokens")
            usage_data["total_tokens"] = usage.get("total_token_count") or usage.get("totalTokens")
    
    return usage_data


def google_llm(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    timeout: Optional[float] = 30.0,
    backoff_factor: float = 0.5,
) -> dict:
    """Call a Google generative model and return structured response with metadata.

    Args:
        prompt: The prompt / input text to send to the model. Must be non-empty.
        model: Model identifier (e.g. "gemini-pro" or other supported model name).
        api_key: API key to use. If omitted, the function will try the
            environment variable ``GOOGLE_API_KEY``.
        temperature: Controls randomness (0.0-2.0). Higher = more random.
        top_p: Nucleus sampling threshold (0.0-1.0). Alternative to temperature.
        top_k: Top-k sampling. Limits to k most likely tokens.
        max_tokens: Maximum tokens to generate (max_output_tokens).
        max_retries: Number of attempts to make on transient failures.
        timeout: Optional timeout (seconds) to pass to the underlying client.
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
        GoogleLLMImportError: If the Google client is not installed.
        GoogleLLMAPIError: If all retry attempts fail.
        GoogleLLMResponseError: If a response is returned but contains no text.
    """

    # Input Validation
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if not isinstance(max_retries, int) or max_retries < 1:
        raise ValueError("max_retries must be an integer >= 1")
    
    # Validate generation parameters
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")
    if top_p is not None and not (0.0 <= top_p <= 1.0):
        raise ValueError("top_p must be between 0.0 and 1.0")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    # Build Generation Configuration
    generation_config = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if top_p is not None:
        generation_config["top_p"] = top_p
    if top_k is not None:
        generation_config["top_k"] = top_k
    if max_tokens is not None:
        generation_config["max_output_tokens"] = max_tokens

    # API Key Configuration
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise GoogleLLMImportError(
            "No API key provided and environment variable GOOGLE_API_KEY is not set"
        )

    # Client Availability Check
    if not _GOOGLE_GENAI_AVAILABLE or genai_module is None:
        raise GoogleLLMImportError(
            "Failed to import or initialize google.generativeai client"
        )

    # Client Configuration
    genai = genai_module
    client = None
    
    # Configure API key (global state for some SDK versions)
    try:
        with suppress_stderr():
            cfg = getattr(genai, "configure", None)
            if callable(cfg):
                cfg(api_key=api_key)
    except Exception:
        pass

    # Initialize client object (for client-based API pattern)
    try:
        with suppress_stderr():
            ClientCls = getattr(genai, "Client", None)
            if callable(ClientCls):
                try:
                    client = ClientCls(api_key=api_key)
                except TypeError:
                    client = ClientCls()
    except Exception:
        pass

    # Retry Loop with Exponential Backoff
    last_exc: Optional[BaseException] = None

    for attempt in range(1, max_retries + 1):
        try:
            with suppress_stderr():
                # Strategy 1: Client-based API
                if client is not None:
                    models_attr = getattr(client, "models", None)
                    gen_fn = getattr(models_attr, "generate_content", None) if models_attr else None
                    if callable(gen_fn):
                        resp = gen_fn(model=model, contents=prompt)
                        text = _extract_text_from_response(resp)
                        if text:
                            usage = _extract_usage_from_response(resp)
                            return {
                                "content": text,
                                "model": model,
                                "usage": usage,
                            }

                # Strategy 2: GenerativeModel Class
                GenerativeModel = getattr(genai, "GenerativeModel", None)
                if callable(GenerativeModel):
                    try:
                        if generation_config:
                            model_obj = GenerativeModel(model, generation_config=generation_config)
                        else:
                            model_obj = GenerativeModel(model)
                        
                        gen_fn = getattr(model_obj, "generate_content", None)
                        if callable(gen_fn):
                            resp = gen_fn(prompt)
                            text = _extract_text_from_response(resp)
                            if text:
                                usage = _extract_usage_from_response(resp)
                                return {
                                    "content": text,
                                    "model": model,
                                    "usage": usage,
                                }
                    except Exception:
                        pass

                # Strategy 3: Top-level convenience functions
                for helper_name in ("generate_text", "generate", "model_generate"):
                    helper = getattr(genai, helper_name, None)
                    if callable(helper):
                        try:
                            resp = helper(model=model, prompt=prompt)
                            text = _extract_text_from_response(resp)
                            if text:
                                usage = _extract_usage_from_response(resp)
                                return {
                                    "content": text,
                                    "model": model,
                                    "usage": usage,
                                }
                        except Exception:
                            pass

            # No valid response extracted
            raise GoogleLLMResponseError("No text could be extracted from the API response")

        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                raise GoogleLLMAPIError(
                    f"Google LLM request failed after {max_retries} attempts: {exc}"
                ) from exc

            # Exponential backoff
            time.sleep(backoff_factor * (2 ** (attempt - 1)))

    raise GoogleLLMAPIError("Google LLM request failed") from last_exc


def google_llm_stream(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = 30.0,
):
    """Call a Google generative model and stream the generated text chunks.

    Args:
        prompt: The prompt / input text to send to the model. Must be non-empty.
        model: Model identifier (e.g. "gemini-pro" or other supported model name).
        api_key: API key to use. If omitted, the function will try the
            environment variable ``GOOGLE_API_KEY``.
        temperature: Controls randomness (0.0-2.0). Higher = more random.
        top_p: Nucleus sampling threshold (0.0-1.0). Alternative to temperature.
        top_k: Top-k sampling. Limits to k most likely tokens.
        max_tokens: Maximum tokens to generate (max_output_tokens).
        timeout: Optional timeout (seconds) to pass to the underlying client.

    Yields:
        Text chunks as they are generated by the model.

    Raises:
        ValueError: If required arguments are missing or invalid.
        GoogleLLMImportError: If the Google client is not installed.
        GoogleLLMAPIError: If the streaming request fails.
        GoogleLLMResponseError: If response chunks cannot be extracted.
    """

    # Input Validation
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    
    # Validate generation parameters
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("temperature must be between 0.0 and 2.0")
    if top_p is not None and not (0.0 <= top_p <= 1.0):
        raise ValueError("top_p must be between 0.0 and 1.0")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    # Build Generation Configuration
    generation_config = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if top_p is not None:
        generation_config["top_p"] = top_p
    if top_k is not None:
        generation_config["top_k"] = top_k
    if max_tokens is not None:
        generation_config["max_output_tokens"] = max_tokens

    # API Key Configuration
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise GoogleLLMImportError(
            "No API key provided and environment variable GOOGLE_API_KEY is not set"
        )

    # Client Availability Check
    if not _GOOGLE_GENAI_AVAILABLE or genai_module is None:
        raise GoogleLLMImportError(
            "Failed to import or initialize google.generativeai client"
        )

    # Client Configuration
    genai = genai_module
    client = None
    
    # Configure API key (global state for some SDK versions)
    try:
        with suppress_stderr():
            cfg = getattr(genai, "configure", None)
            if callable(cfg):
                cfg(api_key=api_key)
    except Exception:
        pass

    # Initialize client object (for client-based API pattern)
    try:
        with suppress_stderr():
            ClientCls = getattr(genai, "Client", None)
            if callable(ClientCls):
                try:
                    client = ClientCls(api_key=api_key)
                except TypeError:
                    client = ClientCls()
    except Exception:
        pass

    # Streaming Request
    try:
        with suppress_stderr():
            # Strategy 1: GenerativeModel with stream parameter
            GenerativeModel = getattr(genai, "GenerativeModel", None)
            if callable(GenerativeModel):
                try:
                    if generation_config:
                        model_obj = GenerativeModel(model, generation_config=generation_config)
                    else:
                        model_obj = GenerativeModel(model)
                    
                    gen_fn = getattr(model_obj, "generate_content", None)
                    if callable(gen_fn):
                        response_stream = gen_fn(prompt, stream=True)
                        
                        # Iterate through streaming response
                        for chunk in response_stream:  # type: ignore
                            text = _extract_text_from_response(chunk)
                            if text:
                                yield text
                        return
                except Exception:
                    # If this strategy fails, try alternatives
                    pass
            
            # Strategy 2: Client-based streaming API
            if client is not None:
                models_attr = getattr(client, "models", None)
                stream_fn = getattr(models_attr, "stream_generate_content", None) if models_attr else None
                if callable(stream_fn):
                    response_stream = stream_fn(model=model, contents=prompt, timeout=timeout)
                    for chunk in response_stream:  # type: ignore
                        text = _extract_text_from_response(chunk)
                        if text:
                            yield text
                    return
            
            # Strategy 3: Explicit stream method
            if callable(GenerativeModel):
                try:
                    if generation_config:
                        model_obj = GenerativeModel(model, generation_config=generation_config)
                    else:
                        model_obj = GenerativeModel(model)
                    
                    stream_fn = getattr(model_obj, "generate_content_stream", None)
                    if callable(stream_fn):
                        response_stream = stream_fn(prompt)
                        for chunk in response_stream:  # type: ignore
                            text = _extract_text_from_response(chunk)
                            if text:
                                yield text
                        return
                except Exception:
                    pass

        raise GoogleLLMAPIError("Failed to initialize streaming response")

    except Exception as exc:
        raise GoogleLLMAPIError(
            f"Google LLM streaming request failed: {exc}"
        ) from exc


class GoogleLLM:
    """
    Class-based wrapper for Google Gemini LLM with generate_response method.
    
    This class wraps the google_llm function to provide a stateful interface
    suitable for use with agents and other systems that expect an object
    with a generate_response(prompt) method.
    
    Example:
        >>> llm = GoogleLLM(model="gemini-1.5-pro", api_key="your-key")
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
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        timeout: Optional[float] = 30.0,
        backoff_factor: float = 0.5,
    ):
        """
        Initialize Google Gemini LLM wrapper.
        
        Args:
            model: Model identifier (e.g. "gemini-2.5-pro", "gemini-3.0-pro")
            api_key: API key (optional if GOOGLE_API_KEY env var is set)
            temperature: Controls randomness (0.0-2.0)
            top_p: Nucleus sampling threshold (0.0-1.0)
            top_k: Top-k sampling
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts on failure
            timeout: Request timeout in seconds
            backoff_factor: Exponential backoff factor for retries
        """
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
        """
        Generate a response from the Google Gemini model.
        
        Args:
            prompt: The input prompt text
            
        Returns:
            Generated response text (content only, not full metadata)
            
        Raises:
            ValueError: If prompt is invalid
            GoogleLLMImportError: If Google client not available
            GoogleLLMAPIError: If API request fails
            GoogleLLMResponseError: If response is invalid
            
        Note:
            This method returns only the text content for backward compatibility.
            Use the google_llm() function directly if you need full metadata.
        """
        result = google_llm(
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
        """
        Generate a streaming response from the Google Gemini model.
        
        This method yields text chunks as they are generated by the model,
        allowing for real-time display of the response.
        
        Args:
            prompt: The input prompt text
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            ValueError: If prompt is invalid
            GoogleLLMImportError: If Google client not available
            GoogleLLMAPIError: If streaming request fails
            GoogleLLMResponseError: If response chunks cannot be extracted
            
        Example:
            >>> llm = GoogleLLM(model="gemini-1.5-pro", api_key="your-key")
            >>> for chunk in llm.generate_response_stream("Write a story"):
            ...     print(chunk, end='', flush=True)
        """
        return google_llm_stream(
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
    "google_llm",
    "google_llm_stream",
    "GoogleLLM",
    "GoogleLLMError",
    "GoogleLLMAPIError",
    "GoogleLLMImportError",
    "GoogleLLMResponseError",
]