"""
Google Vertex AI LLM Provider (via OpenAI-compatible endpoint)

Production-ready wrapper using Python built-in urllib (no pip dependency).
Authenticates via `gcloud auth print-access-token` or a user-supplied token.
Supports first-party (Gemini) and third-party Model Garden models such as
Claude, Llama, Mistral, Moonshot Kimi, DeepSeek, and more.

Author: Jentis Developer
Version: 1.0.0
"""

from typing import Optional, Generator
import json
import os
import subprocess
import time
import urllib.request
import urllib.error
import ssl


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================
class VertexAILLMError(Exception):
    """Base exception class for all Vertex AI LLM-related errors."""


class VertexAILLMAuthError(VertexAILLMError):
    """Raised when authentication fails (missing token / gcloud unavailable)."""


class VertexAILLMAPIError(VertexAILLMError):
    """Raised when the API request fails after all retry attempts."""


class VertexAILLMResponseError(VertexAILLMError):
    """Raised when the API response cannot be interpreted or is malformed."""


# ============================================================================
# Helper: obtain access token
# ============================================================================
def _find_gcloud() -> Optional[str]:
    """Locate the gcloud CLI executable.

    Checks PATH first, then falls back to well-known Windows/Linux/macOS
    installation directories so it works even right after installation
    (before a terminal restart refreshes PATH).

    Returns:
        Absolute path to gcloud executable, or None if not found.
    """
    import shutil
    import pathlib
    import platform

    # 1. Check PATH
    found = shutil.which("gcloud")
    if found:
        return found

    # 2. Probe common installation directories
    candidates: list[str] = []

    if platform.system() == "Windows":
        local_app = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        home = pathlib.Path.home()

        candidates = [
            os.path.join(local_app, "Google", "Cloud SDK", "google-cloud-sdk", "bin", "gcloud.cmd"),
            os.path.join(program_files, "Google", "Cloud SDK", "google-cloud-sdk", "bin", "gcloud.cmd"),
            os.path.join(program_x86, "Google", "Cloud SDK", "google-cloud-sdk", "bin", "gcloud.cmd"),
            str(home / "AppData" / "Local" / "Google" / "Cloud SDK" / "google-cloud-sdk" / "bin" / "gcloud.cmd"),
            str(home / "google-cloud-sdk" / "bin" / "gcloud.cmd"),
        ]
    else:
        home = pathlib.Path.home()
        candidates = [
            str(home / "google-cloud-sdk" / "bin" / "gcloud"),
            "/usr/lib/google-cloud-sdk/bin/gcloud",
            "/usr/local/bin/gcloud",
            "/snap/bin/gcloud",
        ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


def _get_access_token(access_token: Optional[str] = None) -> str:
    """Return a valid GCP access token.

    Resolution order:
        1. Explicitly supplied *access_token* parameter.
        2. ``VERTEX_AI_ACCESS_TOKEN`` environment variable.
        3. ``gcloud auth print-access-token`` CLI command.

    Raises:
        VertexAILLMAuthError: If no token can be obtained.
    """
    token = access_token or os.environ.get("VERTEX_AI_ACCESS_TOKEN")
    if token:
        return token.strip()

    # Fall back to gcloud CLI
    gcloud_path = _find_gcloud()
    if not gcloud_path:
        raise VertexAILLMAuthError(
            "No access token provided, VERTEX_AI_ACCESS_TOKEN is not set, "
            "and 'gcloud' CLI is not found on PATH or common install locations. "
            "Install the Google Cloud SDK or provide a token explicitly."
        )

    try:
        result = subprocess.run(
            [gcloud_path, "auth", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        raise VertexAILLMAuthError(
            f"gcloud returned non-zero exit code ({result.returncode}). "
            f"stderr: {result.stderr.strip()}"
        )
    except FileNotFoundError:
        raise VertexAILLMAuthError(
            "No access token provided, VERTEX_AI_ACCESS_TOKEN is not set, "
            "and 'gcloud' CLI could not be executed. Install the Google Cloud SDK "
            "or provide a token explicitly."
        )
    except subprocess.TimeoutExpired:
        raise VertexAILLMAuthError(
            "gcloud auth print-access-token timed out after 15 seconds."
        )


# ============================================================================
# Helper: build endpoint URL
# ============================================================================
def _build_url(
    endpoint: str,
    project_id: str,
    region: str,
) -> str:
    """Construct the Vertex AI OpenAI-compatible chat completions URL.

    Supports both the ``global`` endpoint and regional endpoints.

    Examples:
        Global:
            https://aiplatform.googleapis.com/v1beta1/projects/{pid}/locations/global/endpoints/openapi/chat/completions
        Regional:
            https://{region}-aiplatform.googleapis.com/v1beta1/projects/{pid}/locations/{region}/endpoints/openapi/chat/completions
    """
    base = f"https://{endpoint}/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi/chat/completions"
    return base


def _default_endpoint(region: str) -> str:
    """Return the appropriate hostname for the given region."""
    if region == "global":
        return "aiplatform.googleapis.com"
    return f"{region}-aiplatform.googleapis.com"


# ============================================================================
# Core: non-streaming request
# ============================================================================
def vertexai_llm(
    prompt: str,
    model: str,
    project_id: Optional[str] = None,
    *,
    region: str = "global",
    endpoint: Optional[str] = None,
    access_token: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    timeout: Optional[float] = 60.0,
    backoff_factor: float = 0.5,
) -> dict:
    """Call Vertex AI OpenAI-compatible endpoint and return a structured response.

    This function uses **only** Python built-in modules (``urllib``, ``json``,
    ``subprocess``) — no pip dependency is required.

    Args:
        prompt: The user message to send to the model. Must be non-empty.
        model: Model identifier on Vertex AI Model Garden.
            Examples:
                - ``google/gemini-2.0-flash``  (first-party)
                - ``meta/llama-4-maverick-17b-128e-instruct-maas``
                - ``anthropic/claude-sonnet-4@20250514``
                - ``mistralai/mistral-large-2411-maas``
                - ``moonshotai/kimi-k2-thinking-maas``
                - ``deepseek/deepseek-r1-0528-maas``
        project_id: GCP project ID. Falls back to ``VERTEX_AI_PROJECT_ID`` or
            ``GOOGLE_CLOUD_PROJECT`` environment variables.
        region: GCP region (default ``"global"``). Other examples:
            ``"us-central1"``, ``"europe-west4"``.
        endpoint: Override the API hostname. Defaults are auto-derived from *region*.
        access_token: Bearer token. If omitted the function tries the
            ``VERTEX_AI_ACCESS_TOKEN`` env var, then ``gcloud auth print-access-token``.
        temperature: Controls randomness (0.0–2.0).
        top_p: Nucleus sampling threshold (0.0–1.0).
        max_tokens: Maximum tokens to generate.
        max_retries: Retry attempts on transient failures.
        timeout: HTTP timeout in seconds.
        backoff_factor: Exponential back-off base factor.

    Returns:
        Dictionary containing:
            - content: The generated text
            - model: The model identifier
            - usage: Dictionary with ``input_tokens``, ``output_tokens``,
              ``total_tokens``

    Raises:
        ValueError: If required arguments are missing or invalid.
        VertexAILLMAuthError: If authentication fails.
        VertexAILLMAPIError: If all retries are exhausted.
        VertexAILLMResponseError: If the response is malformed.
    """
    # ── input validation ────────────────────────────────────────────────
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

    # ── resolve project / endpoint / token ──────────────────────────────
    project_id = project_id or os.environ.get("VERTEX_AI_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError(
            "project_id is required. Pass it explicitly or set the "
            "VERTEX_AI_PROJECT_ID / GOOGLE_CLOUD_PROJECT environment variable."
        )

    endpoint = endpoint or _default_endpoint(region)
    token = _get_access_token(access_token)
    url = _build_url(endpoint, project_id, region)

    # ── build request body ──────────────────────────────────────────────
    body: dict = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    payload = json.dumps(body).encode("utf-8")

    # ── SSL context ─────────────────────────────────────────────────────
    ctx = ssl.create_default_context()

    # ── retry loop ──────────────────────────────────────────────────────
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                method="POST",
            )
            timeout_sec = timeout if timeout else 60.0
            with urllib.request.urlopen(req, timeout=timeout_sec, context=ctx) as resp:
                raw = resp.read().decode("utf-8")

            data = json.loads(raw)

            # ── extract content ─────────────────────────────────────────
            content = ""
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")

            if not content:
                raise VertexAILLMResponseError(
                    f"No content in response. Raw: {raw[:500]}"
                )

            # ── extract usage ───────────────────────────────────────────
            usage_raw = data.get("usage", {})
            usage_data = {
                "input_tokens": usage_raw.get("prompt_tokens"),
                "output_tokens": usage_raw.get("completion_tokens"),
                "total_tokens": usage_raw.get("total_tokens"),
            }

            return {
                "content": content,
                "model": model,
                "usage": usage_data,
            }

        except (VertexAILLMResponseError,):
            raise  # don't retry on response-parse errors
        except urllib.error.HTTPError as exc:
            # Read the error body for a better diagnostic message
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            last_exc = exc
            if attempt == max_retries:
                raise VertexAILLMAPIError(
                    f"Vertex AI request failed after {max_retries} attempts: "
                    f"HTTP {exc.code} {exc.reason}. Response: {error_body[:1000]}"
                ) from exc
            time.sleep(backoff_factor * (2 ** (attempt - 1)))
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                raise VertexAILLMAPIError(
                    f"Vertex AI request failed after {max_retries} attempts: {exc}"
                ) from exc
            time.sleep(backoff_factor * (2 ** (attempt - 1)))

    raise VertexAILLMAPIError("Vertex AI request failed") from last_exc


# ============================================================================
# Core: streaming request
# ============================================================================
def vertexai_llm_stream(
    prompt: str,
    model: str,
    project_id: Optional[str] = None,
    *,
    region: str = "global",
    endpoint: Optional[str] = None,
    access_token: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = 60.0,
) -> Generator[str, None, None]:
    """Call Vertex AI and stream text chunks via SSE (Server-Sent Events).

    Uses only Python built-in modules — no pip dependency required.

    Args:
        prompt: User message to send.
        model: Model identifier on Vertex AI Model Garden.
        project_id: GCP project ID (falls back to env vars).
        region: GCP region (default ``"global"``).
        endpoint: Override API hostname.
        access_token: Bearer token (falls back to env var / gcloud).
        temperature: Controls randomness (0.0–2.0).
        top_p: Nucleus sampling threshold (0.0–1.0).
        max_tokens: Maximum tokens to generate.
        timeout: HTTP timeout in seconds.

    Yields:
        Text chunks as they are generated.

    Raises:
        ValueError: If required arguments are invalid.
        VertexAILLMAuthError: If authentication fails.
        VertexAILLMAPIError: If the streaming request fails.
    """
    # ── input validation ────────────────────────────────────────────────
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

    # ── resolve project / endpoint / token ──────────────────────────────
    project_id = project_id or os.environ.get("VERTEX_AI_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise ValueError(
            "project_id is required. Pass it explicitly or set the "
            "VERTEX_AI_PROJECT_ID / GOOGLE_CLOUD_PROJECT environment variable."
        )

    endpoint = endpoint or _default_endpoint(region)
    token = _get_access_token(access_token)
    url = _build_url(endpoint, project_id, region)

    # ── build request body ──────────────────────────────────────────────
    body: dict = {
        "model": model,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    payload = json.dumps(body).encode("utf-8")

    # ── SSL context ─────────────────────────────────────────────────────
    ctx = ssl.create_default_context()

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        timeout_sec = timeout if timeout else 60.0
        resp = urllib.request.urlopen(req, timeout=timeout_sec, context=ctx)

        # Read SSE stream line-by-line
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

        resp.close()

    except VertexAILLMError:
        raise
    except Exception as exc:
        raise VertexAILLMAPIError(
            f"Vertex AI streaming failed: {exc}"
        ) from exc


# ============================================================================
# Class-based wrapper
# ============================================================================
class VertexAILLM:
    """Class-based wrapper for Vertex AI LLM (OpenAI-compatible endpoint).

    Supports all models available via the Vertex AI Model Garden, including
    first-party (Gemini) and third-party models (Claude, Llama, Mistral,
    Moonshot Kimi, DeepSeek, etc.).

    Uses only Python built-in modules — no extra pip dependencies.

    Example:
        >>> llm = VertexAILLM(
        ...     model="moonshotai/kimi-k2-thinking-maas",
        ...     project_id="my-gcp-project",
        ... )
        >>> response = llm.generate_response("Hello, how are you?")
        >>> print(response)
    """

    def __init__(
        self,
        model: str,
        project_id: Optional[str] = None,
        *,
        region: str = "global",
        endpoint: Optional[str] = None,
        access_token: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        timeout: Optional[float] = 60.0,
        backoff_factor: float = 0.5,
    ):
        """Initialize Vertex AI LLM wrapper.

        Args:
            model: Model identifier on Vertex AI Model Garden.
            project_id: GCP project ID. Falls back to env vars.
            region: GCP region (default ``"global"``).
            endpoint: Override the API hostname.
            access_token: Bearer token (falls back to env var / gcloud).
            temperature: Controls randomness (0.0–2.0).
            top_p: Nucleus sampling threshold (0.0–1.0).
            max_tokens: Maximum tokens to generate.
            max_retries: Number of retry attempts.
            timeout: HTTP timeout in seconds.
            backoff_factor: Exponential back-off base factor.
        """
        self.model = model
        self.project_id = project_id
        self.region = region
        self.endpoint = endpoint
        self.access_token = access_token
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_factor = backoff_factor

    def generate_response(self, prompt: str) -> str:
        """Generate a response from Vertex AI.

        Returns only the text content for backward compatibility.
        Use ``vertexai_llm()`` function directly for full metadata.
        """
        result = vertexai_llm(
            prompt=prompt,
            model=self.model,
            project_id=self.project_id,
            region=self.region,
            endpoint=self.endpoint,
            access_token=self.access_token,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            timeout=self.timeout,
            backoff_factor=self.backoff_factor,
        )
        return result["content"]

    def generate_response_stream(self, prompt: str):
        """Generate a streaming response from Vertex AI.

        Yields text chunks as they are generated.

        Example:
            >>> llm = VertexAILLM(model="google/gemini-2.0-flash", project_id="my-proj")
            >>> for chunk in llm.generate_response_stream("Write a story"):
            ...     print(chunk, end='', flush=True)
        """
        return vertexai_llm_stream(
            prompt=prompt,
            model=self.model,
            project_id=self.project_id,
            region=self.region,
            endpoint=self.endpoint,
            access_token=self.access_token,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )


__all__ = [
    "vertexai_llm",
    "vertexai_llm_stream",
    "VertexAILLM",
    "VertexAILLMError",
    "VertexAILLMAPIError",
    "VertexAILLMAuthError",
    "VertexAILLMResponseError",
]
