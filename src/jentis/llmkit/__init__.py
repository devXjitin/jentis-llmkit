"""
Jentis LLM Kit - Unified LLM Provider Interface

A unified interface for multiple LLM providers including Google Gemini, 
Anthropic Claude, OpenAI GPT, Grok, Azure OpenAI, and Ollama.

Author: Jentis Developer
Version: 1.0.1
"""

from typing import Optional, Any

# ── Direct class imports for convenience ────────────────────────────────
# Allows: from jentis.llmkit import OpenAILLM, VertexAILLM, etc.
from jentis.llmkit.Google.base import GoogleLLM
from jentis.llmkit.Anthropic.base import AnthropicLLM
from jentis.llmkit.Openai.base import OpenAILLM
from jentis.llmkit.Grok.base import GrokLLM
from jentis.llmkit.Microsoft.base import AzureLLM
from jentis.llmkit.Ollamacloud.base import OllamaCloudLLM
from jentis.llmkit.Ollamalocal.base import OllamaLocalLLM
from jentis.llmkit.Vertexai.base import VertexAILLM


def init_llm(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    **kwargs
) -> Any:
    """Initialize an LLM provider with a unified interface.
    
    Args:
        provider: LLM provider name. Supported values:
            - "google" or "gemini" - Google Gemini
            - "anthropic" or "claude" - Anthropic Claude
            - "openai" or "gpt" - OpenAI GPT models
            - "grok" or "xai" - xAI Grok
            - "azure" or "microsoft" - Azure OpenAI
            - "ollama-cloud" - Ollama Cloud
            - "ollama-local" or "ollama" - Local Ollama
            - "vertexai" or "vertex-ai" or "vertex" - Google Vertex AI Model Garden
        model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        api_key: API key for the provider (if required)
        **kwargs: Additional provider-specific parameters:
            - temperature: Controls randomness (0.0-2.0)
            - top_p: Nucleus sampling threshold (0.0-1.0)
            - max_tokens: Maximum tokens to generate
            - timeout: Request timeout in seconds
            - max_retries: Maximum retry attempts
            - backoff_factor: Exponential backoff factor
            
            Azure-specific:
            - azure_endpoint: Azure OpenAI endpoint URL
            - api_version: Azure API version
            - deployment_name: Azure deployment name
            
            Ollama-specific:
            - host: Host URL (for cloud or local)
            - base_url: Alternative to host parameter
            
            Vertex AI-specific:
            - project_id: GCP project ID (or set VERTEX_AI_PROJECT_ID env var)
            - region: GCP region (default "global")
            - endpoint: Override API hostname
            - access_token: Bearer token (or pass via api_key)
    
    Returns:
        An initialized LLM class instance with generate_response() and 
        generate_response_stream() methods.
    
    Raises:
        ValueError: If provider is not supported or required parameters are missing.
    
    Example:
        >>> from jentis.llmkit import init_llm
        >>> 
        >>> # OpenAI GPT-4
        >>> llm = init_llm(provider="openai", model="gpt-4o", api_key="your-key")
        >>> response = llm.generate_response("What is Python?")
        >>> print(response)
        >>> 
        >>> # Streaming response
        >>> for chunk in llm.generate_response_stream("Write a story"):
        ...     print(chunk, end='', flush=True)
        >>> 
        >>> # Google Gemini
        >>> llm = init_llm(provider="google", model="gemini-2.0-flash-exp", api_key="your-key")
        >>> response = llm.generate_response("Explain AI")
        >>> 
        >>> # Anthropic Claude
        >>> llm = init_llm(provider="anthropic", model="claude-3-5-sonnet-20241022", 
        ...                api_key="your-key", max_tokens=1024)
        >>> 
        >>> # Local Ollama
        >>> llm = init_llm(provider="ollama", model="llama2")
        >>> response = llm.generate_response("Hello!")
    """
    provider_lower = provider.lower().strip()
    
    # Google Gemini
    if provider_lower in ["google", "gemini"]:
        from jentis.llmkit.Google.base import GoogleLLM
        return GoogleLLM(
            model=model,
            api_key=api_key,
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            timeout=kwargs.get("timeout", 30.0),
        )
    
    # Anthropic Claude
    elif provider_lower in ["anthropic", "claude"]:
        from jentis.llmkit.Anthropic.base import AnthropicLLM
        
        # Anthropic requires max_tokens
        max_tokens = kwargs.get("max_tokens", 1024)
        
        return AnthropicLLM(
            model=model,
            api_key=api_key,
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=max_tokens,
            max_retries=kwargs.get("max_retries", 3),
            timeout=kwargs.get("timeout", 30.0),
            backoff_factor=kwargs.get("backoff_factor", 0.5),
        )
    
    # OpenAI GPT
    elif provider_lower in ["openai", "gpt"]:
        from jentis.llmkit.Openai.base import OpenAILLM
        return OpenAILLM(
            model=model,
            api_key=api_key,
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
            max_retries=kwargs.get("max_retries", 3),
            timeout=kwargs.get("timeout", 30.0),
            backoff_factor=kwargs.get("backoff_factor", 0.5),
        )
    
    # xAI Grok
    elif provider_lower in ["grok", "xai"]:
        from jentis.llmkit.Grok.base import GrokLLM
        return GrokLLM(
            model=model,
            api_key=api_key,
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            max_retries=kwargs.get("max_retries", 3),
            timeout=kwargs.get("timeout", 30.0),
            backoff_factor=kwargs.get("backoff_factor", 0.5),
        )
    
    # Azure OpenAI
    elif provider_lower in ["azure", "microsoft"]:
        from jentis.llmkit.Microsoft.base import AzureLLM
        
        # Azure requires specific parameters
        azure_endpoint = kwargs.get("azure_endpoint")
        if not azure_endpoint:
            raise ValueError("azure_endpoint is required for Azure OpenAI provider")
        
        return AzureLLM(
            deployment_name=kwargs.get("deployment_name", model),
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=kwargs.get("api_version", "2024-08-01-preview"),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            frequency_penalty=kwargs.get("frequency_penalty"),
            presence_penalty=kwargs.get("presence_penalty"),
            max_retries=kwargs.get("max_retries", 3),
            timeout=kwargs.get("timeout", 30.0),
            backoff_factor=kwargs.get("backoff_factor", 0.5),
        )
    
    # Ollama Cloud
    elif provider_lower == "ollama-cloud":
        from jentis.llmkit.Ollamacloud.base import OllamaCloudLLM
        return OllamaCloudLLM(
            model=model,
            api_key=api_key,
            host=kwargs.get("host", kwargs.get("base_url", "https://ollama.com")),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            max_retries=kwargs.get("max_retries", 3),
            timeout=kwargs.get("timeout", 30.0),
            backoff_factor=kwargs.get("backoff_factor", 0.5),
        )
    
    # Ollama Local
    elif provider_lower in ["ollama-local", "ollama"]:
        from jentis.llmkit.Ollamalocal.base import OllamaLocalLLM
        return OllamaLocalLLM(
            model=model,
            host=kwargs.get("host", kwargs.get("base_url", "http://localhost:11434")),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            max_retries=kwargs.get("max_retries", 3),
            timeout=kwargs.get("timeout", 30.0),
            backoff_factor=kwargs.get("backoff_factor", 0.5),
        )
    
    # Google Vertex AI (Model Garden - third-party & first-party models)
    elif provider_lower in ["vertexai", "vertex-ai", "vertex"]:
        from jentis.llmkit.Vertexai.base import VertexAILLM
        return VertexAILLM(
            model=model,
            project_id=kwargs.get("project_id"),
            region=kwargs.get("region", "global"),
            endpoint=kwargs.get("endpoint"),
            access_token=kwargs.get("access_token", api_key),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            max_tokens=kwargs.get("max_tokens"),
            max_retries=kwargs.get("max_retries", 3),
            timeout=kwargs.get("timeout", 60.0),
            backoff_factor=kwargs.get("backoff_factor", 0.5),
        )
    
    else:
        supported = [
            "google/gemini", "anthropic/claude", "openai/gpt", 
            "grok/xai", "azure/microsoft", "ollama-cloud", "ollama-local/ollama",
            "vertexai/vertex-ai/vertex"
        ]
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            f"Supported providers: {', '.join(supported)}"
        )


__all__ = [
    "init_llm",
    "GoogleLLM",
    "AnthropicLLM",
    "OpenAILLM",
    "GrokLLM",
    "AzureLLM",
    "OllamaCloudLLM",
    "OllamaLocalLLM",
    "VertexAILLM",
]
