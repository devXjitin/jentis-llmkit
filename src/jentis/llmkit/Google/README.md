# Google Gemini Provider

Google Gemini is a family of advanced AI models developed by [Google](https://ai.google.dev/gemini-api/docs/models) to handle text, images, audio, video, and code in a single system. It is designed to be natively multimodal, allowing more natural and powerful reasoning across different types of data. Gemini powers a wide range of applications, from chat-based assistants to developer APIs. It focuses on high performance, scalability, and seamless integration with Google’s ecosystem.

## Supported Models
| Model Name | Model ID | Input Types | Output Type | Max Input Tokens | Max Output Tokens | Key Capabilities | Best Use Case |
|-----------|----------|-------------|-------------|------------------|-------------------|------------------|---------------|
| Gemini 3 Pro (Preview) | gemini-3-pro-preview | Text, Audio, Video, PDF | Text | 1,048,576 | 65,536 | Thinking, Function Calling, Structured Output, Batch API, Code & File Search | Advanced reasoning, long-context agents |
| Gemini 3 Flash (Preview) | gemini-3-flash-preview | Text, Audio, Video, PDF | Text | 1,048,576 | 65,536 | Thinking, Function Calling, Fast responses | Low-latency intelligent agents |
| Gemini 2.5 Flash | gemini-2.5-flash | Text, Audio, Video | Text | 1,048,576 | 65,536 | Thinking, Structured Output, Tool Use | General-purpose agent workflows |
| Gemini 2.5 Flash (Preview) | gemini-2.5-flash-preview-09-2025 | Text, Audio, Video | Text | 1,048,576 | 65,536 | Same as Flash with latest updates | Testing upcoming features |
| Gemini 2.5 Flash-Lite | gemini-2.5-flash-lite | Text, Audio, Video, PDF | Text | 1,048,576 | 65,536 | Ultra-fast, cost-efficient | High-throughput pipelines |
| Gemini 2.5 Flash-Lite (Preview) | gemini-2.5-flash-lite-preview-09-2025 | Text, Audio, Video, PDF | Text | 1,048,576 | 65,536 | Experimental optimizations | Preview benchmarking |

## Installation

```bash
pip install jentis
```

## Configuration

Set your API key via environment variable or parameter:

```bash
import os
os.environ["GOOGLE_API_KEY"] = "your-api-key"
```

## Usage

### Class-based API

#### Standard Response

```python
from jentis.llmkit.Google import GoogleLLM

llm = GoogleLLM(
    model="gemini-3-flash-preview",
    api_key="your-key",          # Optional if env var set
    temperature=0.7,              # 0.0-2.0
    top_p=0.9,                    # 0.0-1.0
    top_k=40,                     # >= 1
    max_tokens=1000,
    max_retries=3,
    timeout=30.0,
    backoff_factor=0.5
)

# Returns only the text content (for backward compatibility)
response = llm.generate_response("What is Python?")
print(response)
```

#### Streaming Response

```python
from jentis.llmkit.Google import GoogleLLM

llm = GoogleLLM(
    model="gemini-3-flash-preview",
    api_key="your-key",
    temperature=0.7
)

# Stream the response in real-time
for chunk in llm.generate_response_stream("Write a story about AI"):
    print(chunk, end='', flush=True)
```

### Function-based API

#### Standard Response

```python
from jentis.llmkit.Google import google_llm

response = google_llm(
    prompt="Explain quantum computing",
    model="gemini-3-flash-preview",
    api_key="your-key",
    temperature=0.7,
    max_tokens=500
)

# Response is a dictionary with metadata
print(response["content"])        # The generated text
print(response["model"])           # Model used
print(response["usage"])           # Token usage information

# Example output:
# {
#     "content": "Quantum computing is...",
#     "model": "gemini-3-flash-preview",
#     "usage": {
#         "input_tokens": 15,
#         "output_tokens": 250,
#         "total_tokens": 265
#     }
# }
```

#### Streaming Response

```python
from jentis.llmkit.Google import google_llm_stream

# Stream text chunks as they're generated
for chunk in google_llm_stream(
    prompt="Explain machine learning",
    model="gemini-3-flash-preview",
    api_key="your-key",
    temperature=0.7
):
    print(chunk, end='', flush=True)
```

## Parameters

### `google_llm()` and `GoogleLLM` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model identifier |
| `api_key` | str | None | API key (falls back to env var) |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `top_k` | int | None | Top-k sampling |
| `max_tokens` | int | None | Maximum output tokens |
| `max_retries` | int | 3 | Retry attempts (non-streaming only) |
| `timeout` | float | 30.0 | Request timeout (seconds) |
| `backoff_factor` | float | 0.5 | Exponential backoff base (non-streaming only) |

### `google_llm_stream()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model identifier |
| `api_key` | str | None | API key (falls back to env var) |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `top_k` | int | None | Top-k sampling |
| `max_tokens` | int | None | Maximum output tokens |
| `timeout` | float | 30.0 | Request timeout (seconds) |

**Note**: Streaming functions do not support `max_retries` or `backoff_factor` parameters.

### Return Values

#### `google_llm()` Function

Returns a dictionary with the following structure:

```python
{
    "content": str,              # The generated text
    "model": str,                # Model identifier used
    "usage": {
        "input_tokens": int,     # Tokens in the input prompt (may be None)
        "output_tokens": int,    # Tokens in the generated output (may be None)
        "total_tokens": int      # Total tokens used (may be None)
    }
}
```

#### `GoogleLLM.generate_response()` Method

Returns only the generated text as a string (for backward compatibility).

#### `google_llm_stream()` and `GoogleLLM.generate_response_stream()`

Yields text chunks as strings. Token usage information is not available in streaming mode.

## Error Handling

```python
from jentis.llmkit.Google import (
    google_llm,
    GoogleLLM,
    GoogleLLMError,
    GoogleLLMAPIError,
    GoogleLLMImportError,
    GoogleLLMResponseError
)

try:
    # Function-based: Returns dict with metadata
    response = google_llm(
        prompt="Hello!",
        model="gemini-3-flash-preview",
        api_key="your-key"
    )
    print(f"Content: {response['content']}")
    print(f"Tokens used: {response['usage']['total_tokens']}")
    
    # Class-based: Returns string only
    llm = GoogleLLM(model="gemini-3-flash-preview", api_key="your-key")
    text = llm.generate_response("Hello!")
    print(text)
    
    # Streaming response
    for chunk in llm.generate_response_stream("Tell me a story"):
        print(chunk, end='', flush=True)
        
except GoogleLLMImportError as e:
    print(f"SDK not installed: {e}")
except GoogleLLMAPIError as e:
    print(f"API request failed: {e}")
except GoogleLLMResponseError as e:
    print(f"Invalid response: {e}")
except GoogleLLMError as e:
    print(f"General error: {e}")
```

## Features

- **Multimodal Support**: Handle text, audio, video, and PDF inputs
- **Long Context Windows**: Up to 1M+ tokens for complex reasoning tasks
- **Advanced Thinking**: Built-in reasoning capabilities with Gemini 3 models
- **Structured Output**: JSON mode for reliable structured responses
- **Function Calling**: Native tool use and function calling support
- **Streaming Responses**: Real-time response streaming with `google_llm_stream()` and `generate_response_stream()`
- **Retry Logic**: Automatic retry with exponential backoff for non-streaming requests
- **Error Handling**: Comprehensive exception hierarchy
- **Type Safety**: Full type hints for better IDE support
- **Silent Logging**: Suppressed verbose gRPC and library logs for cleaner output

## Additional Resources

- [Official Google AI Documentation](https://ai.google.dev/)
- [Gemini API Reference](https://ai.google.dev/gemini-api/docs)
- [Model Pricing](https://ai.google.dev/pricing)
- [API Limits & Quotas](https://ai.google.dev/gemini-api/docs/quota)

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](../../../LICENSE) file for details.

## Support

For issues, questions, or contributions:

- **Issues**: [GitHub Issues](https://github.com/devXjitin/jentis-llmkit/issues)
- **Documentation**: [Full Documentation](https://github.com/devXjitin/jentis-llmkit)
- **Community**: [Discussions](https://github.com/devXjitin/jentis-llmkit/discussions)

## Author

Built by **Jentis developers**
