# Anthropic Claude Provider

Anthropic Claude is a family of highly capable AI models designed for safe, accurate, and helpful interactions. Built with Constitutional AI principles, Claude excels at understanding context, following complex instructions, and maintaining coherent conversations. Claude models are available through Anthropic's API and support advanced features like extended context windows, vision capabilities, and tool use.

## Supported Models
| Model Name | Model ID | Input Types | Output Type | Max Input Tokens | Max Output Tokens | Key Capabilities | Best Use Case |
|-----------|----------|-------------|-------------|------------------|-------------------|------------------|---------------|
| Claude Opus 4 | claude-opus-4-20250514 | Text, Image | Text | 200,000 | 32,000 | Deep reasoning, Agentic coding, Vision, Tool use | Most complex multi-step tasks |
| Claude Sonnet 4 | claude-sonnet-4-20250514 | Text, Image | Text | 200,000 | 64,000 | Extended thinking, Vision, Tool use, Structured output | Complex tasks, coding, analysis |
| Claude 3.5 Sonnet | claude-3-5-sonnet-20241022 | Text, Image | Text | 200,000 | 8,192 | Vision, Tool use, Advanced reasoning | General-purpose high-performance |
| Claude 3.5 Haiku | claude-3-5-haiku-20241022 | Text | Text | 200,000 | 8,192 | Fast, Vision, Tool use | Quick responses, high-throughput |
| Claude 3 Opus | claude-3-opus-20240229 | Text, Image | Text | 200,000 | 4,096 | Maximum capability, Vision | Complex reasoning |
| Claude 3 Haiku | claude-3-haiku-20240307 | Text, Image | Text | 200,000 | 4,096 | Fast and efficient | Simple tasks, low latency |

## Installation

```bash
pip install jentis-llmkit[anthropic]
```

## Configuration

Set your API key via environment variable or parameter:

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
```

## Usage

### Class-based API

#### Standard Response

```python
from jentis.llmkit.Anthropic import AnthropicLLM

llm = AnthropicLLM(
    model="claude-3-5-sonnet-20241022",
    api_key="your-key",          # Optional if env var set
    temperature=0.7,              # 0.0-1.0
    top_p=0.9,                    # 0.0-1.0
    top_k=40,                     # >= 1
    max_tokens=1024,              # Required
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
from jentis.llmkit.Anthropic import AnthropicLLM

llm = AnthropicLLM(
    model="claude-3-5-sonnet-20241022",
    api_key="your-key",
    temperature=0.7,
    max_tokens=2048
)

# Stream the response in real-time
for chunk in llm.generate_response_stream("Write a story about AI"):
    print(chunk, end='', flush=True)
```

### Function-based API

#### Standard Response

```python
from jentis.llmkit.Anthropic import anthropic_llm

response = anthropic_llm(
    prompt="Explain quantum computing",
    model="claude-3-5-sonnet-20241022",
    api_key="your-key",
    temperature=0.7,
    max_tokens=1024
)

# Response is a dictionary with metadata
print(response["content"])        # The generated text
print(response["model"])           # Model used
print(response["usage"])           # Token usage information

# Example output:
# {
#     "content": "Quantum computing is...",
#     "model": "claude-3-5-sonnet-20241022",
#     "usage": {
#         "input_tokens": 15,
#         "output_tokens": 250,
#         "total_tokens": 265
#     }
# }
```

#### Streaming Response

```python
from jentis.llmkit.Anthropic import anthropic_llm_stream

# Stream text chunks as they're generated
for chunk in anthropic_llm_stream(
    prompt="Explain machine learning",
    model="claude-3-5-sonnet-20241022",
    api_key="your-key",
    temperature=0.7,
    max_tokens=1024
):
    print(chunk, end='', flush=True)
```

## Parameters

### `anthropic_llm()` and `AnthropicLLM` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model identifier |
| `api_key` | str | None | API key (falls back to env var) |
| `temperature` | float | None | Randomness (0.0-1.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `top_k` | int | None | Top-k sampling |
| `max_tokens` | int | 1024 | Maximum output tokens (required) |
| `max_retries` | int | 3 | Retry attempts (non-streaming only) |
| `timeout` | float | 30.0 | Request timeout (seconds) |
| `backoff_factor` | float | 0.5 | Exponential backoff base (non-streaming only) |

### `anthropic_llm_stream()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model identifier |
| `api_key` | str | None | API key (falls back to env var) |
| `temperature` | float | None | Randomness (0.0-1.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `top_k` | int | None | Top-k sampling |
| `max_tokens` | int | 1024 | Maximum output tokens |
| `timeout` | float | 30.0 | Request timeout (seconds) |

**Note**: Streaming functions do not support `max_retries` or `backoff_factor` parameters.

### Return Values

#### `anthropic_llm()` Function

Returns a dictionary with the following structure:

```python
{
    "content": str,              # The generated text
    "model": str,                # Model identifier used
    "usage": {
        "input_tokens": int,     # Tokens in the input prompt
        "output_tokens": int,    # Tokens in the generated output
        "total_tokens": int      # Total tokens used
    }
}
```

#### `AnthropicLLM.generate_response()` Method

Returns only the generated text as a string (for backward compatibility).

#### `anthropic_llm_stream()` and `AnthropicLLM.generate_response_stream()`

Yields text chunks as strings. Token usage information is not available in streaming mode.

## Error Handling

```python
from jentis.llmkit.Anthropic import (
    anthropic_llm,
    AnthropicLLM,
    AnthropicLLMError,
    AnthropicLLMAPIError,
    AnthropicLLMImportError,
    AnthropicLLMResponseError
)

try:
    # Function-based: Returns dict with metadata
    response = anthropic_llm(
        prompt="Hello!",
        model="claude-3-5-sonnet-20241022",
        api_key="your-key",
        max_tokens=100
    )
    print(f"Content: {response['content']}")
    print(f"Tokens used: {response['usage']['total_tokens']}")
    
    # Class-based: Returns string only
    llm = AnthropicLLM(model="claude-3-5-sonnet-20241022", api_key="your-key")
    text = llm.generate_response("Hello!")
    print(text)
    
    # Streaming response
    for chunk in llm.generate_response_stream("Tell me a story"):
        print(chunk, end='', flush=True)
        
except AnthropicLLMImportError as e:
    print(f"SDK not installed: {e}")
except AnthropicLLMAPIError as e:
    print(f"API request failed: {e}")
except AnthropicLLMResponseError as e:
    print(f"Invalid response: {e}")
except AnthropicLLMError as e:
    print(f"General error: {e}")
```

## Features

- **Constitutional AI**: Built with safety and helpfulness principles
- **Extended Context**: Up to 200K token context window
- **Vision Capabilities**: Analyze images with Claude 3+ models
- **Tool Use**: Native function calling support
- **Streaming Responses**: Real-time response streaming
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Handling**: Comprehensive exception hierarchy
- **Type Safety**: Full type hints for better IDE support

## Additional Resources

- [Official Anthropic Documentation](https://docs.anthropic.com/)
- [Claude API Reference](https://docs.anthropic.com/claude/reference/)
- [Model Pricing](https://www.anthropic.com/pricing)
- [API Limits](https://docs.anthropic.com/claude/reference/rate-limits)

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License - see the `pyproject.toml` file for details.

## Support

For issues, questions, or contributions:

- **Issues**: [GitHub Issues](https://github.com/devXjitin/jentis-llmkit/issues)
- **Documentation**: [Full Documentation](https://github.com/devXjitin/jentis-llmkit)
- **Community**: [Discussions](https://github.com/devXjitin/jentis-llmkit/discussions)

## Author

Built by **Jentis developers**
