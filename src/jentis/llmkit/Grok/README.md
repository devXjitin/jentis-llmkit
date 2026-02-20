# xAI Grok Provider

xAI's Grok is a state-of-the-art AI model trained on real-time data from the X platform (formerly Twitter). Grok provides witty, conversational responses with access to current events and trending topics. The API is OpenAI-compatible, making integration seamless.

## Supported Models
| Model Name | Model ID | Input Types | Output Type | Max Input Tokens | Max Output Tokens | Key Capabilities | Best Use Case |
|-----------|----------|-------------|-------------|------------------|-------------------|------------------|---------------|
| Grok 3 | grok-3 | Text, Image | Text | 131,072 | 131,072 | Deep reasoning, Real-time data, Vision | Complex analysis with live context |
| Grok 3 Mini | grok-3-mini | Text | Text | 131,072 | 131,072 | Fast reasoning, Real-time data | Quick reasoning, high-throughput |
| Grok 2 | grok-2-latest | Text, Image | Text | 131,072 | 131,072 | Advanced reasoning, Real-time, Vision | Complex tasks with current context |
| Grok Beta | grok-beta | Text | Text | 131,072 | 131,072 | Conversational, Real-time data | General conversations, current events |

## Installation

```bash
pip install jentis-llmkit[openai]
```

## Configuration

Set your API key via environment variable or parameter:

```python
import os
os.environ["XAI_API_KEY"] = "your-api-key"
```

## Usage

### Class-based API

#### Standard Response

```python
from jentis.llmkit.Grok import GrokLLM

llm = GrokLLM(
    model="grok-beta",           # Default model
    api_key="your-key",          # Optional if env var set
    temperature=0.7,              # 0.0-2.0
    top_p=0.9,                    # 0.0-1.0
    max_tokens=1000,
    frequency_penalty=0.0,        # -2.0 to 2.0
    presence_penalty=0.0,         # -2.0 to 2.0
    max_retries=3,
    timeout=30.0,
    backoff_factor=0.5
)

# Returns only the text content (for backward compatibility)
response = llm.generate_response("What's trending today?")
print(response)
```

#### Streaming Response

```python
from jentis.llmkit.Grok import GrokLLM

llm = GrokLLM(
    model="grok-beta",
    api_key="your-key",
    temperature=0.7
)

# Stream the response in real-time
for chunk in llm.generate_response_stream("Tell me about recent AI developments"):
    print(chunk, end='', flush=True)
```

### Function-based API

#### Standard Response

```python
from jentis.llmkit.Grok import grok_llm

response = grok_llm(
    prompt="What's happening in AI today?",
    model="grok-beta",
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
#     "content": "Here's what's trending in AI...",
#     "model": "grok-beta",
#     "usage": {
#         "input_tokens": 20,
#         "output_tokens": 300,
#         "total_tokens": 320
#     }
# }
```

#### Streaming Response

```python
from jentis.llmkit.Grok import grok_llm_stream

# Stream text chunks as they're generated
for chunk in grok_llm_stream(
    prompt="Explain recent tech news",
    model="grok-beta",
    api_key="your-key",
    temperature=0.7
):
    print(chunk, end='', flush=True)
```

## Parameters

### `grok_llm()` and `GrokLLM` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "grok-beta" | Model identifier |
| `api_key` | str | None | API key (falls back to env var) |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `max_tokens` | int | None | Maximum output tokens |
| `frequency_penalty` | float | None | Token frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | float | None | Token presence penalty (-2.0 to 2.0) |
| `max_retries` | int | 3 | Retry attempts (non-streaming only) |
| `timeout` | float | 30.0 | Request timeout (seconds) |
| `backoff_factor` | float | 0.5 | Exponential backoff base (non-streaming only) |

### `grok_llm_stream()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "grok-beta" | Model identifier |
| `api_key` | str | None | API key (falls back to env var) |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `max_tokens` | int | None | Maximum output tokens |
| `frequency_penalty` | float | None | Token frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | float | None | Token presence penalty (-2.0 to 2.0) |
| `timeout` | float | 30.0 | Request timeout (seconds) |

**Note**: Streaming functions do not support `max_retries` or `backoff_factor` parameters.

### Return Values

#### `grok_llm()` Function

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

#### `GrokLLM.generate_response()` Method

Returns only the generated text as a string (for backward compatibility).

#### `grok_llm_stream()` and `GrokLLM.generate_response_stream()`

Yields text chunks as strings. Token usage information is not available in streaming mode.

## Error Handling

```python
from jentis.llmkit.Grok import (
    grok_llm,
    GrokLLM,
    GrokLLMError,
    GrokLLMAPIError,
    GrokLLMImportError,
    GrokLLMResponseError
)

try:
    # Function-based: Returns dict with metadata
    response = grok_llm(
        prompt="What's trending?",
        model="grok-beta",
        api_key="your-key"
    )
    print(f"Content: {response['content']}")
    print(f"Tokens used: {response['usage']['total_tokens']}")
    
    # Class-based: Returns string only
    llm = GrokLLM(model="grok-beta", api_key="your-key")
    text = llm.generate_response("Tell me about AI")
    print(text)
    
    # Streaming response
    for chunk in llm.generate_response_stream("Write a witty explanation"):
        print(chunk, end='', flush=True)
        
except GrokLLMImportError as e:
    print(f"SDK not installed: {e}")
except GrokLLMAPIError as e:
    print(f"API request failed: {e}")
except GrokLLMResponseError as e:
    print(f"Invalid response: {e}")
except GrokLLMError as e:
    print(f"General error: {e}")
```

## Features

- **Real-time Data**: Access to current events via X platform integration
- **Conversational Style**: Witty and engaging responses
- **OpenAI-Compatible**: Uses familiar OpenAI SDK
- **Large Context**: 131K token context window
- **Streaming Responses**: Real-time response streaming
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Handling**: Comprehensive exception hierarchy
- **Type Safety**: Full type hints for better IDE support

## Additional Resources

- [Official xAI Documentation](https://docs.x.ai/)
- [API Console](https://console.x.ai/)
- [Grok Models](https://docs.x.ai/docs/overview#models)

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
