# OpenAI Provider

OpenAI provides advanced AI models including GPT-4o and GPT-4o-mini, designed for natural language understanding, generation, and reasoning. These models support chat, completion, function calling, and vision capabilities with state-of-the-art performance across diverse tasks.

## Supported Models
| Model Name | Model ID | Input Types | Output Type | Max Input Tokens | Max Output Tokens | Key Capabilities | Best Use Case |
|-----------|----------|-------------|-------------|------------------|-------------------|------------------|---------------|
| o3 | o3 | Text, Image | Text | 200,000 | 100,000 | Deep reasoning, Vision, Tool use | Complex multi-step reasoning |
| o3-mini | o3-mini | Text | Text | 200,000 | 100,000 | Fast reasoning, Cost-effective | Balanced reasoning & throughput |
| o1 | o1 | Text, Image | Text | 200,000 | 100,000 | Reasoning, Vision, Structured output | Scientific & math reasoning |
| o1-mini | o1-mini | Text | Text | 128,000 | 65,536 | Efficient reasoning | Cost-effective STEM tasks |
| GPT-4o | gpt-4o | Text, Image, Audio | Text, Audio | 128,000 | 16,384 | Vision, Function calling, JSON mode, Audio | Complex reasoning, multimodal tasks |
| GPT-4o Mini | gpt-4o-mini | Text, Image | Text | 128,000 | 16,384 | Fast, Efficient, Function calling | Cost-effective general tasks |
| GPT-4 Turbo | gpt-4-turbo | Text, Image | Text | 128,000 | 4,096 | Advanced reasoning, Vision | Complex problem-solving |
| GPT-4 | gpt-4 | Text | Text | 8,192 | 8,192 | Strong reasoning | Complex tasks |

## Installation

```bash
pip install jentis-llmkit openai
```

## Configuration

Set your API key via environment variable or parameter:

```bash
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

## Usage

### Class-based API

#### Standard Response

```python
from jentis.llmkit.Openai import OpenAILLM

llm = OpenAILLM(
    model="gpt-4o",
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
response = llm.generate_response("What is Python?")
print(response)
```

#### Streaming Response

```python
from jentis.llmkit.Openai import OpenAILLM

llm = OpenAILLM(
    model="gpt-4o",
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
from jentis.llmkit.Openai import openai_llm

response = openai_llm(
    prompt="Explain quantum computing",
    model="gpt-4o",
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
#     "model": "gpt-4o",
#     "usage": {
#         "input_tokens": 15,
#         "output_tokens": 250,
#         "total_tokens": 265
#     }
# }
```

#### Streaming Response

```python
from jentis.llmkit.Openai import openai_llm_stream

# Stream text chunks as they're generated
for chunk in openai_llm_stream(
    prompt="Explain machine learning",
    model="gpt-4o",
    api_key="your-key",
    temperature=0.7
):
    print(chunk, end='', flush=True)
```

## Parameters

### `openai_llm()` and `OpenAILLM` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model identifier |
| `api_key` | str | None | API key (falls back to env var) |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `max_tokens` | int | None | Maximum output tokens |
| `frequency_penalty` | float | None | Token frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | float | None | Token presence penalty (-2.0 to 2.0) |
| `max_retries` | int | 3 | Retry attempts (non-streaming only) |
| `timeout` | float | 30.0 | Request timeout (seconds) |
| `backoff_factor` | float | 0.5 | Exponential backoff base (non-streaming only) |

### `openai_llm_stream()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model identifier |
| `api_key` | str | None | API key (falls back to env var) |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `max_tokens` | int | None | Maximum output tokens |
| `frequency_penalty` | float | None | Token frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | float | None | Token presence penalty (-2.0 to 2.0) |
| `timeout` | float | 30.0 | Request timeout (seconds) |

**Note**: Streaming functions do not support `max_retries` or `backoff_factor` parameters.

### Return Values

#### `openai_llm()` Function

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

#### `OpenAILLM.generate_response()` Method

Returns only the generated text as a string (for backward compatibility).

#### `openai_llm_stream()` and `OpenAILLM.generate_response_stream()`

Yields text chunks as strings. Token usage information is not available in streaming mode.

## Error Handling

```python
from jentis.llmkit.Openai import (
    openai_llm,
    OpenAILLM,
    OpenAILLMError,
    OpenAILLMAPIError,
    OpenAILLMImportError,
    OpenAILLMResponseError
)

try:
    # Function-based: Returns dict with metadata
    response = openai_llm(
        prompt="Hello!",
        model="gpt-4o",
        api_key="your-key"
    )
    print(f"Content: {response['content']}")
    print(f"Tokens used: {response['usage']['total_tokens']}")
    
    # Class-based: Returns string only
    llm = OpenAILLM(model="gpt-4o", api_key="your-key")
    text = llm.generate_response("Hello!")
    print(text)
    
    # Streaming response
    for chunk in llm.generate_response_stream("Tell me a story"):
        print(chunk, end='', flush=True)
        
except OpenAILLMImportError as e:
    print(f"SDK not installed: {e}")
except OpenAILLMAPIError as e:
    print(f"API request failed: {e}")
except OpenAILLMResponseError as e:
    print(f"Invalid response: {e}")
except OpenAILLMError as e:
    print(f"General error: {e}")
```

## Features

- **Advanced Reasoning**: GPT-4o models with strong reasoning capabilities
- **Vision Support**: Analyze images with multimodal models
- **Function Calling**: Native tool use and function calling
- **JSON Mode**: Structured output generation
- **Streaming Responses**: Real-time response streaming
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Handling**: Comprehensive exception hierarchy
- **Type Safety**: Full type hints for better IDE support

## Additional Resources

- [Official OpenAI Documentation](https://platform.openai.com/docs/)
- [API Reference](https://platform.openai.com/docs/api-reference/)
- [Model Pricing](https://openai.com/pricing)
- [Rate Limits](https://platform.openai.com/docs/guides/rate-limits)

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

- **Issues**: [GitHub Issues](https://github.com/jentis/jentis/issues)
- **Documentation**: [Full Documentation](https://jentis.readthedocs.io/)
- **Community**: [Discussions](https://github.com/jentis/jentis/discussions)

## Author

Built by **Jentis developers**
