# Microsoft Azure OpenAI Provider

Azure OpenAI Service provides REST API access to OpenAI's powerful language models including GPT-4, GPT-3.5-Turbo, and Embeddings. The service combines OpenAI's models with Azure's enterprise-grade security, compliance, and regional availability.

## Supported Models

Azure OpenAI supports all OpenAI models through custom deployments. Common deployments include:

| Model Name | Deployment Name | Input Types | Output Type | Max Input Tokens | Max Output Tokens | Key Capabilities | Best Use Case |
|-----------|-----------------|-------------|-------------|------------------|-------------------|------------------|---------------|
| o3 | o3-deployment | Text, Image | Text | 200,000 | 100,000 | Deep reasoning, Vision, Tool use | Complex multi-step reasoning |
| o3-mini | o3-mini-deployment | Text | Text | 200,000 | 100,000 | Fast reasoning, Cost-effective | Balanced reasoning & throughput |
| GPT-4o | gpt-4o-deployment | Text, Image, Audio | Text, Audio | 128,000 | 16,384 | Vision, Function calling, JSON mode | Complex reasoning, multimodal tasks |
| GPT-4o Mini | gpt-4o-mini-deployment | Text, Image | Text | 128,000 | 16,384 | Fast, Efficient, Function calling | Cost-effective general tasks |
| GPT-4 Turbo | gpt-4-turbo-deployment | Text, Image | Text | 128,000 | 4,096 | Advanced reasoning, Vision | Complex problem-solving |
| GPT-4 | gpt-4-deployment | Text | Text | 8,192 | 8,192 | Strong reasoning | Complex tasks |

## Installation

```bash
pip install jentis-llmkit[openai]
```

## Configuration

Set your credentials via environment variables or parameters:

```python
import os
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"
```

## Usage

### Class-based API

#### Standard Response

```python
from jentis.llmkit.Microsoft import AzureLLM

llm = AzureLLM(
    deployment_name="gpt-4-deployment",  # Your deployment name
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-key",          # Optional if env var set
    api_version="2024-02-15-preview",
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
from jentis.llmkit.Microsoft import AzureLLM

llm = AzureLLM(
    deployment_name="gpt-4-deployment",
    azure_endpoint="https://your-resource.openai.azure.com/",
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
from jentis.llmkit.Microsoft import azure_llm

response = azure_llm(
    prompt="Explain quantum computing",
    deployment_name="gpt-4-deployment",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-key",
    temperature=0.7,
    max_tokens=500
)

# Response is a dictionary with metadata
print(response["content"])        # The generated text
print(response["model"])           # Deployment name used
print(response["usage"])           # Token usage information

# Example output:
# {
#     "content": "Quantum computing is...",
#     "model": "gpt-4-deployment",
#     "usage": {
#         "input_tokens": 15,
#         "output_tokens": 250,
#         "total_tokens": 265
#     }
# }
```

#### Streaming Response

```python
from jentis.llmkit.Microsoft import azure_llm_stream

# Stream text chunks as they're generated
for chunk in azure_llm_stream(
    prompt="Explain machine learning",
    deployment_name="gpt-4-deployment",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-key",
    temperature=0.7
):
    print(chunk, end='', flush=True)
```

## Parameters

### `azure_llm()` and `AzureLLM` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deployment_name` | str | Required | Your Azure deployment name |
| `azure_endpoint` | str | None | Azure endpoint URL (falls back to env var) |
| `api_key` | str | None | API key (falls back to env var) |
| `api_version` | str | "2024-02-15-preview" | Azure API version |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `max_tokens` | int | None | Maximum output tokens |
| `frequency_penalty` | float | None | Token frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | float | None | Token presence penalty (-2.0 to 2.0) |
| `max_retries` | int | 3 | Retry attempts (non-streaming only) |
| `timeout` | float | 30.0 | Request timeout (seconds) |
| `backoff_factor` | float | 0.5 | Exponential backoff base (non-streaming only) |

### `azure_llm_stream()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deployment_name` | str | Required | Your Azure deployment name |
| `azure_endpoint` | str | None | Azure endpoint URL (falls back to env var) |
| `api_key` | str | None | API key (falls back to env var) |
| `api_version` | str | "2024-02-15-preview" | Azure API version |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `max_tokens` | int | None | Maximum output tokens |
| `frequency_penalty` | float | None | Token frequency penalty (-2.0 to 2.0) |
| `presence_penalty` | float | None | Token presence penalty (-2.0 to 2.0) |
| `timeout` | float | 30.0 | Request timeout (seconds) |

**Note**: Streaming functions do not support `max_retries` or `backoff_factor` parameters.

### Return Values

#### `azure_llm()` Function

Returns a dictionary with the following structure:

```python
{
    "content": str,              # The generated text
    "model": str,                # Deployment name used
    "usage": {
        "input_tokens": int,     # Tokens in the input prompt
        "output_tokens": int,    # Tokens in the generated output
        "total_tokens": int      # Total tokens used
    }
}
```

#### `AzureLLM.generate_response()` Method

Returns only the generated text as a string (for backward compatibility).

#### `azure_llm_stream()` and `AzureLLM.generate_response_stream()`

Yields text chunks as strings. Token usage information is not available in streaming mode.

## Error Handling

```python
from jentis.llmkit.Microsoft import (
    azure_llm,
    AzureLLM,
    AzureLLMError,
    AzureLLMAPIError,
    AzureLLMImportError,
    AzureLLMResponseError
)

try:
    # Function-based: Returns dict with metadata
    response = azure_llm(
        prompt="Hello!",
        deployment_name="gpt-4-deployment",
        azure_endpoint="https://your-resource.openai.azure.com/",
        api_key="your-key"
    )
    print(f"Content: {response['content']}")
    print(f"Tokens used: {response['usage']['total_tokens']}")
    
    # Class-based: Returns string only
    llm = AzureLLM(
        deployment_name="gpt-4-deployment",
        azure_endpoint="https://your-resource.openai.azure.com/",
        api_key="your-key"
    )
    text = llm.generate_response("Hello!")
    print(text)
    
    # Streaming response
    for chunk in llm.generate_response_stream("Tell me a story"):
        print(chunk, end='', flush=True)
        
except AzureLLMImportError as e:
    print(f"SDK not installed: {e}")
except AzureLLMAPIError as e:
    print(f"API request failed: {e}")
except AzureLLMResponseError as e:
    print(f"Invalid response: {e}")
except AzureLLMError as e:
    print(f"General error: {e}")
```

## Features

- **Enterprise Security**: Azure's enterprise-grade security and compliance
- **Regional Availability**: Deploy in your preferred Azure region
- **Private Networking**: VNet and Private Link support
- **Advanced Reasoning**: GPT-4 models with strong reasoning capabilities
- **Vision Support**: Analyze images with multimodal models
- **Function Calling**: Native tool use and function calling
- **Streaming Responses**: Real-time response streaming
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Handling**: Comprehensive exception hierarchy
- **Type Safety**: Full type hints for better IDE support

## Additional Resources

- [Official Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- [API Reference](https://learn.microsoft.com/azure/ai-services/openai/reference)
- [Pricing](https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/)
- [Quotas and Limits](https://learn.microsoft.com/azure/ai-services/openai/quotas-limits)

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
