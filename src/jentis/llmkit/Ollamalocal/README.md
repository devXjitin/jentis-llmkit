# Ollama Local Provider

Ollama allows you to run open-source LLMs locally on your own hardware. It provides an OpenAI-compatible API for seamless integration and supports popular models including Llama 2, Mistral, CodeLlama, and more. Perfect for privacy-sensitive applications and offline usage.

## Supported Models

Popular models available for local Ollama (download with `ollama pull <model>`):

| Model Name | Model ID | Input Types | Output Type | Max Input Tokens | Max Output Tokens | Key Capabilities | Best Use Case |
|-----------|----------|-------------|-------------|------------------|-------------------|------------------|---------------|
| Llama 4 Scout | llama4-scout | Text, Image | Text | 512,000 | 256,000 | Multimodal, MoE, Multilingual | General-purpose agents, long context |
| Llama 4 Maverick | llama4-maverick | Text, Image | Text | 256,000 | 128,000 | Multimodal, MoE, Strong reasoning | Complex reasoning tasks |
| Llama 3.3 | llama3.3, llama3.3:70b | Text | Text | 128,000 | 128,000 | General purpose, Instruction tuned | General tasks, conversations |
| Llama 3.1 | llama3.1, llama3.1:70b, llama3.1:405b | Text | Text | 128,000 | 128,000 | Multilingual, Tool use | Large-scale reasoning |
| Mistral Large | mistral-large | Text | Text | 128,000 | 128,000 | Multilingual, Code generation | Enterprise reasoning |
| Mixtral | mixtral:8x7b, mixtral:8x22b | Text | Text | 32,000 | 32,000 | MoE, Strong reasoning | Complex reasoning tasks |
| CodeLlama | codellama, codellama:34b, codellama:70b | Text | Text | 16,384 | 16,384 | Code generation, Infilling | Programming assistance |
| Phi-4 | phi4, phi4:14b | Text | Text | 16,384 | 16,384 | Small, Efficient, Reasoning | Edge deployment, quick tasks |
| Gemma 2 | gemma2, gemma2:27b | Text | Text | 8,192 | 8,192 | Lightweight, Open-source | Research, resource-constrained tasks |
| DeepSeek R1 | deepseek-r1, deepseek-r1:70b | Text | Text | 128,000 | 128,000 | Reasoning, Code generation | Research, complex problem-solving |

## Installation

1. Install Ollama: [https://ollama.ai/download](https://ollama.ai/download)
2. Pull a model: `ollama pull llama2`
3. Install Python package:

```bash
pip install jentis-llmkit[ollama]
```

## Configuration

Ensure Ollama is running locally (default: http://localhost:11434)

```bash
# Start Ollama service (if not auto-started)
ollama serve
```

## Usage

### Class-based API

#### Standard Response

```python
from jentis.llmkit.Ollamalocal import OllamaLocalLLM

llm = OllamaLocalLLM(
    model="llama2",              # Model name (must be pulled locally)
    base_url="http://localhost:11434/v1",  # Default local endpoint
    temperature=0.7,              # 0.0-2.0
    top_p=0.9,                    # 0.0-1.0
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
from jentis.llmkit.Ollamalocal import OllamaLocalLLM

llm = OllamaLocalLLM(
    model="llama2",
    temperature=0.7
)

# Stream the response in real-time
for chunk in llm.generate_response_stream("Write a story about AI"):
    print(chunk, end='', flush=True)
```

### Function-based API

#### Standard Response

```python
from jentis.llmkit.Ollamalocal import ollama_local_llm

response = ollama_local_llm(
    prompt="Explain quantum computing",
    model="llama2",
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
#     "model": "llama2",
#     "usage": {
#         "input_tokens": 15,
#         "output_tokens": 250,
#         "total_tokens": 265
#     }
# }
```

#### Streaming Response

```python
from jentis.llmkit.Ollamalocal import ollama_local_llm_stream

# Stream text chunks as they're generated
for chunk in ollama_local_llm_stream(
    prompt="Explain machine learning",
    model="llama2",
    temperature=0.7
):
    print(chunk, end='', flush=True)
```

## Parameters

### `ollama_local_llm()` and `OllamaLocalLLM` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model identifier (e.g., "llama2", "mistral") |
| `base_url` | str | "http://localhost:11434/v1" | Local Ollama API endpoint |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `max_tokens` | int | None | Maximum output tokens |
| `max_retries` | int | 3 | Retry attempts (non-streaming only) |
| `timeout` | float | 30.0 | Request timeout (seconds) |
| `backoff_factor` | float | 0.5 | Exponential backoff base (non-streaming only) |

### `ollama_local_llm_stream()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model identifier (e.g., "llama2", "mistral") |
| `base_url` | str | "http://localhost:11434/v1" | Local Ollama API endpoint |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `max_tokens` | int | None | Maximum output tokens |
| `timeout` | float | 30.0 | Request timeout (seconds) |

**Note**: Streaming functions do not support `max_retries` or `backoff_factor` parameters. No API key required for local Ollama.

### Return Values

#### `ollama_local_llm()` Function

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

#### `OllamaLocalLLM.generate_response()` Method

Returns only the generated text as a string (for backward compatibility).

#### `ollama_local_llm_stream()` and `OllamaLocalLLM.generate_response_stream()`

Yields text chunks as strings. Token usage information is not available in streaming mode.

## Error Handling

```python
from jentis.llmkit.Ollamalocal import (
    ollama_local_llm,
    OllamaLocalLLM,
    OllamaLocalLLMError,
    OllamaLocalLLMAPIError,
    OllamaLocalLLMImportError,
    OllamaLocalLLMResponseError
)

try:
    # Function-based: Returns dict with metadata
    response = ollama_local_llm(
        prompt="Hello!",
        model="llama2"
    )
    print(f"Content: {response['content']}")
    print(f"Tokens used: {response['usage']['total_tokens']}")
    
    # Class-based: Returns string only
    llm = OllamaLocalLLM(model="llama2")
    text = llm.generate_response("Hello!")
    print(text)
    
    # Streaming response
    for chunk in llm.generate_response_stream("Tell me a story"):
        print(chunk, end='', flush=True)
        
except OllamaLocalLLMImportError as e:
    print(f"SDK not installed: {e}")
except OllamaLocalLLMAPIError as e:
    print(f"API request failed (is Ollama running?): {e}")
except OllamaLocalLLMResponseError as e:
    print(f"Invalid response: {e}")
except OllamaLocalLLMError as e:
    print(f"General error: {e}")
```

## Features

- **Privacy-First**: Run LLMs completely offline on your hardware
- **No API Keys**: No authentication required for local usage
- **Open Source Models**: Access to popular open-source LLMs
- **OpenAI-Compatible**: Uses familiar OpenAI SDK
- **Cost-Free**: No per-token costs after initial setup
- **Customizable**: Download and run any compatible model
- **Streaming Responses**: Real-time response streaming
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Handling**: Comprehensive exception hierarchy
- **Type Safety**: Full type hints for better IDE support

## Model Management

```bash
# List available models
ollama list

# Pull a model
ollama pull llama2

# Remove a model
ollama rm llama2

# Run a model interactively
ollama run llama2
```

## Additional Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Available Models](https://ollama.ai/library)
- [Model Files](https://github.com/ollama/ollama/tree/main/docs)
- [API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)

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
