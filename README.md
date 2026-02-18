# Jentis LLM Kit

A unified Python interface for multiple Large Language Model (LLM) providers. Access Google Gemini, Anthropic Claude, OpenAI GPT, xAI Grok, Azure OpenAI, and Ollama through a single, consistent API.

## Features

- 🔄 **Unified Interface**: One API for all LLM providers
- 🚀 **Easy to Use**: Simple `init_llm()` function to get started
- 📡 **Streaming Support**: Real-time response streaming for all providers
- 📊 **Token Tracking**: Consistent token usage reporting across providers
- 🔧 **Flexible Configuration**: Provider-specific parameters when needed
- 🛡️ **Error Handling**: Comprehensive exception hierarchy for debugging

## Supported Providers

| Provider | Aliases | Models |
|----------|---------|--------|
| Google Gemini | `google`, `gemini` | gemini-2.0-flash-exp, gemini-1.5-pro, etc. |
| Anthropic Claude | `anthropic`, `claude` | claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022 |
| OpenAI | `openai`, `gpt` | gpt-4o, gpt-4o-mini, gpt-4-turbo |
| xAI Grok | `grok`, `xai` | grok-2-latest, grok-2-vision-latest |
| Azure OpenAI | `azure`, `microsoft` | Your deployment names |
| Ollama Cloud | `ollama-cloud` | llama2, mistral, codellama, etc. |
| Ollama Local | `ollama`, `ollama-local` | Any locally installed model |
| Vertex AI | `vertexai`, `vertex-ai`, `vertex` | Any Vertex AI Model Garden model |

## Installation

```bash
# Install the base package
pip install Jentis

# Install provider-specific dependencies
pip install google-generativeai  # For Google Gemini
pip install anthropic            # For Anthropic Claude
pip install openai               # For OpenAI, Grok, Azure
pip install ollama               # For Ollama (Cloud & Local)
# Vertex AI requires no pip packages — only gcloud CLI
```

## Quick Start

### Basic Usage

```python
from Jentis.llmkit import init_llm

# Initialize OpenAI GPT-4 (requires OpenAI API key)
llm = init_llm(
    provider="openai",
    model="gpt-4o",
    api_key="sk-proj-xxxxxxxxxxxx"  # Your OpenAI API key
)

# Generate a response
response = llm.generate_response("What is Python?")
print(response)
```

### Streaming Responses

```python
from Jentis.llmkit import init_llm

# Each provider requires its own API key
llm = init_llm(
    provider="openai",
    model="gpt-4o",
    api_key="sk-proj-xxxxxxxxxxxx"  # OpenAI-specific key
)

# Stream the response
for chunk in llm.generate_response_stream("Write a short story about AI"):
    print(chunk, end='', flush=True)
```

## Provider Examples

### Google Gemini

```python
from Jentis.llmkit import init_llm

# Requires Google AI Studio API key
llm = init_llm(
    provider="google",
    model="gemini-2.0-flash-exp",
    api_key="AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxx",  # Google API key
    temperature=0.7,
    max_tokens=1024
)

response = llm.generate_response("Explain quantum computing")
print(response)
```

### Anthropic Claude

```python
from Jentis.llmkit import init_llm

# Requires Anthropic API key
llm = init_llm(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    api_key="sk-ant-api03-xxxxxxxxxxxxxxxxx",  # Anthropic API key
    max_tokens=2048,
    temperature=0.8
)

response = llm.generate_response("Write a haiku about programming")
print(response)
```

### OpenAI GPT

```python
from Jentis.llmkit import init_llm

# Requires OpenAI API key
llm = init_llm(
    provider="openai",
    model="gpt-4o",
    api_key="sk-proj-xxxxxxxxxxxxxxxxxxxx",  # OpenAI API key
    temperature=0.9,
    max_tokens=1500,
    frequency_penalty=0.5,
    presence_penalty=0.3
)

response = llm.generate_response("Design a simple REST API")
print(response)
```

### xAI Grok

```python
from Jentis.llmkit import init_llm

# Requires xAI API key
llm = init_llm(
    provider="grok",
    model="grok-2-latest",
    api_key="xai-xxxxxxxxxxxxxxxxxxxxxxxx",  # xAI API key
    temperature=0.7
)

response = llm.generate_response("What's happening in tech?")
print(response)
```

### Azure OpenAI

```python
from Jentis.llmkit import init_llm

# Requires Azure OpenAI API key and endpoint
llm = init_llm(
    provider="azure",
    model="gpt-4o",
    api_key="a1b2c3d4e5f6xxxxxxxxxxxx",  # Azure API key
    azure_endpoint="https://your-resource.openai.azure.com/",
    deployment_name="gpt-4o-deployment",
    api_version="2024-08-01-preview",
    temperature=0.7
)

response = llm.generate_response("Explain Azure services")
print(response)
```

### Ollama Local

```python
from Jentis.llmkit import init_llm

# No API key needed for local Ollama
llm = init_llm(
    provider="ollama",
    model="llama2",
    temperature=0.7
)

response = llm.generate_response("Hello, Ollama!")
print(response)
```

### Ollama Cloud

```python
from Jentis.llmkit import init_llm

# Requires Ollama Cloud API key
llm = init_llm(
    provider="ollama-cloud",
    model="llama2",
    api_key="ollama_xxxxxxxxxxxxxxxx",  # Ollama Cloud API key
    host="https://ollama.com"
)

response = llm.generate_response("Explain machine learning")
print(response)
```

### Vertex AI (Model Garden)

```python
from Jentis.llmkit import init_llm

# Uses gcloud CLI for authentication (no API key needed)
llm = init_llm(
    provider="vertexai",
    model="moonshotai/kimi-k2-thinking-maas",
    project_id="gen-lang-client-0152852093",
    region="global",
    temperature=0.6,
    max_tokens=8192
)

response = llm.generate_response("What is quantum computing?")
print(response)
```

## Advanced Usage

### Using Function-Based API with Metadata

If you need detailed metadata (token usage, model info), import the provider-specific functions:

```python
from Jentis.llmkit.Openai import openai_llm

result = openai_llm(
    prompt="What is AI?",
    model="gpt-4o",
    api_key="sk-proj-xxxxxxxxxxxxxxxxxxxx",  # Your OpenAI API key
    temperature=0.7
)

print(f"Content: {result['content']}")
print(f"Model: {result['model']}")
print(f"Input tokens: {result['usage']['input_tokens']}")
print(f"Output tokens: {result['usage']['output_tokens']}")
print(f"Total tokens: {result['usage']['total_tokens']}")
```

**Other Providers:**

```python
# Google Gemini
from Jentis.llmkit.Google import google_llm
result = google_llm(prompt="...", model="gemini-2.0-flash-exp", api_key="...")

# Anthropic Claude
from Jentis.llmkit.Anthropic import anthropic_llm
result = anthropic_llm(prompt="...", model="claude-3-5-sonnet-20241022", api_key="...", max_tokens=1024)

# Grok
from Jentis.llmkit.Grok import grok_llm
result = grok_llm(prompt="...", model="grok-2-latest", api_key="...")

# Azure OpenAI
from Jentis.llmkit.Microsoft import azure_llm
result = azure_llm(prompt="...", deployment_name="gpt-4o", azure_endpoint="...", api_key="...")

# Ollama Cloud
from Jentis.llmkit.Ollamacloud import ollama_cloud_llm
result = ollama_cloud_llm(prompt="...", model="llama2", api_key="...")

# Ollama Local
from Jentis.llmkit.Ollamalocal import ollama_local_llm
result = ollama_local_llm(prompt="...", model="llama2")

# Vertex AI
from Jentis.llmkit.Vertexai import vertexai_llm
result = vertexai_llm(prompt="...", model="google/gemini-2.0-flash", project_id="my-project")
```

**Streaming with Functions:**

```python
from Jentis.llmkit.Openai import openai_llm_stream

for chunk in openai_llm_stream(
    prompt="Write a story",
    model="gpt-4o",
    api_key="sk-proj-xxxxxxxxxxxxxxxxxxxx"
):
    print(chunk, end='', flush=True)
```

### Custom Configuration

```python
from Jentis.llmkit import init_llm

llm = init_llm(
    provider="openai",
    model="gpt-4o",
    api_key="sk-proj-xxxxxxxxxxxxxxxxxxxx",  # Your OpenAI API key
    temperature=0.8,
    top_p=0.9,
    max_tokens=2000,
    max_retries=5,
    timeout=60.0,
    backoff_factor=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.3
)
```

## Parameters

### Common Parameters

All providers support these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | **Required** | Provider name or alias |
| `model` | str | **Required** | Model identifier |
| `api_key` | str | None | API key (env var if not provided) |
| `temperature` | float | None | Randomness (0.0-2.0) |
| `top_p` | float | None | Nucleus sampling (0.0-1.0) |
| `max_tokens` | int | None | Maximum tokens to generate |
| `timeout` | float | 30.0 | Request timeout (seconds) |
| `max_retries` | int | 3 | Retry attempts |
| `backoff_factor` | float | 0.5 | Exponential backoff factor |

### Provider-Specific Parameters

**OpenAI & Grok:**
- `frequency_penalty`: Penalty for token frequency (0.0-2.0)
- `presence_penalty`: Penalty for token presence (0.0-2.0)

**Azure OpenAI:**
- `azure_endpoint`: Azure endpoint URL (**Required**)
- `deployment_name`: Deployment name (defaults to model)
- `api_version`: API version (default: "2024-08-01-preview")

**Ollama (Cloud & Local):**
- `host`: Host URL (Cloud: "https://ollama.com", Local: "http://localhost:11434")

## Environment Variables

**Each provider uses its own environment variable for API keys.** Set them to avoid hardcoding:

```bash
# Google Gemini
export GOOGLE_API_KEY="AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxx"

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-api03-xxxxxxxxxxxxxxxxx"

# OpenAI
export OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxxxxxxxxx"

# xAI Grok
export XAI_API_KEY="xai-xxxxxxxxxxxxxxxxxxxxxxxx"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="a1b2c3d4e5f6xxxxxxxxxxxx"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# Ollama Cloud
export OLLAMA_API_KEY="ollama_xxxxxxxxxxxxxxxx"

# Vertex AI (uses gcloud auth, or set token explicitly)
export VERTEX_AI_ACCESS_TOKEN="ya29.xxxxx..."
export VERTEX_AI_PROJECT_ID="your-project-id"
```

Then initialize without api_key parameter:

```python
from Jentis.llmkit import init_llm

# OpenAI - reads from OPENAI_API_KEY environment variable
llm = init_llm(provider="openai", model="gpt-4o")

# Google - reads from GOOGLE_API_KEY environment variable
llm = init_llm(provider="google", model="gemini-2.0-flash-exp")

# Anthropic - reads from ANTHROPIC_API_KEY environment variable
llm = init_llm(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Vertex AI - reads from VERTEX_AI_PROJECT_ID, authenticates via gcloud
llm = init_llm(provider="vertexai", model="google/gemini-2.0-flash")
```

## Methods

All initialized LLM instances have two methods:

### `generate_response(prompt: str) -> str`

Generate a complete response.

```python
response = llm.generate_response("Your prompt here")
print(response)  # String output
```

### `generate_response_stream(prompt: str) -> Generator`

Stream the response in real-time.

```python
for chunk in llm.generate_response_stream("Your prompt here"):
    print(chunk, end='', flush=True)
```

## Error Handling

```python
from Jentis.llmkit import init_llm

try:
    llm = init_llm(
        provider="openai",
        model="gpt-4o",
        api_key="sk-invalid-key-xxxxxxxxxx"  # Wrong API key
    )
    response = llm.generate_response("Test")
except ValueError as e:
    print(f"Invalid configuration: {e}")
except Exception as e:
    print(f"API Error: {e}")
```

Each provider has its own exception hierarchy for detailed error handling. Import from provider modules:

```python
from Jentis.llmkit.Openai import (
    OpenAILLMError,
    OpenAILLMAPIError,
    OpenAILLMImportError,
    OpenAILLMResponseError
)

try:
    from Jentis.llmkit.Openai import openai_llm
    result = openai_llm(prompt="Test", model="gpt-4o", api_key="invalid")
except OpenAILLMAPIError as e:
    print(f"API Error: {e}")
except OpenAILLMError as e:
    print(f"General Error: {e}")
```

## Complete Example

```python
from Jentis.llmkit import init_llm

def chat_with_llm(provider_name: str, user_message: str):
    """Simple chat function supporting multiple providers."""
    try:
        # Initialize LLM
        llm = init_llm(
            provider=provider_name,
            model="gpt-4o" if provider_name == "openai" else "llama2",
            api_key=None,  # Uses environment variables
            temperature=0.7,
            max_tokens=1024
        )
        
        # Stream response
        print(f"\n{provider_name.upper()} Response:\n")
        for chunk in llm.generate_response_stream(user_message):
            print(chunk, end='', flush=True)
        print("\n")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Error: {e}")

# Use different providers
chat_with_llm("openai", "What is machine learning?")
chat_with_llm("anthropic", "Explain neural networks")
chat_with_llm("ollama", "What is Python?")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the terms of the [LICENSE](../../LICENSE) file.

## Support

- **Issues**: [GitHub Issues](https://github.com/jentisgit/J.E.N.T.I.S/issues)
- **Documentation**: [Project Docs](https://github.com/jentisgit/J.E.N.T.I.S)
- **Community**: [Discussions](https://github.com/jentisgit/J.E.N.T.I.S/discussions)

## Author

Built with care by the **J.E.N.T.I.S** team.
