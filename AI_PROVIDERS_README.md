# Multi-Provider AI Integration

This document describes the new multi-provider AI integration feature that allows you to test your generated prompts against various AI providers beyond just OpenAI/ChatGPT.

## 🚀 Features

- **Multiple AI Providers**: Support for OpenAI, Anthropic Claude, Google Gemini, and Mistral AI
- **Dynamic Model Selection**: Automatically loads available models for each provider
- **Unified Testing Interface**: Test prompts against any provider using the same interface
- **Comprehensive Results**: View responses, token usage, and provider metadata
- **Rate Limiting Protection**: Built-in delays to prevent API rate limiting

## 🔌 Supported AI Providers

### 1. OpenAI (ChatGPT)
- **API Endpoint**: `https://api.openai.com/v1/chat/completions`
- **Authentication**: Bearer token (API key)
- **Models**: GPT-3.5 Turbo, GPT-4, GPT-4 Turbo, GPT-4o, and more
- **Rate Limits**: Varies by model and account tier

### 2. Anthropic Claude
- **API Endpoint**: `https://api.anthropic.com/v1/messages`
- **Authentication**: x-api-key header
- **Models**: Claude 3 Haiku, Sonnet, Opus, and Claude 3.5 variants
- **Rate Limits**: Varies by model and account tier

### 3. Google Gemini
- **API Endpoint**: `https://generativelanguage.googleapis.com/v1beta`
- **Authentication**: API key as query parameter
- **Models**: Gemini 1.5 Flash, Pro, and legacy models
- **Rate Limits**: Varies by model and account tier

### 4. Mistral AI
- **API Endpoint**: `https://api.mistral.ai/v1/chat/completions`
- **Authentication**: Bearer token (API key)
- **Models**: Mistral Tiny, Small, Medium, Large, and Large Latest
- **Rate Limits**: Varies by model and account tier

## 🛠️ API Endpoints

### Get Available AI Providers
```http
GET /api/ai-providers
```

**Response:**
```json
{
  "providers": ["openai", "anthropic", "google", "mistral"]
}
```

### Get Models for a Provider
```http
GET /api/ai-provider-models/{provider}
```

**Response:**
```json
{
  "provider": "openai",
  "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
}
```

### Test Prompt with AI Provider
```http
POST /api/test-with-ai
Content-Type: application/json

{
  "prompt": "Your test prompt here",
  "provider": "openai",
  "model": "gpt-3.5-turbo",
  "api_key": "your-api-key-here"
}
```

**Response:**
```json
{
  "success": true,
  "prompt": "Your test prompt here",
  "response": "AI provider response...",
  "model": "gpt-3.5-turbo",
  "provider": "OpenAI",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  },
  "metadata": {
    "response_id": "chatcmpl-123..."
  }
}
```

## 🎯 Usage Examples

### Python Example
```python
import requests

# Test a prompt with OpenAI
response = requests.post('http://localhost:5000/api/test-with-ai', json={
    'prompt': 'Hello, how are you?',
    'provider': 'openai',
    'model': 'gpt-3.5-turbo',
    'api_key': 'your-openai-api-key'
})

if response.status_code == 200:
    result = response.json()
    print(f"AI Response: {result['response']}")
    print(f"Provider: {result['provider']}")
    print(f"Model: {result['model']}")
    print(f"Token Usage: {result['usage']}")
```

### JavaScript Example
```javascript
// Test a prompt with Anthropic Claude
const response = await fetch('/api/test-with-ai', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        prompt: 'Hello, how are you?',
        provider: 'anthropic',
        model: 'claude-3-haiku-20240307',
        api_key: 'your-anthropic-api-key'
    })
});

const result = await response.json();
if (result.success) {
    console.log('AI Response:', result.response);
    console.log('Provider:', result.provider);
    console.log('Model:', result.model);
}
```

## 🌐 Web Interface

The web interface provides a user-friendly way to test prompts with different AI providers:

1. **Select AI Provider**: Choose from OpenAI, Anthropic, Google, or Mistral
2. **Choose Model**: Select a specific model from the provider's available options
3. **Enter API Key**: Provide your API key for the selected provider
4. **Test Prompts**: Test generated prompts or enter custom prompts manually
5. **View Results**: See AI responses, token usage, and provider information

## 🔑 API Key Requirements

### OpenAI
- Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- Format: `sk-...`

### Anthropic
- Get your API key from [Anthropic Console](https://console.anthropic.com/)
- Format: `sk-ant-...`

### Google Gemini
- Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Format: `AIza...`

### Mistral
- Get your API key from [Mistral AI Platform](https://console.mistral.ai/)
- Format: `...`

## ⚠️ Important Notes

1. **API Key Security**: API keys are stored locally in the browser and never sent to our servers
2. **Rate Limiting**: The system includes built-in delays to prevent API rate limiting
3. **Cost Management**: Be aware of your API usage costs for each provider
4. **Model Availability**: Some models may not be available in all regions or account tiers
5. **Error Handling**: The system provides detailed error messages for debugging

## 🚀 Adding New Providers

To add a new AI provider:

1. **Create Provider Class**: Extend the `AIProvider` base class
2. **Implement Methods**: Implement `test_prompt`, `get_available_models`, and `get_provider_name`
3. **Register Provider**: Add the provider to the `AIProviderManager`
4. **Update Frontend**: Add provider-specific UI elements if needed

Example:
```python
class NewProvider(AIProvider):
    def test_prompt(self, prompt: str, model: str, **kwargs) -> AIProviderResponse:
        # Implement your provider's API call logic
        pass
    
    def get_available_models(self) -> List[str]:
        return ["model1", "model2"]
    
    def get_provider_name(self) -> str:
        return "New Provider"

# Register in AIProviderManager
ai_provider_manager.register_provider("new_provider", NewProvider)
```

## 🔧 Configuration

The system automatically detects available providers and models. No additional configuration is required beyond providing valid API keys.

## 📊 Monitoring and Logging

All AI provider interactions are logged with:
- Provider name and model
- Prompt length and response length
- API response times
- Error details for debugging

## 🆘 Troubleshooting

### Common Issues

1. **API Key Invalid**: Ensure your API key is correct and has sufficient credits
2. **Model Not Available**: Check if the model is available in your account tier
3. **Rate Limiting**: The system automatically handles rate limiting with delays
4. **Network Errors**: Check your internet connection and firewall settings

### Debug Mode

Enable debug logging to see detailed API interactions:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Considerations

- **Concurrent Testing**: The system tests prompts sequentially to avoid rate limiting
- **Timeout Handling**: 30-second timeout for API calls with configurable options
- **Error Recovery**: Automatic retry logic for transient failures
- **Memory Management**: Efficient handling of large response data

## 🔮 Future Enhancements

Planned features include:
- **Batch Testing**: Test multiple prompts simultaneously
- **Provider Comparison**: Compare responses across different providers
- **Custom Providers**: User-defined provider configurations
- **Advanced Analytics**: Detailed performance metrics and cost analysis
- **Provider Health Monitoring**: Real-time availability checking
