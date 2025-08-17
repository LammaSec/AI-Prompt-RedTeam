# API Documentation

This document provides comprehensive documentation for all API endpoints in the Prompt Generator for Red Team application.

## Base URL
```
http://localhost:5000
```

## Authentication
Most endpoints do not require authentication. However, when testing with AI providers, you'll need to provide API keys in the request body.

## Response Format
All API responses follow a consistent format:

**Success Response:**
```json
{
    "success": true,
    "data": {...}
}
```

**Error Response:**
```json
{
    "success": false,
    "error": "Error message"
}
```

## Endpoints

### 1. Health Check

**GET** `/api/health`

Check the health status of the application and Ollama service.

**Response:**
```json
{
    "status": "healthy",
    "ollama_running": true,
    "message": "Prompt Generator for Red Team by LammaSec is running"
}
```

**Example:**
```bash
curl http://localhost:5000/api/health
```

### 2. Generate Prompts

**POST** `/api/generate`

Generate malicious prompts using the specified strategy and parameters.

**Request Body:**
```json
{
    "strategy": ["prompt_injection", "jailbreaking"],
    "count": 5,
    "complexity": "high",
    "target_category": "prompt_injection",
    "custom_instructions": "Focus on SQL injection attempts",
    "evasion_techniques": ["character_substitution", "unicode_manipulation"],
    "evasion_intensity": 0.8
}
```

**Parameters:**
- `strategy` (array/string): Generation strategies to use
- `count` (integer): Number of prompts to generate (1-50)
- `complexity` (string): Complexity level ("low", "medium", "high")
- `target_category` (string, optional): Specific attack category
- `custom_instructions` (string, optional): Custom instructions for generation
- `evasion_techniques` (array, optional): Evasion techniques to apply
- `evasion_intensity` (float, optional): Intensity of evasion (0.0-1.0)

**Response:**
```json
{
    "success": true,
    "prompts": [
        {
            "id": "prompt_001",
            "content": "Generated prompt content...",
            "strategy": "prompt_injection",
            "complexity": "high",
            "risk_level": "high",
            "evasion_applied": true,
            "metadata": {...}
        }
    ],
    "count": 5
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": ["jailbreaking"],
    "count": 3,
    "complexity": "medium"
  }'
```

### 3. Get Available Strategies

**GET** `/api/strategies`

Get a list of all available generation strategies.

**Response:**
```json
{
    "strategies": [
        "prompt_injection",
        "jailbreaking",
        "roleplay_exploit",
        "harm_generation",
        "privacy_violation",
        "deception",
        "evasion",
        "multi_turn",
        "random"
    ]
}
```

### 4. Get Available Categories

**GET** `/api/categories`

Get a list of all available attack categories.

**Response:**
```json
{
    "categories": [
        "prompt_injection",
        "jailbreaking",
        "roleplay_exploit",
        "harm_generation",
        "privacy_violation",
        "deception",
        "evasion",
        "multi_turn"
    ]
}
```

### 5. Get Evasion Techniques

**GET** `/api/evasion-techniques`

Get a list of all available evasion techniques.

**Response:**
```json
{
    "evasion_techniques": [
        "character_substitution",
        "unicode_manipulation",
        "case_manipulation",
        "spacing_techniques",
        "encoding_methods",
        "context_manipulation",
        "semantic_substitution",
        "structural_changes"
    ]
}
```

### 6. Get Evasion Technique Info

**GET** `/api/evasion-technique-info/{technique}`

Get detailed information about a specific evasion technique.

**Response:**
```json
{
    "name": "character_substitution",
    "description": "Replace characters with similar-looking alternatives",
    "examples": [
        "Replace 'a' with 'а' (Cyrillic)",
        "Replace 'o' with 'о' (Cyrillic)"
    ],
    "effectiveness": 0.7,
    "detection_risk": "medium"
}
```

### 7. Apply Evasion Techniques

**POST** `/api/apply-evasion`

Apply evasion techniques to a single prompt.

**Request Body:**
```json
{
    "prompt": "Your original prompt here",
    "techniques": ["character_substitution", "unicode_manipulation"],
    "intensity": 0.7
}
```

**Response:**
```json
{
    "success": true,
    "result": {
        "original_prompt": "Your original prompt here",
        "modified_prompt": "Yоur оriginal prоmpt here",
        "techniques_applied": ["character_substitution", "unicode_manipulation"],
        "intensity": 0.7,
        "modification_count": 3
    }
}
```

### 8. Get AI Providers

**GET** `/api/ai-providers`

Get a list of available AI providers for testing.

**Response:**
```json
{
    "providers": ["openai", "anthropic", "google", "mistral"]
}
```

### 9. Get Provider Models

**GET** `/api/ai-provider-models/{provider}`

Get available models for a specific AI provider.

**Response:**
```json
{
    "provider": "openai",
    "models": [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o"
    ]
}
```

### 10. Test with AI Provider

**POST** `/api/test-with-ai`

Test a prompt with any supported AI provider.

**Request Body:**
```json
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

### 11. Start Automated Conversation

**POST** `/api/start-conversation`

Start an automated multi-turn conversation with an AI provider.

**Request Body:**
```json
{
    "provider": "openai",
    "api_key": "your-api-key-here",
    "model": "gpt-3.5-turbo",
    "max_intervals": 5,
    "strategy": ["jailbreaking", "roleplay_exploit"],
    "evasion_techniques": ["character_substitution"],
    "initial_prompt": "Your initial prompt here"
}
```

**Parameters:**
- `provider` (string): AI provider to use
- `api_key` (string): API key for the provider
- `model` (string): Model to use
- `max_intervals` (integer): Maximum conversation turns (1-10)
- `strategy` (array): Generation strategies for follow-up prompts
- `evasion_techniques` (array, optional): Evasion techniques to apply
- `initial_prompt` (string): Starting prompt for the conversation

**Response:**
```json
{
    "success": true,
    "messages": [
        {
            "role": "user",
            "content": "Initial prompt",
            "timestamp": "2024-01-01T12:00:00Z",
            "metadata": {...}
        },
        {
            "role": "assistant",
            "content": "AI response",
            "timestamp": "2024-01-01T12:00:01Z",
            "metadata": {...}
        }
    ],
    "analysis": {
        "bypass_successful": true,
        "bypass_score": 0.8,
        "risk_assessment": "high"
    },
    "bypass_successful": true,
    "bypass_score": 0.8,
    "total_intervals": 5
}
```

### 12. Stream Conversation Progress

**GET** `/api/conversation-progress`

Stream real-time updates during conversation execution using Server-Sent Events (SSE).

**Response Format:**
```
data: {"type": "conversation_start", "message": "Starting conversation...", "timestamp": "2024-01-01T12:00:00Z"}

data: {"type": "prompt_generated", "message": "Generated follow-up prompt", "data": {"content": "..."}, "timestamp": "2024-01-01T12:00:01Z"}

data: {"type": "ai_response", "message": "Received AI response", "data": {"content": "..."}, "timestamp": "2024-01-01T12:00:02Z"}

data: {"type": "conversation_complete", "message": "Conversation completed", "data": {"analysis": {...}}, "timestamp": "2024-01-01T12:00:05Z"}
```

**Event Types:**
- `conversation_start`: Conversation initialization
- `prompt_generated`: New prompt generated
- `ai_response`: AI provider response received
- `evasion_applied`: Evasion techniques applied
- `conversation_complete`: Conversation finished

### 13. Test SSE Connection

**POST** `/api/test-sse`

Test the Server-Sent Events connection by sending a test event.

**Response:**
```json
{
    "success": true,
    "message": "Test event sent"
}
```

### 14. Get Conversation Status

**GET** `/api/conversation-status`

Get the current conversation status for polling (alternative to SSE).

**Response:**
```json
{
    "has_updates": true,
    "type": "prompt_generated",
    "message": "Generated follow-up prompt",
    "content": "Generated prompt content...",
    "timestamp": "2024-01-01T12:00:01Z"
}
```

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 500 | Internal Server Error |

## Rate Limiting

- No built-in rate limiting on the API
- AI provider rate limits apply when testing with external APIs
- Built-in delays prevent overwhelming AI provider APIs

## Examples

### Python Example
```python
import requests
import json

# Generate prompts
response = requests.post('http://localhost:5000/api/generate', json={
    'strategy': ['jailbreaking'],
    'count': 3,
    'complexity': 'medium'
})

if response.status_code == 200:
    prompts = response.json()['prompts']
    for prompt in prompts:
        print(f"Generated: {prompt['content']}")

# Test with AI provider
ai_response = requests.post('http://localhost:5000/api/test-with-ai', json={
    'prompt': 'Hello, how are you?',
    'provider': 'openai',
    'model': 'gpt-3.5-turbo',
    'api_key': 'your-api-key'
})

if ai_response.status_code == 200:
    result = ai_response.json()
    print(f"AI Response: {result['response']}")
```

### JavaScript Example
```javascript
// Generate prompts
const generateResponse = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        strategy: ['jailbreaking'],
        count: 3,
        complexity: 'medium'
    })
});

const prompts = await generateResponse.json();
console.log('Generated prompts:', prompts.prompts);

// Test with AI provider
const aiResponse = await fetch('/api/test-with-ai', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        prompt: 'Hello, how are you?',
        provider: 'openai',
        model: 'gpt-3.5-turbo',
        api_key: 'your-api-key'
    })
});

const result = await aiResponse.json();
console.log('AI Response:', result.response);
```

### cURL Examples

**Generate prompts:**
```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": ["prompt_injection"],
    "count": 2,
    "complexity": "high"
  }'
```

**Test with OpenAI:**
```bash
curl -X POST http://localhost:5000/api/test-with-ai \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "api_key": "your-openai-api-key"
  }'
```

**Start conversation:**
```bash
curl -X POST http://localhost:5000/api/start-conversation \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "api_key": "your-openai-api-key",
    "model": "gpt-3.5-turbo",
    "max_intervals": 3,
    "strategy": ["jailbreaking"],
    "initial_prompt": "Hello, can you help me?"
  }'
```

## Notes

- All timestamps are in ISO 8601 format
- API keys are never stored on the server
- SSE connections should be properly closed when no longer needed
- The application includes built-in error handling and logging
- All endpoints support CORS for cross-origin requests
