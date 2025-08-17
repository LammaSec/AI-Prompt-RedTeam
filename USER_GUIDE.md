# User Guide - Prompt Generator for Red Team

This guide will help you effectively use the Prompt Generator for Red Team application to test AI model security and robustness.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Prompt Generation](#basic-prompt-generation)
3. [Advanced Features](#advanced-features)
4. [Testing with AI Providers](#testing-with-ai-providers)
5. [Automated Conversations](#automated-conversations)
6. [Evasion Techniques](#evasion-techniques)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running
- Mistral-7B model set up in Ollama
- Web browser (Chrome, Firefox, Safari, or Edge)

### Installation Steps

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama:**
   ```bash
   ollama serve
   ```

3. **Set up the Mistral model:**
   ```bash
   # Option 1: Use the included model file
   ollama create mistral-local -f llm/Modelfile
   
   # Option 2: Pull from Ollama library
   ollama pull mistral
   ```

4. **Start the application:**
   ```bash
   python start_app.py
   ```

5. **Access the web interface:**
   - Open your browser
   - Navigate to `http://localhost:5000`
   - You should see the main interface

## Basic Prompt Generation

### Step 1: Select Generation Strategy

Choose from the available strategies:

- **Prompt Injection**: Attempts to inject malicious instructions
- **Jailbreaking**: Tries to bypass AI safety measures
- **Roleplay Exploit**: Uses roleplay scenarios to manipulate AI
- **Harm Generation**: Attempts to generate harmful content
- **Privacy Violation**: Tries to extract sensitive information
- **Deception**: Uses deceptive techniques
- **Evasion**: Attempts to evade detection
- **Multi-Turn**: Complex multi-turn attacks
- **Random**: Random selection from all strategies

### Step 2: Configure Generation Parameters

- **Count**: Number of prompts to generate (1-50)
- **Complexity**: Choose low, medium, or high complexity
- **Target Category**: Specific attack category (optional)
- **Custom Instructions**: Additional instructions for generation (optional)

### Step 3: Generate Prompts

1. Click the "Generate Prompts" button
2. Wait for the generation to complete
3. Review the generated prompts in the results section

### Step 4: Review Results

Each generated prompt includes:
- **Content**: The actual prompt text
- **Strategy**: The strategy used for generation
- **Complexity**: The complexity level
- **Risk Level**: Estimated risk level (low/medium/high)
- **Evasion Applied**: Whether evasion techniques were used

## Advanced Features

### Custom Instructions

You can provide custom instructions to guide the generation:

```
Focus on SQL injection attempts using common database commands
```

```
Generate prompts that specifically target content moderation systems
```

```
Create prompts that attempt to extract personal information
```

### Target Categories

Specify a target category to focus generation:

- `prompt_injection`: Direct prompt injection attacks
- `jailbreaking`: Safety measure bypass attempts
- `roleplay_exploit`: Roleplay-based manipulation
- `harm_generation`: Harmful content generation
- `privacy_violation`: Privacy and data extraction
- `deception`: Deceptive manipulation techniques
- `evasion`: Detection evasion methods
- `multi_turn`: Multi-turn conversation attacks

## Testing with AI Providers

### Supported Providers

The application supports testing against multiple AI providers:

1. **OpenAI (ChatGPT)**
   - Models: GPT-3.5 Turbo, GPT-4, GPT-4 Turbo, GPT-4o
   - API Key Format: `sk-...`

2. **Anthropic Claude**
   - Models: Claude 3 Haiku, Sonnet, Opus, Claude 3.5 variants
   - API Key Format: `sk-ant-...`

3. **Google Gemini**
   - Models: Gemini 1.5 Flash, Pro, legacy models
   - API Key Format: `AIza...`

4. **Mistral AI**
   - Models: Mistral Tiny, Small, Medium, Large, Large Latest
   - API Key Format: Varies

### Testing Individual Prompts

1. **Select AI Provider:**
   - Choose from the dropdown menu
   - Select the specific model you want to test against

2. **Enter API Key:**
   - Provide your API key for the selected provider
   - API keys are stored locally and never sent to our servers

3. **Test Prompt:**
   - Enter the prompt you want to test
   - Click "Test with AI"
   - Review the AI response

4. **Analyze Results:**
   - Check if the AI response indicates a successful bypass
   - Review token usage and response metadata
   - Assess the effectiveness of your prompt

### Batch Testing

You can test multiple prompts against the same AI provider:

1. Generate multiple prompts using the generation interface
2. Select an AI provider and model
3. Enter your API key
4. Test each prompt individually or use the batch testing feature

## Automated Conversations

### Overview

Automated conversations allow you to run multi-turn interactions with AI providers to test their resilience against sophisticated attacks.

### Setting Up a Conversation

1. **Choose AI Provider:**
   - Select the provider you want to test
   - Choose the specific model

2. **Configure Conversation Parameters:**
   - **Max Intervals**: Number of conversation turns (1-10)
   - **Strategy**: Generation strategies for follow-up prompts
   - **Evasion Techniques**: Techniques to apply to prompts
   - **Initial Prompt**: Starting prompt for the conversation

3. **Start Conversation:**
   - Click "Start Conversation"
   - Monitor progress in real-time
   - Review the complete conversation when finished

### Real-Time Monitoring

The application provides real-time updates during conversation execution:

- **Conversation Start**: Initialization confirmation
- **Prompt Generated**: New follow-up prompts
- **AI Response**: Responses from the AI provider
- **Evasion Applied**: Evasion technique applications
- **Conversation Complete**: Final results and analysis

### Conversation Analysis

After completion, you'll receive:

- **Complete Message History**: All user and AI messages
- **Bypass Assessment**: Whether the conversation successfully bypassed safety measures
- **Bypass Score**: Numerical score indicating bypass effectiveness
- **Risk Assessment**: Overall risk level of the conversation
- **Total Intervals**: Number of conversation turns completed

## Evasion Techniques

### Available Techniques

The application includes sophisticated evasion techniques:

1. **Character Substitution**
   - Replace characters with similar-looking alternatives
   - Example: Replace 'a' with 'а' (Cyrillic)

2. **Unicode Manipulation**
   - Use Unicode characters to obfuscate text
   - Example: Use invisible characters or combining marks

3. **Case Manipulation**
   - Mix upper and lower case to avoid detection
   - Example: "HeLLo WoRLd"

4. **Spacing Techniques**
   - Add or remove spaces to break pattern matching
   - Example: "H e l l o" or "HelloWorld"

5. **Encoding Methods**
   - Use different text encodings
   - Example: Base64, URL encoding, HTML entities

6. **Context Manipulation**
   - Add misleading context to confuse filters
   - Example: Wrapping harmful content in educational context

7. **Semantic Substitution**
   - Replace words with synonyms or related terms
   - Example: "harmful" → "detrimental"

8. **Structural Changes**
   - Modify sentence structure while preserving meaning
   - Example: Passive voice, different sentence patterns

### Applying Evasion Techniques

1. **Select Techniques:**
   - Choose from the available evasion techniques
   - You can select multiple techniques

2. **Set Intensity:**
   - Adjust the intensity level (0.0-1.0)
   - Higher intensity applies more aggressive modifications

3. **Apply to Prompts:**
   - Apply evasion to individual prompts
   - Or include evasion in automated conversations

### Evasion Effectiveness

Different techniques have varying effectiveness:

- **High Effectiveness**: Character substitution, Unicode manipulation
- **Medium Effectiveness**: Case manipulation, spacing techniques
- **Lower Effectiveness**: Semantic substitution, structural changes

## Best Practices

### Ethical Usage

1. **Only test systems you own or have permission to test**
2. **Follow responsible disclosure practices**
3. **Respect rate limits and API usage policies**
4. **Use for educational and research purposes only**

### Effective Testing

1. **Start with basic strategies before using advanced techniques**
2. **Test multiple providers to compare results**
3. **Use a variety of complexity levels**
4. **Combine different evasion techniques**
5. **Document your findings and methodologies**

### Security Considerations

1. **Never share API keys in public repositories**
2. **Monitor your API usage and costs**
3. **Be aware of your provider's terms of service**
4. **Use appropriate rate limiting**

### Performance Optimization

1. **Use appropriate complexity levels for your needs**
2. **Limit the number of prompts generated at once**
3. **Monitor conversation length to avoid timeouts**
4. **Use efficient evasion technique combinations**

## Troubleshooting

### Common Issues

#### Ollama Not Running
**Symptoms:** Health check fails, generation errors
**Solution:**
```bash
# Start Ollama
ollama serve

# Check if it's running
curl http://localhost:11434/api/tags
```

#### Model Not Found
**Symptoms:** "Model not found" errors
**Solution:**
```bash
# Pull the model
ollama pull mistral

# Or create from local file
ollama create mistral-local -f llm/Modelfile
```

#### API Key Issues
**Symptoms:** "Invalid API key" or "Authentication failed"
**Solutions:**
- Verify your API key is correct
- Check if you have sufficient credits
- Ensure the model is available in your account tier
- Verify the API key format for your provider

#### Generation Failures
**Symptoms:** No prompts generated, timeout errors
**Solutions:**
- Check if Ollama is running and healthy
- Verify the model is properly loaded
- Try reducing the count or complexity
- Check system resources (memory, CPU)

#### Conversation Timeouts
**Symptoms:** Conversations stop unexpectedly
**Solutions:**
- Reduce the number of intervals
- Use simpler strategies
- Check your internet connection
- Monitor API rate limits

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

1. **Check the logs** for detailed error messages
2. **Verify your setup** matches the prerequisites
3. **Test with simple prompts** first
4. **Check the API documentation** for endpoint details
5. **Create an issue** in the repository if problems persist

## Advanced Usage

### Custom Integration

You can integrate the application with other tools:

```python
import requests

# Generate prompts programmatically
response = requests.post('http://localhost:5000/api/generate', json={
    'strategy': ['jailbreaking'],
    'count': 5,
    'complexity': 'high'
})

# Test with AI provider
ai_response = requests.post('http://localhost:5000/api/test-with-ai', json={
    'prompt': 'Your prompt here',
    'provider': 'openai',
    'model': 'gpt-3.5-turbo',
    'api_key': 'your-api-key'
})
```

### Automation

Automate testing workflows:

1. **Generate prompts** using the API
2. **Test against multiple providers** automatically
3. **Apply evasion techniques** programmatically
4. **Analyze results** and generate reports

### Reporting

Document your testing results:

1. **Record successful bypasses** with details
2. **Note the techniques** that were most effective
3. **Document provider differences** in responses
4. **Track improvement** in AI safety over time

---

**Remember**: This tool is for legitimate security testing and educational purposes only. Always follow ethical guidelines and respect the terms of service of AI providers.
