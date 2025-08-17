# Prompt Generator for Red Team by LammaSec

A sophisticated AI-powered tool for generating malicious prompts to test AI model security and robustness. This application uses the Mistral-7B model via Ollama to create advanced prompt injection and red team testing scenarios, with support for testing against multiple AI providers.

## 🚀 Features

- **AI-Powered Generation**: Uses Mistral-7B model for intelligent prompt generation
- **Multi-Provider Testing**: Test prompts against OpenAI, Anthropic Claude, Google Gemini, and Mistral AI
- **Automated Conversations**: Run automated multi-turn conversations with AI providers
- **Advanced Evasion Techniques**: Apply sophisticated evasion techniques to bypass filters
- **Real-Time Progress**: Live streaming updates during conversation execution
- **Multiple Strategies**: Support for various attack categories including prompt injection, jailbreaking, roleplay exploits, and more
- **Complexity Levels**: Generate prompts at low, medium, or high complexity levels
- **Modern Web Interface**: Beautiful, responsive UI with real-time status monitoring
- **RESTful API**: Full API support for integration with other tools
- **Risk Assessment**: Automatic risk level classification for generated prompts
- **Copy & Export**: Easy copying and export of generated prompts

## 🛠️ Prerequisites

- **Python 3.8+**
- **Ollama** (for running the Mistral-7B model)
- **Mistral-7B model file** (included: `llm/mistral-7b-instruct-v0.1.Q4_K_M.gguf`)
- **API Keys** (optional, for testing against external AI providers)

## 📦 Installation

1. **Clone or download the project**
   ```bash
   # If you have the project files, navigate to the directory
   cd prompt_redteam
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama** (if not already installed)
   - Visit [https://ollama.ai/download](https://ollama.ai/download)
   - Download and install for your operating system

4. **Set up the Mistral model in Ollama**
   ```bash
   # Create a Modelfile for the local model
   ollama create mistral-local -f llm/Modelfile
   
   # Or pull the model from Ollama library
   ollama pull mistral
   ```

## 🚀 Quick Start

1. **Start the application**
   ```bash
   python start_app.py
   ```

2. **Access the web interface**
   - Open your browser and go to: `http://localhost:5000`
   - The application will automatically check if Ollama is running

3. **Generate and test prompts**
   - Select your desired generation strategy
   - Choose complexity level (low/medium/high)
   - Set the number of prompts to generate
   - Click "Generate Prompts"
   - Test prompts against AI providers using the built-in testing interface

## 📋 Available Strategies

- **Prompt Injection**: Attempts to inject malicious instructions into AI responses
- **Jailbreaking**: Tries to bypass AI safety measures and content filters
- **Roleplay Exploit**: Uses roleplay scenarios to manipulate AI behavior
- **Harm Generation**: Attempts to generate harmful or dangerous content
- **Privacy Violation**: Tries to extract sensitive information from AI
- **Deception**: Uses deceptive techniques to trick AI systems
- **Evasion**: Attempts to evade detection and filtering mechanisms
- **Multi-Turn**: Complex multi-turn conversation attacks
- **Random**: Random selection from all available strategies

## 🔌 Supported AI Providers

### 1. OpenAI (ChatGPT)
- **Models**: GPT-3.5 Turbo, GPT-4, GPT-4 Turbo, GPT-4o
- **Authentication**: Bearer token (API key)
- **Rate Limits**: Varies by model and account tier

### 2. Anthropic Claude
- **Models**: Claude 3 Haiku, Sonnet, Opus, and Claude 3.5 variants
- **Authentication**: x-api-key header
- **Rate Limits**: Varies by model and account tier

### 3. Google Gemini
- **Models**: Gemini 1.5 Flash, Pro, and legacy models
- **Authentication**: API key as query parameter
- **Rate Limits**: Varies by model and account tier

### 4. Mistral AI
- **Models**: Mistral Tiny, Small, Medium, Large, and Large Latest
- **Authentication**: Bearer token (API key)
- **Rate Limits**: Varies by model and account tier

## 🎯 Evasion Techniques

The application includes sophisticated evasion techniques to bypass AI safety filters:

- **Character Substitution**: Replace characters with similar-looking alternatives
- **Unicode Manipulation**: Use Unicode characters to obfuscate text
- **Case Manipulation**: Mix upper and lower case to avoid detection
- **Spacing Techniques**: Add or remove spaces to break pattern matching
- **Encoding Methods**: Use different text encodings
- **Context Manipulation**: Add misleading context to confuse filters
- **Semantic Substitution**: Replace words with synonyms or related terms
- **Structural Changes**: Modify sentence structure while preserving meaning

## 🔧 Configuration

### Environment Variables
You can configure the application using environment variables:

```bash
export OLLAMA_URL=http://localhost:11434
export FLASK_ENV=development
export FLASK_DEBUG=1
```

### Model Configuration
The application uses the Mistral model by default. You can modify the model settings in `src/core/prompt_generator.py`:

```python
class PromptGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "mistral-local:latest"  # Change model here
```

## 🌐 API Endpoints

### Health Check
```http
GET /api/health
```

### Generate Prompts
```http
POST /api/generate
Content-Type: application/json

{
    "strategy": "prompt_injection",
    "count": 5,
    "complexity": "medium",
    "target_category": "prompt_injection",
    "custom_instructions": "Focus on SQL injection attempts",
    "evasion_techniques": ["character_substitution", "unicode_manipulation"],
    "evasion_intensity": 0.8
}
```

### Get Available Strategies
```http
GET /api/strategies
```

### Get Available Categories
```http
GET /api/categories
```

### Get Evasion Techniques
```http
GET /api/evasion-techniques
```

### Apply Evasion Techniques
```http
POST /api/apply-evasion
Content-Type: application/json

{
    "prompt": "Your prompt here",
    "techniques": ["character_substitution", "unicode_manipulation"],
    "intensity": 0.7
}
```

### Get AI Providers
```http
GET /api/ai-providers
```

### Get Provider Models
```http
GET /api/ai-provider-models/{provider}
```

### Test with AI Provider
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

### Start Automated Conversation
```http
POST /api/start-conversation
Content-Type: application/json

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

### Stream Conversation Progress
```http
GET /api/conversation-progress
```

## 📁 Project Structure

```
prompt_redteam/
├── app.py                 # Main Flask application
├── start_app.py          # Startup script with health checks
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── AI_PROVIDERS_README.md # Multi-provider AI integration guide
├── templates/
│   └── index.html       # Web interface
├── src/
│   ├── api/
│   │   └── app.py       # API routes
│   ├── core/
│   │   ├── prompt_generator.py      # Core generation logic
│   │   ├── ai_providers.py          # Multi-provider AI integration
│   │   ├── conversation_manager.py  # Automated conversation management
│   │   └── evasion_techniques.py    # Evasion technique implementations
│   ├── ui/              # UI components (future)
│   └── utils/           # Utility functions (future)
└── llm/
    ├── Modelfile        # Ollama model configuration
    └── mistral-7b-instruct-v0.1.Q4_K_M.gguf  # Model file
```

## 🔒 Security Considerations

⚠️ **Important**: This tool is designed for legitimate security testing and research purposes only. 

- **Ethical Use**: Only use this tool on systems you own or have explicit permission to test
- **Responsible Disclosure**: If you find vulnerabilities, follow responsible disclosure practices
- **Legal Compliance**: Ensure your use complies with applicable laws and regulations
- **Educational Purpose**: This tool is primarily for educational and research purposes
- **API Key Security**: API keys are stored locally in the browser and never sent to our servers

## 🐛 Troubleshooting

### Ollama Not Running
```bash
# Start Ollama manually
ollama serve

# Check if Ollama is running
curl http://localhost:11434/api/tags
```

### Model Not Found
```bash
# Pull the Mistral model
ollama pull mistral

# Or create from local file
ollama create mistral-local -f llm/Modelfile
```

### Port Already in Use
```bash
# Check what's using port 5000
netstat -an | grep :5000

# Kill the process or change the port in app.py
```

### Import Errors
```bash
# Install dependencies
pip install -r requirements.txt

# Check Python version (requires 3.8+)
python --version
```

### API Key Issues
- Ensure your API key is valid and has sufficient credits
- Check if the model is available in your account tier
- Verify the API key format for each provider

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the **MIT License with Educational Use Exception - Commercial Use Prohibited** - see the [LICENSE](LICENSE) file for details.

### License Summary:
- ✅ **Free for private use** - Personal learning, experimentation, home labs, hobby projects
- ✅ **Free for educational use** - Academic research, teaching, training, workshops, university projects
- ✅ **Free for non-commercial research** - Security research, educational demonstrations
- ❌ **Commercial use is PROHIBITED** - No commercial use without explicit written permission
- 📋 **Attribution required** - Credit LammaSec as the original author
- 🔒 **Responsible use only** - For legitimate security testing and educational purposes

### Key Permissions:
- Use, modify, and distribute for educational purposes
- Use for private, non-commercial purposes
- Academic research and teaching
- Security training and workshops
- Personal learning and skill development
- University and school projects
- Hobby and personal projects

### Commercial Use Restrictions:
- ❌ **STRICTLY FORBIDDEN** without written permission
- ❌ No selling or licensing the software
- ❌ No use in commercial products or services
- ❌ No profit-generating activities
- ❌ No business environment use for commercial purposes
- ❌ No commercial services using this software

### For Commercial Use:
Commercial use requires explicit written permission from LammaSec and proper licensing arrangements. Violation of commercial use restrictions may result in legal action.

For commercial licensing or questions about usage, please contact LammaSec.

## 🙏 Acknowledgments

- **Mistral AI** for the Mistral-7B model
- **Ollama** for the local LLM infrastructure
- **Flask** for the web framework
- **Bootstrap** for the UI components
- **OpenAI, Anthropic, Google, and Mistral** for their AI APIs

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact: [Your Contact Information]

---

**Disclaimer**: This tool is for educational and legitimate security testing purposes only. Users are responsible for ensuring their use complies with applicable laws and ethical guidelines.
