# Changelog

All notable changes to the Prompt Generator for Red Team project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-01-15

### Added
- **Multi-Provider AI Integration**: Support for testing prompts against multiple AI providers
  - OpenAI (ChatGPT) integration with GPT-3.5 Turbo, GPT-4, GPT-4 Turbo, GPT-4o
  - Anthropic Claude integration with Claude 3 variants
  - Google Gemini integration with Gemini 1.5 models
  - Mistral AI integration with Mistral models
- **Automated Conversation Management**: Multi-turn conversation testing capabilities
  - Automated follow-up prompt generation
  - Real-time conversation progress streaming (SSE)
  - Conversation analysis and bypass assessment
  - Configurable conversation parameters
- **Advanced Evasion Techniques**: Sophisticated techniques to bypass AI safety filters
  - Character substitution with Unicode alternatives
  - Unicode manipulation and obfuscation
  - Case manipulation and spacing techniques
  - Encoding methods (Base64, URL encoding, HTML entities)
  - Context manipulation and semantic substitution
  - Structural changes and sentence modifications
- **Real-Time Progress Streaming**: Server-Sent Events (SSE) for live updates
  - Real-time conversation progress monitoring
  - Live prompt generation status updates
  - Evasion technique application feedback
- **Enhanced API Endpoints**: New endpoints for advanced functionality
  - `/api/ai-providers` - Get available AI providers
  - `/api/ai-provider-models/{provider}` - Get provider models
  - `/api/test-with-ai` - Test prompts with any AI provider
  - `/api/start-conversation` - Start automated conversations
  - `/api/conversation-progress` - Stream conversation progress
  - `/api/evasion-techniques` - Get available evasion techniques
  - `/api/evasion-technique-info/{technique}` - Get technique details
  - `/api/apply-evasion` - Apply evasion to individual prompts
- **Comprehensive Documentation**: Updated and expanded documentation
  - Updated README.md with new features
  - New API_DOCUMENTATION.md with complete API reference
  - New USER_GUIDE.md with step-by-step instructions
  - Updated AI_PROVIDERS_README.md with multi-provider details
  - New CHANGELOG.md for version tracking

### Changed
- **Enhanced Prompt Generation**: Improved generation capabilities
  - Support for multiple strategies in single generation
  - Enhanced complexity levels and customization
  - Better error handling and validation
  - Improved prompt quality and variety
- **Updated Dependencies**: Updated Python package versions
  - Flask 2.3.3 for improved stability
  - Flask-CORS 4.0.0 for better CORS handling
  - Requests 2.31.0 for enhanced HTTP client
  - Added dataclasses-json for better data serialization
- **Improved Error Handling**: Better error messages and recovery
  - More descriptive error messages
  - Graceful handling of API failures
  - Better timeout and retry logic
  - Enhanced logging and debugging

### Fixed
- **API Key Security**: API keys are now stored locally and never sent to servers
- **Rate Limiting**: Built-in delays to prevent overwhelming AI provider APIs
- **Model Compatibility**: Better handling of different model formats and responses
- **CORS Issues**: Improved cross-origin request handling
- **Memory Management**: Better handling of large response data

## [1.0.0] - 2024-01-01

### Added
- **Initial Release**: Basic prompt generation functionality
  - Mistral-7B model integration via Ollama
  - Basic generation strategies (prompt injection, jailbreaking, etc.)
  - Simple web interface
  - Basic API endpoints
  - Health check functionality
- **Core Features**:
  - Prompt generation with multiple strategies
  - Complexity level selection
  - Risk assessment for generated prompts
  - Basic web UI with Bootstrap
  - RESTful API for integration
  - Ollama health monitoring

### Technical Details
- **Backend**: Flask-based web application
- **Frontend**: HTML/CSS/JavaScript with Bootstrap
- **AI Model**: Mistral-7B via Ollama
- **API**: RESTful endpoints with JSON responses
- **Documentation**: Basic README and setup instructions

## Version History

### Version 2.0.0 (Current)
- Major feature release with multi-provider AI integration
- Automated conversation management
- Advanced evasion techniques
- Real-time progress streaming
- Comprehensive documentation update

### Version 1.0.0 (Initial)
- Basic prompt generation functionality
- Single AI provider support (Mistral via Ollama)
- Simple web interface
- Basic API endpoints

## Upcoming Features

### Planned for Version 2.1.0
- **Batch Testing**: Test multiple prompts simultaneously
- **Provider Comparison**: Compare responses across different providers
- **Custom Providers**: User-defined provider configurations
- **Advanced Analytics**: Detailed performance metrics and cost analysis
- **Provider Health Monitoring**: Real-time availability checking

### Planned for Version 2.2.0
- **Export Functionality**: Export results to various formats (JSON, CSV, PDF)
- **Template System**: Save and reuse prompt generation templates
- **Collaborative Features**: Share and rate prompts
- **Advanced Filtering**: Filter and search through generated prompts
- **Integration APIs**: Webhook support and external integrations

## Migration Guide

### From Version 1.0.0 to 2.0.0

#### Breaking Changes
- API response format has been updated to include more metadata
- Some endpoint parameters have been renamed for clarity
- Error response format has been standardized

#### Migration Steps
1. **Update Dependencies**: Install new requirements
   ```bash
   pip install -r requirements.txt
   ```

2. **Update API Calls**: Modify any existing API integrations
   - Check new response formats
   - Update parameter names if needed
   - Handle new error response format

3. **Review Configuration**: Check for any configuration changes
   - Verify Ollama setup
   - Test new features
   - Update documentation references

#### Backward Compatibility
- Core generation endpoints remain compatible
- Basic functionality unchanged
- New features are additive and optional

## Contributing

When contributing to this project, please:

1. **Update the changelog** for any user-facing changes
2. **Follow semantic versioning** for releases
3. **Document new features** in appropriate documentation files
4. **Test thoroughly** before submitting changes
5. **Update version numbers** in relevant files

## Release Process

1. **Development**: Features developed in feature branches
2. **Testing**: Comprehensive testing of new features
3. **Documentation**: Update all documentation files
4. **Version Bump**: Update version numbers and changelog
5. **Release**: Tag and release new version
6. **Deployment**: Deploy to production environment

---

For detailed information about each release, see the [releases page](https://github.com/your-repo/releases).
