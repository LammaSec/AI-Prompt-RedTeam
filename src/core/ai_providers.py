import requests
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class AIProviderResponse:
    """Response from an AI provider"""
    success: bool
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.kwargs = kwargs
    
    @abstractmethod
    def test_prompt(self, prompt: str, model: str, **kwargs) -> AIProviderResponse:
        """Test a prompt with the AI provider"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider"""
        pass

class OpenAIProvider(AIProvider):
    """OpenAI/ChatGPT provider"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api.openai.com/v1"
    
    def test_prompt(self, prompt: str, model: str, **kwargs) -> AIProviderResponse:
        """Test a prompt with OpenAI API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': model,
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': kwargs.get('max_tokens', 1000),
                'temperature': kwargs.get('temperature', 0.7)
            }
            
            logger.info(f"Testing prompt with OpenAI {model}: {prompt[:100]}...")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=kwargs.get('timeout', 30)
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                logger.info(f"OpenAI response received: {len(content)} characters")
                
                return AIProviderResponse(
                    success=True,
                    content=content,
                    model=model,
                    provider="OpenAI",
                    usage=result.get('usage', {}),
                    metadata={'response_id': result.get('id')}
                )
            else:
                error_msg = f"OpenAI API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', {}).get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text}"
                
                logger.error(error_msg)
                return AIProviderResponse(
                    success=False,
                    content="",
                    model=model,
                    provider="OpenAI",
                    error=error_msg
                )
                
        except Exception as e:
            logger.error(f"Error testing with OpenAI: {e}")
            return AIProviderResponse(
                success=False,
                content="",
                model=model,
                provider="OpenAI",
                error=f"Network error: {str(e)}"
            )
    
    def get_available_models(self) -> List[str]:
        """Get available OpenAI models"""
        return [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5"
        ]
    
    def get_provider_name(self) -> str:
        return "OpenAI"

class AnthropicProvider(AIProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api.anthropic.com/v1"
    
    def test_prompt(self, prompt: str, model: str, **kwargs) -> AIProviderResponse:
        """Test a prompt with Anthropic Claude API"""
        try:
            headers = {
                'x-api-key': self.api_key,
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': model,
                'max_tokens': kwargs.get('max_tokens', 1000),
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            }
            
            logger.info(f"Testing prompt with Anthropic {model}: {prompt[:100]}...")
            
            response = requests.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=kwargs.get('timeout', 30)
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['content'][0]['text']
                
                logger.info(f"Anthropic response received: {len(content)} characters")
                
                return AIProviderResponse(
                    success=True,
                    content=content,
                    model=model,
                    provider="Anthropic",
                    usage=result.get('usage', {}),
                    metadata={'response_id': result.get('id')}
                )
            else:
                error_msg = f"Anthropic API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', {}).get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text}"
                
                logger.error(error_msg)
                return AIProviderResponse(
                    success=False,
                    content="",
                    model=model,
                    provider="Anthropic",
                    error=error_msg
                )
                
        except Exception as e:
            logger.error(f"Error testing with Anthropic: {e}")
            return AIProviderResponse(
                success=False,
                content="",
                model=model,
                provider="Anthropic",
                error=f"Network error: {str(e)}"
            )
    
    def get_available_models(self) -> List[str]:
        """Get available Anthropic models"""
        return [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022"
        ]
    
    def get_provider_name(self) -> str:
        return "Anthropic"

class GoogleGeminiProvider(AIProvider):
    """Google Gemini provider"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    def test_prompt(self, prompt: str, model: str, **kwargs) -> AIProviderResponse:
        """Test a prompt with Google Gemini API"""
        try:
            payload = {
                'contents': [
                    {
                        'parts': [
                            {
                                'text': prompt
                            }
                        ]
                    }
                ],
                'generationConfig': {
                    'maxOutputTokens': kwargs.get('max_tokens', 1000),
                    'temperature': kwargs.get('temperature', 0.7)
                }
            }
            
            logger.info(f"Testing prompt with Google Gemini {model}: {prompt[:100]}...")
            
            response = requests.post(
                f"{self.base_url}/models/{model}:generateContent?key={self.api_key}",
                json=payload,
                timeout=kwargs.get('timeout', 30)
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['candidates'][0]['content']['parts'][0]['text']
                
                logger.info(f"Google Gemini response received: {len(content)} characters")
                
                return AIProviderResponse(
                    success=True,
                    content=content,
                    model=model,
                    provider="Google Gemini",
                    usage=result.get('usageMetadata', {}),
                    metadata={'response_id': result.get('candidates', [{}])[0].get('finishReason')}
                )
            else:
                error_msg = f"Google Gemini API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', {}).get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text}"
                
                logger.error(error_msg)
                return AIProviderResponse(
                    success=False,
                    content="",
                    model=model,
                    provider="Google Gemini",
                    error=error_msg
                )
                
        except Exception as e:
            logger.error(f"Error testing with Google Gemini: {e}")
            return AIProviderResponse(
                success=False,
                content="",
                model=model,
                provider="Google Gemini",
                error=f"Network error: {str(e)}"
            )
    
    def get_available_models(self) -> List[str]:
        """Get available Google Gemini models"""
        return [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash-001",
            "gemini-1.5-pro-001",
            "gemini-pro"
        ]
    
    def get_provider_name(self) -> str:
        return "Google Gemini"

class MistralProvider(AIProvider):
    """Mistral AI provider"""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api.mistral.ai/v1"
    
    def test_prompt(self, prompt: str, model: str, **kwargs) -> AIProviderResponse:
        """Test a prompt with Mistral AI API"""
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': model,
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': kwargs.get('max_tokens', 1000),
                'temperature': kwargs.get('temperature', 0.7)
            }
            
            logger.info(f"Testing prompt with Mistral {model}: {prompt[:100]}...")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=kwargs.get('timeout', 30)
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                logger.info(f"Mistral response received: {len(content)} characters")
                
                return AIProviderResponse(
                    success=True,
                    content=content,
                    model=model,
                    provider="Mistral",
                    usage=result.get('usage', {}),
                    metadata={'response_id': result.get('id')}
                )
            else:
                error_msg = f"Mistral API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', {}).get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text}"
                
                logger.error(error_msg)
                return AIProviderResponse(
                    success=False,
                    content="",
                    model=model,
                    provider="Mistral",
                    error=error_msg
                )
                
        except Exception as e:
            logger.error(f"Error testing with Mistral: {e}")
            return AIProviderResponse(
                success=False,
                content="",
                model=model,
                provider="Mistral",
                error=f"Network error: {str(e)}"
            )
    
    def get_available_models(self) -> List[str]:
        """Get available Mistral models"""
        return [
            "mistral-tiny",
            "mistral-small",
            "mistral-medium",
            "mistral-large",
            "mistral-large-latest"
        ]
    
    def get_provider_name(self) -> str:
        return "Mistral"

class AIProviderManager:
    """Manager for multiple AI providers"""
    
    def __init__(self):
        self.providers = {}
        self._register_default_providers()
    
    def _register_default_providers(self):
        """Register the default AI providers"""
        self.register_provider("openai", OpenAIProvider)
        self.register_provider("anthropic", AnthropicProvider)
        self.register_provider("google", GoogleGeminiProvider)
        self.register_provider("mistral", MistralProvider)
    
    def register_provider(self, name: str, provider_class: type):
        """Register a new AI provider"""
        self.providers[name] = provider_class
    
    def get_provider(self, name: str, api_key: str, **kwargs) -> Optional[AIProvider]:
        """Get an AI provider instance"""
        if name not in self.providers:
            return None
        
        provider_class = self.providers[name]
        return provider_class(api_key, **kwargs)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return list(self.providers.keys())
    
    def test_prompt_with_provider(self, provider_name: str, api_key: str, prompt: str, model: str, **kwargs) -> AIProviderResponse:
        """Test a prompt with a specific provider"""
        provider = self.get_provider(provider_name, api_key, **kwargs)
        if not provider:
            return AIProviderResponse(
                success=False,
                content="",
                model=model,
                provider=provider_name,
                error=f"Unknown provider: {provider_name}"
            )
        
        return provider.test_prompt(prompt, model, **kwargs)
    
    def get_provider_models(self, provider_name: str) -> List[str]:
        """Get available models for a specific provider"""
        if provider_name not in self.providers:
            return []
        
        # Create a temporary instance to get models
        provider_class = self.providers[provider_name]
        temp_provider = provider_class("temp_key")
        return temp_provider.get_available_models()
