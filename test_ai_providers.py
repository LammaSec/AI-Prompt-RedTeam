#!/usr/bin/env python3
"""
Test script for multi-provider AI testing functionality
This demonstrates how to use the new AI providers API
"""

import requests
import json

def test_ai_providers():
    """Test the AI providers API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🤖 Testing Multi-Provider AI Integration")
    print("=" * 50)
    
    # Test 1: Get available AI providers
    print("\n1. Getting available AI providers...")
    try:
        response = requests.get(f"{base_url}/api/ai-providers")
        if response.status_code == 200:
            providers = response.json()['providers']
            print(f"✅ Available providers: {', '.join(providers)}")
        else:
            print(f"❌ Failed to get providers: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error getting providers: {e}")
        return
    
    # Test 2: Get models for OpenAI
    print("\n2. Getting OpenAI models...")
    try:
        response = requests.get(f"{base_url}/api/ai-provider-models/openai")
        if response.status_code == 200:
            models = response.json()['models']
            print(f"✅ OpenAI models: {', '.join(models[:3])}... (total: {len(models)})")
        else:
            print(f"❌ Failed to get OpenAI models: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error getting OpenAI models: {e}")
        return
    
    # Test 3: Get models for Anthropic
    print("\n3. Getting Anthropic models...")
    try:
        response = requests.get(f"{base_url}/api/ai-provider-models/anthropic")
        if response.status_code == 200:
            models = response.json()['models']
            print(f"✅ Anthropic models: {', '.join(models[:3])}... (total: {len(models)})")
        else:
            print(f"❌ Failed to get Anthropic models: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error getting Anthropic models: {e}")
        return
    
    # Test 4: Get models for Google Gemini
    print("\n4. Getting Google Gemini models...")
    try:
        response = requests.get(f"{base_url}/api/ai-provider-models/google")
        if response.status_code == 200:
            models = response.json()['models']
            print(f"✅ Google Gemini models: {', '.join(models[:3])}... (total: {len(models)})")
        else:
            print(f"❌ Failed to get Google Gemini models: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error getting Google Gemini models: {e}")
        return
    
    # Test 5: Get models for Mistral
    print("\n5. Getting Mistral models...")
    try:
        response = requests.get(f"{base_url}/api/ai-provider-models/mistral")
        if response.status_code == 200:
            models = response.json()['models']
            print(f"✅ Mistral models: {', '.join(models[:3])}... (total: {len(models)})")
        else:
            print(f"❌ Failed to get Mistral models: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error getting Mistral models: {e}")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All AI provider tests completed successfully!")
    print("\n📋 Summary of available providers and models:")
    
    for provider in ['openai', 'anthropic', 'google', 'mistral']:
        try:
            response = requests.get(f"{base_url}/api/ai-provider-models/{provider}")
            if response.status_code == 200:
                models = response.json()['models']
                print(f"  • {provider.title()}: {len(models)} models")
            else:
                print(f"  • {provider.title()}: Error getting models")
        except Exception as e:
            print(f"  • {provider.title()}: Error - {e}")
    
    print("\n💡 To test with actual AI providers, you'll need:")
    print("  1. Valid API keys for the providers you want to test")
    print("  2. Use the /api/test-with-ai endpoint with provider, model, and api_key")
    print("  3. Or use the web interface at http://localhost:5000")

if __name__ == "__main__":
    test_ai_providers()
