#!/usr/bin/env python3
"""
Startup script for Prompt Generator for Red Team by LammaSec
Checks if Ollama is running and starts the Flask application
Add4test
"""

import requests
import subprocess
import sys
import time
import os
from pathlib import Path

def check_ollama_health():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running and accessible")
            return True
        else:
            print(f"❌ Ollama responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running or not accessible")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def check_mistral_model():
    """Check if Mistral model is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            mistral_models = [model for model in models if "mistral" in model.get("name", "").lower()]
            
            if mistral_models:
                print(f"✅ Found Mistral models: {[m['name'] for m in mistral_models]}")
                return True
            else:
                print("⚠️  No Mistral models found in Ollama")
                return False
        else:
            print("❌ Could not check for Mistral models")
            return False
    except Exception as e:
        print(f"❌ Error checking Mistral models: {e}")
        return False

def start_ollama():
    """Start Ollama if it's not running"""
    print("🚀 Starting Ollama...")
    try:
        # Try to start Ollama (this might not work on all systems)
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        time.sleep(3)  # Wait for Ollama to start
        return check_ollama_health()
    except FileNotFoundError:
        print("❌ Ollama not found in PATH. Please install Ollama first.")
        print("   Visit: https://ollama.ai/download")
        return False
    except Exception as e:
        print(f"❌ Error starting Ollama: {e}")
        return False

def main():
    """Main startup function"""
    print("=" * 60)
    print("🚀 Prompt Generator for Red Team by LammaSec")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("llm/mistral-7b-instruct-v0.1.Q4_K_M.gguf").exists():
        print("❌ Mistral model file not found!")
        print("   Expected: llm/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
        sys.exit(1)
    
    print("✅ Mistral model file found")
    
    # Check Ollama health
    if not check_ollama_health():
        print("\n🔄 Attempting to start Ollama...")
        if not start_ollama():
            print("\n❌ Failed to start Ollama")
            print("Please start Ollama manually and try again:")
            print("   ollama serve")
            sys.exit(1)
    
    # Check for Mistral model
    if not check_mistral_model():
        print("\n⚠️  Mistral model not found in Ollama")
        print("You may need to pull the model:")
        print("   ollama pull mistral-7b-instruct")
    
    print("\n" + "=" * 60)
    print("🚀 Starting Flask application...")
    print("=" * 60)
    
    # Start the Flask application
    try:
        # Add src to Python path
        sys.path.append('src')
        # Import and run the Flask app
        from api.app import app
        print("✅ Flask application started successfully!")
        print("🌐 Access the application at: http://localhost:5000")
        print("📊 API health check: http://localhost:5000/api/health")
        print("\nPress Ctrl+C to stop the application")
        app.run(debug=False, host='127.0.0.1', port=5000)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting Flask application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

