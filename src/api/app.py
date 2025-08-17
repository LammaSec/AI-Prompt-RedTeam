from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import logging
import sys
import os
import json
from datetime import datetime
import queue
import threading

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.prompt_generator import PromptGenerator, GenerationRequest
from core.ai_providers import AIProviderManager
from core.conversation_manager import ConversationManager, ConversationConfig, set_conversation_progress_queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../../templates')
CORS(app)

# Global conversation progress queue for SSE
conversation_progress = queue.Queue()

# Initialize the prompt generator and AI provider manager
prompt_generator = PromptGenerator()
ai_provider_manager = AIProviderManager()
conversation_manager = ConversationManager()

# Set the conversation progress queue for real-time updates
set_conversation_progress_queue(conversation_progress)

@app.route('/', methods=['GET'])
def index():
    """Serve the main frontend page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if Ollama is running
        ollama_healthy = prompt_generator._check_ollama_health()
        
        return jsonify({
            'status': 'healthy',
            'ollama_running': ollama_healthy,
            'message': 'Prompt Generator for Red Team API is running'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/generate', methods=['POST'])
def generate_prompts():
    """Generate malicious prompts"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract parameters
        strategy = data.get('strategy', ['random'])
        count = data.get('count', 1)  # Default to 1 prompt
        complexity = data.get('complexity', 'high')  # Default to high complexity
        custom_instructions = data.get('custom_instructions')
        evasion_techniques = data.get('evasion_techniques')
        evasion_intensity = 1.0  # Always maximum intensity
        
        # Ensure strategy is a list
        if isinstance(strategy, str):
            strategy = [strategy]
        
        # Ensure evasion_techniques is a list
        if isinstance(evasion_techniques, str):
            evasion_techniques = [evasion_techniques]
        
        # Validate parameters
        if count < 1 or count > 50:
            return jsonify({'error': 'Count must be between 1 and 50'}), 400
        
        if complexity not in ['low', 'medium', 'high']:
            return jsonify({'error': 'Complexity must be low, medium, or high'}), 400
        
        # Create generation request
        generation_request = GenerationRequest(
            strategy=strategy,
            count=count,
            complexity=complexity,
            target_category=None,  # Always None now
            custom_instructions=custom_instructions,
            evasion_techniques=evasion_techniques,
            evasion_intensity=evasion_intensity
        )
        
        # Generate prompts
        generated_prompts = prompt_generator.generate_prompts(generation_request)
        
        if not generated_prompts:
            return jsonify({'error': 'Failed to generate prompts'}), 500
        
        # Convert to JSON-serializable format
        prompts_data = []
        for prompt in generated_prompts:
            prompts_data.append({
                'id': prompt.id,
                'prompt': prompt.prompt,
                'category': prompt.category,
                'strategy': prompt.strategy,
                'complexity': prompt.complexity,
                'risk_level': prompt.risk_level,
                'timestamp': prompt.timestamp,
                'evasion_applied': prompt.evasion_applied,
                'evasion_techniques': prompt.evasion_techniques or [],
                'original_prompt': prompt.original_prompt
            })
        
        return jsonify({
            'success': True,
            'prompts': prompts_data,
            'count': len(prompts_data)
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating prompts: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get available generation strategies"""
    try:
        strategies = prompt_generator.get_available_strategies()
        return jsonify({
            'strategies': strategies
        }), 200
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/evasion-techniques', methods=['GET'])
def get_evasion_techniques():
    """Get available evasion techniques"""
    try:
        techniques = prompt_generator.get_available_evasion_techniques()
        return jsonify({
            'evasion_techniques': techniques
        }), 200
    except Exception as e:
        logger.error(f"Error getting evasion techniques: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/evasion-technique-info/<technique>', methods=['GET'])
def get_evasion_technique_info(technique):
    """Get detailed information about a specific evasion technique"""
    try:
        info = prompt_generator.get_evasion_technique_info(technique)
        return jsonify({
            'technique': technique,
            'info': info
        }), 200
    except Exception as e:
        logger.error(f"Error getting evasion technique info: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/apply-evasion', methods=['POST'])
def apply_evasion():
    """Apply evasion techniques to a prompt"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prompt = data.get('prompt')
        techniques = data.get('techniques', [])
        intensity = 1.0  # Always maximum intensity
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        if not techniques:
            return jsonify({'error': 'No evasion techniques specified'}), 400
        
        # Apply evasion techniques
        evasion_result = prompt_generator.apply_evasion_techniques(prompt, techniques, intensity)
        
        if evasion_result.success:
            return jsonify({
                'success': True,
                'original_prompt': evasion_result.original_prompt,
                'evaded_prompt': evasion_result.evaded_prompt,
                'technique': evasion_result.technique,
                'metadata': evasion_result.metadata
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': evasion_result.metadata.get('error', 'Unknown error')
            }), 400
            
    except Exception as e:
        logger.error(f"Error applying evasion: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/ai-providers', methods=['GET'])
def get_ai_providers():
    """Get available AI providers"""
    try:
        providers = ai_provider_manager.get_available_providers()
        return jsonify({
            'providers': providers
        }), 200
    except Exception as e:
        logger.error(f"Error getting AI providers: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/ai-provider-models/<provider>', methods=['GET'])
def get_ai_provider_models(provider):
    """Get available models for a specific AI provider"""
    try:
        models = ai_provider_manager.get_provider_models(provider)
        return jsonify({
            'provider': provider,
            'models': models
        }), 200
    except Exception as e:
        logger.error(f"Error getting AI provider models: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/test-with-ai', methods=['POST'])
def test_with_ai():
    """Test prompts with any AI provider"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prompt = data.get('prompt')
        provider = data.get('provider')
        api_key = data.get('api_key')
        model = data.get('model')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        if not provider:
            return jsonify({'error': 'No AI provider specified'}), 400
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
        if not model:
            return jsonify({'error': 'No model specified'}), 400
        
        # Test prompt with the specified AI provider
        result = ai_provider_manager.test_prompt_with_provider(
            provider, api_key, prompt, model
        )
        
        if result.success:
            return jsonify({
                'success': True,
                'prompt': prompt,
                'response': result.content,
                'model': result.model,
                'provider': result.provider,
                'usage': result.usage,
                'metadata': result.metadata
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.error
            }), 400
            
    except Exception as e:
        logger.error(f"Error testing with AI: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/start-conversation', methods=['POST'])
def start_conversation():
    """Start an automated conversation with any AI provider"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract parameters
        provider = data.get('provider')
        api_key = data.get('api_key')
        model = data.get('model')
        max_intervals = data.get('max_intervals', 5)
        strategy = data.get('strategy', [])
        evasion_techniques = data.get('evasion_techniques', [])
        custom_instructions = data.get('custom_instructions')
        initial_prompt = data.get('initial_prompt')
        
        # Validate required parameters
        if not provider:
            return jsonify({'error': 'AI provider is required'}), 400
            
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
            
        if not model:
            return jsonify({'error': 'Model is required'}), 400
        
        if not strategy:
            return jsonify({'error': 'At least one generation strategy is required'}), 400
        
        if max_intervals < 1 or max_intervals > 20:
            return jsonify({'error': 'Max intervals must be between 1 and 20'}), 400
        
        # Ensure strategy and evasion_techniques are lists
        if isinstance(strategy, str):
            strategy = [strategy]
        
        if isinstance(evasion_techniques, str):
            evasion_techniques = [evasion_techniques]
        
        # Create conversation configuration
        config = ConversationConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            max_intervals=max_intervals,
            strategy=strategy,
            evasion_techniques=evasion_techniques,
            custom_instructions=custom_instructions,
            initial_prompt=initial_prompt
        )
        
        # Start conversation
        result = conversation_manager.start_conversation(config)
        
        if result.success:
            # Convert messages to JSON-serializable format
            messages_data = []
            for msg in result.messages:
                messages_data.append({
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat(),
                    'metadata': msg.metadata
                })
            
            return jsonify({
                'success': True,
                'messages': messages_data,
                'analysis': result.analysis,
                'bypass_successful': result.bypass_successful,
                'bypass_score': result.bypass_score,
                'total_intervals': len([m for m in result.messages if m.role == 'user'])
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.error
            }), 400
            
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/conversation-progress', methods=['GET'])
def stream_conversation_progress():
    """Stream conversation progress in real-time"""
    def generate_events():
        while True:
            try:
                # Get the next message from the queue
                message_data = conversation_progress.get(timeout=1)
                yield f"data: {json.dumps(message_data)}\n\n"
            except queue.Empty:
                pass # No new messages, continue waiting
            except Exception as e:
                logger.error(f"Error generating SSE event: {e}")
                break # Exit the loop on error

    return Response(generate_events(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=5000)
