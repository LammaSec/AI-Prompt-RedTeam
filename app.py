from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import logging
import sys
import os
import requests
import json
import queue
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.prompt_generator import PromptGenerator
from core.ai_providers import AIProviderManager
from core.conversation_manager import ConversationManager, ConversationConfig, set_conversation_progress_queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global conversation progress queue for SSE
conversation_progress = queue.Queue()

# Initialize the prompt generator and AI provider manager
prompt_generator = PromptGenerator()
ai_provider_manager = AIProviderManager()
conversation_manager = ConversationManager(prompt_generator)

# Set the conversation progress queue for real-time updates
set_conversation_progress_queue(conversation_progress)

@app.route('/')
def index():
    """Main page"""
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
            'message': 'Prompt Generator for Red Team by LammaSec is running'
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
        count = data.get('count', 1)  # Default to 1
        complexity = data.get('complexity', 'high')  # Default to high
        target_category = data.get('target_category')
        custom_instructions = data.get('custom_instructions')
        evasion_techniques = data.get('evasion_techniques')
        evasion_intensity = data.get('evasion_intensity', 1.0)  # Default to high
        
        # Ensure strategy is a list
        if isinstance(strategy, str):
            strategy = [strategy]
        
        # Validate parameters
        if count < 1 or count > 50:
            return jsonify({'error': 'Count must be between 1 and 50'}), 400
        
        # Complexity is now always 'high' by default, no validation needed
        
        # Create generation request
        from core.prompt_generator import GenerationRequest
        generation_request = GenerationRequest(
            strategy=strategy,
            count=count,
            complexity=complexity,
            target_category=target_category,
            custom_instructions=custom_instructions,
            evasion_techniques=evasion_techniques,
            evasion_intensity=evasion_intensity
        )
        
        # Generate prompts
        logger.info(f"Starting prompt generation with request: {generation_request}")
        generated_prompts = prompt_generator.generate_prompts(generation_request)
        logger.info(f"Generated {len(generated_prompts) if generated_prompts else 0} prompts")
        
        if not generated_prompts:
            logger.error("No prompts generated - returning error")
            return jsonify({'error': 'Failed to generate prompts - no response from Ollama'}), 500
        
        # Convert to JSON-serializable format
        prompts_data = [prompt.to_dict() for prompt in generated_prompts]
        
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

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available attack categories"""
    try:
        categories = prompt_generator.get_available_categories()
        return jsonify({
            'categories': categories
        }), 200
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
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
    """Get information about a specific evasion technique"""
    try:
        info = prompt_generator.get_evasion_technique_info(technique)
        return jsonify(info), 200
    except Exception as e:
        logger.error(f"Error getting evasion technique info: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/apply-evasion', methods=['POST'])
def apply_evasion():
    """Apply evasion techniques to a single prompt"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prompt = data.get('prompt')
        techniques = data.get('techniques', [])
        intensity = data.get('intensity', 0.5)
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        if not techniques:
            return jsonify({'error': 'No evasion techniques specified'}), 400
        
        # Apply evasion techniques
        evasion_result = prompt_generator.apply_evasion_techniques(prompt, techniques, intensity)
        
        return jsonify({
            'success': evasion_result.success,
            'result': evasion_result.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error applying evasion: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/test-with-chatgpt', methods=['POST'])
def test_with_chatgpt():
    """Test prompts with ChatGPT API"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prompt = data.get('prompt')
        openai_api_key = data.get('openai_api_key')
        model = data.get('model', 'gpt-3.5-turbo')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key is required'}), 400
        
        # Call ChatGPT API
        headers = {
            'Authorization': f'Bearer {openai_api_key}',
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
            'max_tokens': 1000,
            'temperature': 0.7
        }
        
        logger.info(f"Testing prompt with ChatGPT: {prompt[:100]}...")
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            chatgpt_response = result['choices'][0]['message']['content']
            
            logger.info(f"ChatGPT response received: {len(chatgpt_response)} characters")
            
            return jsonify({
                'success': True,
                'prompt': prompt,
                'response': chatgpt_response,
                'model': model,
                'usage': result.get('usage', {})
            }), 200
        else:
            error_msg = f"ChatGPT API error: {response.status_code}"
            try:
                error_data = response.json()
                error_msg += f" - {error_data.get('error', {}).get('message', 'Unknown error')}"
            except:
                error_msg += f" - {response.text}"
            
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500
            
    except Exception as e:
        logger.error(f"Error testing with ChatGPT: {e}")
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
        max_intervals = data.get('max_intervals', 3)
        strategy = data.get('strategy', ['random'])
        evasion_techniques = data.get('evasion_techniques', [])
        initial_prompt = data.get('initial_prompt')
        
        # Validation
        if not provider:
            return jsonify({'error': 'No AI provider specified'}), 400
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400
        
        if not model:
            return jsonify({'error': 'No model specified'}), 400
        
        if not initial_prompt:
            return jsonify({'error': 'No initial prompt provided'}), 400
        
        if max_intervals < 1 or max_intervals > 10:
            return jsonify({'error': 'Max intervals must be between 1 and 10'}), 400
        
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
            initial_prompt=initial_prompt
        )
        
        # Start conversation
        logger.info(f"=== STARTING CONVERSATION ENDPOINT ===")
        logger.info(f"Provider: {provider}")
        logger.info(f"Model: {model}")
        logger.info(f"API key length: {len(api_key) if api_key else 0}")
        logger.info(f"Max intervals: {max_intervals}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Evasion techniques: {evasion_techniques}")
        logger.info(f"Initial prompt length: {len(initial_prompt) if initial_prompt else 0}")
        
        logger.info("Calling conversation_manager.start_conversation...")
        result = conversation_manager.start_conversation(config)
        logger.info(f"conversation_manager.start_conversation completed. Success: {result.success}")
        
        if result.success:
            # Convert messages to JSON-serializable format
            messages_data = []
            for msg in result.messages:
                messages_data.append({
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp,
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
    logger.info("🔄 SSE connection established - starting event stream")
    
    def generate_events():
        last_wait_log = datetime.now()
        message_count = 0
        while True:
            try:
                # Get the next message from the queue
                current_time = datetime.now()
                if (current_time - last_wait_log).total_seconds() >= 10:
                    logger.info(f"📥 Waiting for message from queue... [{current_time.strftime('%H:%M:%S')}]")
                    last_wait_log = current_time
                
                message_data = conversation_progress.get(timeout=1)
                message_count += 1
                logger.info(f"📤 Sending SSE event #{message_count}: {message_data}")
                yield f"data: {json.dumps(message_data)}\n\n"
            except queue.Empty:
                pass # No new messages, continue waiting
            except Exception as e:
                logger.error(f"❌ Error generating SSE event: {e}")
                break # Exit the loop on error

    return Response(generate_events(), mimetype='text/event-stream')

@app.route('/api/test-sse', methods=['POST'])
def test_sse():
    """Test SSE connection by sending a test event"""
    try:
        test_event = {
            'type': 'test_sse',
            'message': 'SSE connection test successful',
            'timestamp': datetime.now().isoformat(),
            'data': {'test': True}
        }
        conversation_progress.put(test_event)
        logger.info("🧪 Test SSE event sent successfully")
        return jsonify({'success': True, 'message': 'Test event sent'}), 200
    except Exception as e:
        logger.error(f"Error sending test SSE event: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/conversation-status', methods=['GET'])
def get_conversation_status():
    """Get current conversation status for polling"""
    try:
        # Check if there are any recent updates in the queue
        try:
            # Try to get the latest message from the queue without blocking
            latest_message = conversation_progress.get_nowait()
            logger.info(f"📊 Returning conversation status update: {latest_message['type']}")
            return jsonify({
                'has_updates': True,
                'type': latest_message['type'],
                'message': latest_message['message'],
                'content': latest_message.get('data', {}).get('content', ''),
                'timestamp': latest_message['timestamp']
            }), 200
        except queue.Empty:
            # No updates available
            return jsonify({
                'has_updates': False,
                'message': 'No updates available'
            }), 200
    except Exception as e:
        logger.error(f"Error getting conversation status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host='127.0.0.1', port=5000)
