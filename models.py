"""
Standalone model implementation for GTA benchmark evaluation.
This version doesn't depend on OpenCompass framework.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)


class GTAModel:
    """
    Standalone model implementation that uses proper OpenAI tool call API structure.
    No dependency on OpenCompass framework.
    """
    
    def __init__(self, 
                 model_name: str = 'qwen3-8b',
                 api_base: str = 'http://0.0.0.0:8080/v1',
                 api_key: str = 'EMPTY',
                 max_tokens: int = 4096,
                 temperature: float = 0.1,
                 timeout: int = 60):
        self.model_name = model_name
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
    def convert_tools_to_openai_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert GTA tool metadata to OpenAI function calling format."""
        openai_tools = []
        
        for tool in tools:
            # Build function parameters schema
            properties = {}
            required = []
            
            for input_param in tool.get('inputs', []):
                param_name = input_param['name']
                param_type = input_param['type']
                
                # Map tool types to JSON schema types
                if param_type == 'text':
                    schema_type = 'string'
                elif param_type == 'int':
                    schema_type = 'integer'
                elif param_type == 'bool':
                    schema_type = 'boolean'
                elif param_type == 'image':
                    schema_type = 'string'  # Image paths as strings
                else:
                    schema_type = 'string'  # Default to string
                
                properties[param_name] = {
                    'type': schema_type,
                    'description': input_param.get('description', f'{param_name} parameter')
                }
                
                if not input_param.get('optional', False):
                    required.append(param_name)
            
            openai_tool = {
                'type': 'function',
                'function': {
                    'name': tool['name'],
                    'description': tool['description'],
                    'parameters': {
                        'type': 'object',
                        'properties': properties,
                        'required': required
                    }
                }
            }
            openai_tools.append(openai_tool)
            
        return openai_tools
    
    def generate(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """
        Generate response using proper OpenAI tool call API.
        
        Args:
            messages: List of conversation messages
            tools: Optional list of available tools
            
        Returns:
            Dict containing the response
        """
        # Prepare request payload
        payload = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
        }
        
        # Add tools if provided
        if tools:
            openai_tools = self.convert_tools_to_openai_format(tools)
            payload['tools'] = openai_tools
            payload['tool_choice'] = 'auto'
        
        # Make API request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        try:
            response = requests.post(
                f'{self.api_base}/chat/completions',
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                message = choice['message']

                if 'tool_call' in message['content']:
                    message['tool_calls'] = [
                        {
                            "function": json.loads(message['content'].split('<tool_call>')[1].split('</tool_call>')[0]),
                            "id": "call_1"
                        }
                    ]
                else:
                    message['tool_calls'] = []

                return {
                    'success': True,
                    'message': message,
                    'usage': result.get('usage', {}),
                    'model': result.get('model', self.model_name)
                }
            else:
                logger.error(f"Unexpected API response format: {result}")
                return {
                    'success': False,
                    'error': 'Unexpected response format',
                    'raw_response': result
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {
                'success': False,
                'error': f'API request failed: {str(e)}'
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            return {
                'success': False,
                'error': f'Failed to parse response: {str(e)}'
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }


class GTAAgent:
    """
    Standalone agent that handles multi-turn conversations with proper tool calling.
    No dependency on OpenCompass framework.
    """
    
    def __init__(self, model: GTAModel, max_turns: int = 10):
        self.model = model
        self.max_turns = max_turns
        
    def chat(self, query: str, tools: List[Dict], files: List[Dict] = None) -> List[Dict]:
        """
        Conduct a multi-turn conversation with tool usage.
        
        Args:
            query: User query
            tools: List of available tools
            files: List of available files
            
        Returns:
            List of conversation steps
        """
        conversation = []
        messages = []
        
        # Add system message if files are provided
        if files:
            file_info = "For image related tools, you can pass in the image path as a string for them to read them. These are the paths to the images related to the conversation: " + ", ".join([f"`{f['path']}`" for f in files])
            messages.append({'role': 'system', 'content': file_info})
        
        # Add user query
        messages.append({'role': 'user', 'content': query})
        conversation.append({'role': 'user', 'content': query})
        
        for turn in range(self.max_turns):
            # Generate response
            response = self.model.generate(messages, tools)
            
            if not response['success']:
                # Handle error
                error_msg = {
                    'role': 'assistant', 
                    'content': f"Error: {response['error']}",
                    'error': True
                }
                conversation.append(error_msg)
                break
            
            message = response['message']
            
            if 'tool_calls' in message and message['tool_calls']:
                # Handle tool call
                tool_call_msg = {
                    'role': 'assistant',
                    'tool_calls': message['tool_calls']
                }
                messages.append(tool_call_msg)
                conversation.append(tool_call_msg)
                
                # Execute tool (simulate for now)
                for tool_call in message['tool_calls']:
                    tool_result = self._execute_tool(tool_call['function'], files)
                    tool_response_msg = {
                        'role': 'tool',
                        'tool_call_id': tool_call.get('id', 'call_1'),
                        'name': tool_call['function']['name'],
                        'content': tool_result
                    }
                    messages.append(tool_response_msg)
                    conversation.append(tool_response_msg)
                
            else:
                # Final text response
                final_msg = {
                    'role': 'assistant', 
                    'content': message.get('content', '')
                }
                messages.append(final_msg)
                conversation.append(final_msg)
                break
                
        return conversation
    
    def predict_single_step(self, chat_history: List[Dict], tools: List[Dict], files: List[Dict] = None) -> Dict:
        """
        Predict a single step given the chat history for step-by-step evaluation.
        
        Args:
            chat_history: Pre-built chat history up to the current step
            tools: List of available tools
            files: List of available files
            
        Returns:
            Single assistant response (tool call or text)
        """
        messages = []
        
        # Add system message if files are provided
        if files:
            file_info = "You are a helpful assistant that can use the following files to answer the user's question. These are the files: " + ", ".join([f"`{f['path']}`" for f in files])
            messages.append({'role': 'system', 'content': file_info})
        
        # Add the provided chat history
        messages.extend(chat_history)
        
        # Generate single response
        response = self.model.generate(messages, tools)
        
        if not response['success']:
            # Handle error
            return {
                'role': 'assistant',
                'error': {
                    'type': 'API_ERROR',
                    'msg': response['error']
                }
            }
        
        message = response['message']
        
        if 'tool_calls' in message and message['tool_calls']:
            # Return tool call message
            return {
                'role': 'assistant',
                'tool_calls': message['tool_calls']
            }
        else:
            # Return text response
            return {
                'role': 'assistant',
                'content': message.get('content', '')
            }
    
    def _execute_tool(self, function_call: Dict, files: List[Dict] = None) -> str:
        """
        Simulate tool execution. In a real implementation, this would
        call the actual tool server or execute the tool locally.
        """
        tool_name = function_call['name']
        arguments = function_call['arguments']
        
        # For demonstration, return a dummy result
        # In practice, this would call the actual tool implementation
        return f"Tool {tool_name} executed with arguments {arguments}. [Dummy result]"


def create_model_from_config(config_file: str) -> GTAModel:
    """Create a model instance from a configuration file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return GTAModel(
        model_name=config.get('model_alias', 'unknown'),
        api_base=f"http://{config['host']}:{config['port']}/v1",
        api_key='EMPTY',
        max_tokens=config.get('n_ctx', 4096),
        temperature=0.1
    ) 