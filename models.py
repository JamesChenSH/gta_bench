"""
Standalone model implementation for GTA benchmark evaluation.
This version doesn't depend on OpenCompass framework.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
import requests
import os
from openai import AzureOpenAI

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
                    'description': input_param.get('description') if input_param.get('description', None) is not None else f'{param_name} parameter'
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
                elif 'tool▁call' in message['content']:
                    # For R1 format
                    tool_call_str = message['content'].split('<｜tool▁call▁begin｜>')[1].split('<｜tool▁call▁end｜>')[0]

                    if '｜tool▁sep｜' in tool_call_str:
                        tool_call_str = tool_call_str.split('｜tool▁sep｜')[1]
                        # Remove the json``` and ```
                        tool_call_name = tool_call_str.split('```json')[0]
                        tool_call_str = tool_call_str.split('```json')[1].split('```')[0]

                        tool_call_dict = {
                            "name": tool_call_name,
                            "arguments": json.loads(tool_call_str)
                        }
                    else:
                        tool_call_str = tool_call_str.split('```json')[1].split('```')[0]
                        tool_call_dict = json.loads(tool_call_str)

                    message['tool_calls'] = [{
                        "function": tool_call_dict,
                        "id": "call_1"
                    }]
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
            file_info = """
            You are a helpful assistant that can use the following files to answer the user's question. 
            The files included should have enough information to answer the user's question. You should
            try to use the tools provided to provide a response to user's question.

            The tools provided are: 
            {tools}

            These are the files: 
            {files}
            """.format(tools="\n".join(str(tool) for tool in tools), files="\n".join([f"`{f['path']}`" for f in files]))
            messages.append({'role': 'system', 'content': file_info})
        else:
            system_message = """
            You are a helpful assistant that can use the following tools to answer the user's question. You should
            try to use the tools provided to provide a response to user's question.

            The tools provided are: 
            {tools}
            """.format(tools="\n".join(str(tool) for tool in tools))
            messages.append({'role': 'system', 'content': system_message})
        
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
                'tool_calls': message['tool_calls'],
                'content': message.get('content', '')
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


class AzureGTAModel(GTAModel):
    """
    Azure OpenAI model implementation using the official OpenAI SDK.
    """
    
    def __init__(self, 
                 model_name: str = 'gpt-4',
                 api_base: str = None,
                 api_key: str = None,
                 api_version: str = '2024-02-15-preview',
                 deployment_name: str = None,
                 max_tokens: int = 4096,
                 temperature: float = 0.1,
                 timeout: int = 60):
        super().__init__(model_name=model_name,
                        api_base=api_base,
                        api_key=api_key,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout)
        self.api_version = api_version
        self.deployment_name = deployment_name or model_name
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_base
        )
        
    def generate(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """
        Generate response using Azure OpenAI API via the official SDK.
        
        Args:
            messages: List of conversation messages
            tools: Optional list of available tools
            
        Returns:
            Dict containing the response
        """
        try:
            # Prepare completion parameters
            completion_params = {
                'model': self.deployment_name,
                'messages': messages,
                'timeout': self.timeout
            }
            
            # Add tools if provided
            if tools:
                openai_tools = self.convert_tools_to_openai_format(tools)
                completion_params['tools'] = openai_tools
                completion_params['tool_choice'] = 'auto'
            
            if self.deployment_name == 'o4-mini':
                completion_params['max_completion_tokens'] = self.max_tokens
            else:
                completion_params['temperature'] = self.temperature
                completion_params['max_tokens'] = self.max_tokens

            # Make API request using the SDK
            response = self.client.chat.completions.create(**completion_params)
            
            # Extract the message from the response
            choice = response.choices[0]
            message = {
                'role': choice.message.role,
                'content': choice.message.content
            }
            
            # Handle tool calls if present
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                tool_calls = []
                for tool_call in choice.message.tool_calls:
                    tool_calls.append({
                        'id': tool_call.id,
                        'function': {
                            'name': tool_call.function.name,
                            'arguments': tool_call.function.arguments
                        }
                    })
                message['tool_calls'] = tool_calls
            
            return {
                'success': True,
                'message': message,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'model': response.model
            }
                
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {str(e)}")
            return {
                'success': False,
                'error': f'Azure OpenAI API error: {str(e)}'
            }


def create_model_from_config(config_file: str) -> GTAModel:
    """Create model instance from configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        if not config.get('models'):
            raise ValueError("No models defined in config")
            
        model_config = config['models'][0]  # Use first model config
        model_type = model_config.get('type', 'default')
        
        # Common parameters
        model_params = {
            'model_name': model_config.get('model_name'),
            'api_base': model_config.get('api_base'),
            'api_key': model_config.get('api_key', os.getenv('OPENAI_API_KEY')),
            'max_tokens': model_config.get('max_tokens', 4096),
            'temperature': model_config.get('temperature', 0.1),
            'timeout': model_config.get('timeout', 60)
        }
        
        if model_type == 'azure':
            # Azure-specific parameters
            azure_params = {
                'api_version': model_config.get('api_version', '2024-02-15-preview'),
                'deployment_name': model_config.get('deployment_name')
            }
            model_params.update(azure_params)
            return AzureGTAModel(**model_params)
        else:
            return GTAModel(**model_params)
            
    except Exception as e:
        logger.error(f"Failed to create model from config: {e}")
        raise 