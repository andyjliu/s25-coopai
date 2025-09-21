from abc import ABC, abstractmethod
import time
from typing import List, Optional, Dict, TypedDict, Any
import logging
from tqdm import tqdm
from utils import gpus_needed
import os

from openai import OpenAI, APIError
from anthropic import Anthropic, APIConnectionError, RateLimitError, APIStatusError
from google import genai
from google.genai import types

import pdb

logger = logging.getLogger(__name__)

class Message(TypedDict):
    role: str
    content: str

class ModelWrapper(ABC):
    """Abstract base class for model API wrappers."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            max_retries: Maximum number of retry attempts
            initial_retry_delay: Initial delay between retries (seconds)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.additional_params = kwargs

    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """Generate a response for a single prompt."""
        pass

    @abstractmethod
    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        """Generate responses for multiple prompts."""
        pass

    @classmethod
    def create(cls, model_name: str, **kwargs) -> 'ModelWrapper':
        """Factory method to create appropriate model wrapper instance."""
        if model_name in OpenAIReasoningClient.REASONING_MODELS:
            return OpenAIReasoningClient(model_name, **kwargs)
        elif "gpt" in model_name.lower():
            return OpenAIClient(model_name, **kwargs)
        elif "claude" in model_name.lower():
            return AnthropicClient(model_name, **kwargs)
        elif "gemini" in model_name.lower():
            return GoogleGeminiClient(model_name, **kwargs)
        elif "openrouter" in model_name.lower():
            return OpenRouterClient(model_name, **kwargs)
        elif "lambda" in model_name.lower():
            return LambdaClient(model_name, **kwargs)
        else:
            return VLLMClient(model_name, **kwargs)

    def _exponential_backoff(self, attempt: int) -> None:
        """Implement exponential backoff between retries."""
        if attempt < self.max_retries:
            delay = self.initial_retry_delay * (2 ** attempt)
            time.sleep(delay)

class OpenAIClient(ModelWrapper):
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = OpenAI()
    
    def generate(self, messages: List[Message]) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    top_p=self.top_p,
                    **self.additional_params
                )
                return response.choices[0].message.content
            except APIError as e:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {str(e)} for prompt {messages}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return None
                self._exponential_backoff(attempt)

    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        #TODO: implement batch API for message_lists that are sufficiently long
        responses = []
        for messages in tqdm(messages_list, desc='Batch Generation'):
            responses.append(self.generate(messages))
        return responses

class OpenAIReasoningClient(ModelWrapper):
    REASONING_MODELS = ['o1', 'o1-pro', 'o3', 'o3-mini', 'o4-mini']
    
    def __init__(self, model_name: str, **kwargs):
        if model_name not in self.REASONING_MODELS:
            raise ValueError(f"Model {model_name} is not a supported reasoning model. Supported models are: {self.REASONING_MODELS}")
        
        # Override default parameters with reasoning-specific ones
        kwargs['reasoning_effort'] = kwargs.get('reasoning_effort', 'medium')
        kwargs['response_format'] = kwargs.get('response_format', {'type': 'text'})
        
        super().__init__(model_name, **kwargs)
        self.max_tokens = 32768
        self.client = OpenAI()
    
    def generate(self, messages: List[Message]) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                    **self.additional_params
                )
                return response.choices[0].message.content
            except APIError as e:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}/{self.max_retries}): {str(e)} for prompt {messages}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return None
                self._exponential_backoff(attempt)

    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        responses = []
        for messages in tqdm(messages_list, desc='Batch Generation'):
            responses.append(self.generate(messages))
        return responses

class LambdaClient(ModelWrapper):
    """
    A ModelWrapper for Lambda-based models
    """
    DEEPSEEK_MODELS = {"lambda-deepseek-r1": "deepseek-r1-671b", "lambda-deepseek-v3": "deepseek-v3-0324"}
    
    def __init__(
        self,
        model_name: str,
        **kwargs
    ):
        """
        :param model_name: e.g. "deepseek/deepseek-r1:free" or "deepseek/deepseek-v3:free"
        :param api_key:    your LAMBDA_API_KEY (will fall back to env var if omitted)
        :param kwargs: other hyperparams (temperature, max_tokens, top_p, max_retries, additional_params) 
        """
        model_name = self.DEEPSEEK_MODELS.get(model_name, model_name)
        kwargs['reasoning_effort'] = kwargs.get('reasoning_effort', 'low')

        super().__init__(model_name, **kwargs)
        self.api_key       = os.getenv("LAMBDA_API_KEY")
        self.base_url      = "https://api.lambda.ai/v1"
        self.max_tokens = 32768

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def generate(self, messages: List[Message]) -> str:
        """
        Generate a completion from the list of messages.
        Returns the assistant's reply (string) or None on failure.
        """
        messages[0]['content'] = "**IMPORTANT RULES:** Do not reason too long about your response. Providing a reasonable response in the correct format is the most important thing!\n\n" + messages[0]['content']
        messages[0]['content'] += "\n\nAgain, please be concise and efficient in your reasoning and response."
            
        if "plain string" in messages[0]['content']:
            response_format = {"type": "text"}
        elif "Example format:" and "JSON" in messages[0]['content']:
            response_format = {"type": "json_object"}

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    top_p=self.top_p,
                    response_format=response_format,
                    **self.additional_params
                )
                return response.choices[0].message.content
            except APIError as e:
                logger.warning(
                    f"[Lambda] API error (attempt {attempt+1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    logger.error(f"[Lambda] giving up after {self.max_retries} tries")
                    return None
                self._exponential_backoff(attempt)
            except Exception as e:
                # catch anything else (network, serialization, etc.)
                logger.error(f"[Lambda] unexpected error: {e}", exc_info=True)
                return None
            
    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        responses = []
        for messages in tqdm(messages_list, desc='Batch Generation'):
            responses.append(self.generate(messages))
        return responses
        
class OpenRouterClient(ModelWrapper):
    """
    A ModelWrapper for OpenRouter-based models (e.g. deepseek/deepseek-r1:free or deepseek/deepseek-v3:free).
    """
    DEEPSEEK_MODELS = {"openrouter-deepseek-r1": "deepseek/deepseek-r1:free", "openrouter-deepseek-v3": "deepseek/deepseek-v3:free"}
    
    def __init__(
        self,
        model_name: str,
        extra_headers: Dict[str, str] = None,
        extra_body: Dict[str, Any]    = None,
        **kwargs
    ):
        """
        :param model_name: e.g. "deepseek/deepseek-r1:free" or "deepseek/deepseek-v3:free"
        :param api_key:    your OPENROUTER_API_KEY (will fall back to env var if omitted)
        :param base_url:   the OpenRouter base URL (defaults to "https://openrouter.ai/api/v1")
        :param extra_headers: optional HTTP headers (e.g. {"HTTP-Referer": ..., "X-Title": ...})
        :param extra_body:    optional extra body to pass through
        :param kwargs: other hyperparams (temperature, max_tokens, top_p, max_retries, additional_params) 
        """
        # Override default parameters with reasoning-specific ones
        model_name = self.DEEPSEEK_MODELS.get(model_name, model_name)

        super().__init__(model_name, **kwargs)
        self.api_key       = os.getenv("OPENROUTER_API_KEY")
        self.base_url      = "https://openrouter.ai/api/v1"
        self.extra_headers = extra_headers or {}
        self.extra_body    = extra_body.update({"reasoning": {"effort": "low", "exclude": False}}) if extra_body else {"reasoning": {"effort": "low", "exclude": False}}
        self.max_tokens = 32768

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def generate(self, messages: List[Message]) -> str:
        """
        Generate a completion from the list of messages.
        Returns the assistant's reply (string) or None on failure.
        """
        messages[0]['content'] = "**IMPORTANT RULES:** Do not think too hard about your response. Providing a response quickly in the correct format is the most important thing!\n\n" + messages[0]['content']
        messages[0]['content'] += "\n\nAgain, limit your thinking and just go with your gut instinct."
        
        # Copy the extra body from the constructor
        extra_body = self.extra_body.copy()

        if "Example format:" and "JSON" in messages[0]['content']:
            extra_body["response_format"] = {"type": "json_object"}
            
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    extra_headers=self.extra_headers,
                    extra_body=extra_body,
                    **self.additional_params
                )
                return response.choices[0].message.content
            except APIError as e:
                logger.warning(
                    f"[OpenRouter] API error (attempt {attempt+1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    logger.error(f"[OpenRouter] giving up after {self.max_retries} tries")
                    return None
                self._exponential_backoff(attempt)
            except Exception as e:
                # catch anything else (network, serialization, etc.)
                logger.error(f"[OpenRouter] unexpected error: {e}", exc_info=True)
                return None
            
    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        responses = []
        for messages in tqdm(messages_list, desc='Batch Generation'):
            responses.append(self.generate(messages))
        return responses
            
class AnthropicClient(ModelWrapper):
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = Anthropic()
    
    def generate(self, messages: List[Message]) -> str:

        if messages[0]['role'] == 'system':
            system = messages[0]['content']
            messages = messages[1:]
        else:
            system = 'You are a helpful assistant.'

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    **self.additional_params
                )
                return response.content[0].text

            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                logger.warning(f"Anthropic API error (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)} for prompt {messages}")
                    return None
                self._exponential_backoff(attempt)

    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        #TODO: implement batch API for message_lists that are sufficiently long
        responses = []
        for messages in tqdm(messages_list, desc='Batch Generation'):
            responses.append(self.generate(messages))
        return responses
        
class VLLMClient(ModelWrapper):
    
    def __init__(self, model_name: str, **kwargs):
        from vllm import LLM, SamplingParams
        
        super().__init__(model_name, **kwargs)
        try:
            self.llm = LLM(
                model=model_name, 
                gpu_memory_utilization=0.9, 
                tensor_parallel_size = gpus_needed(model_name),
                max_model_len=4096)

            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                **self.additional_params
            )
        except Exception as e:
            raise Exception(f"Failed to initialize vLLM model: {str(e)}")

    def format_messages_for_llama(self, messages: List[Message]) -> str:
        formatted_msg = []
        for message in messages:
            formatted_msg.append(f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content']}<|eot_id|>")
        formatted_str = '<|begin_of_text|>' + '\n\n'.join(formatted_msg)

        if messages[-1]['role'] == 'assistant':
            formatted_str += f"<|start_header_id|>user<|end_header_id|>\n\n"
        else:
            formatted_str += f"<|start_header_id|>assistant<|end_header_id|>\n\n"

        return(formatted_str)

    def generate(self, messages: List[Message]) -> str:
        formatted_msg = self.format_messages_for_llama(messages)
        print(formatted_msg)
        response = self.llm.generate([formatted_msg], sampling_params=self.sampling_params)
        return response[0].outputs[0].text

    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        formatted_msgs = [self.format_messages_for_llama(messages) for messages in messages_list]
        response = self.llm.generate(formatted_msgs, sampling_params=self.sampling_params)
        return [out.outputs[0].text for out in response]

class GoogleGeminiClient(ModelWrapper):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.max_tokens = 32768
    
    def generate(self, messages: List[Message]) -> str:
        prompt = self._format_messages(messages)
        attempt = 0
        while attempt < self.max_retries:
            try:
                response = self.client.models.generate_content(
                    model = self.model_name,
                    contents = prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                        top_p=self.top_p,
                        **self.additional_params
                    )
                )
                return response.text
            except Exception as e:
                logger.warning(f"Gemini API error (attempt {attempt + 1}/{self.max_retries}): {str(e)} for prompt {messages}")
                print(f"Gemini API error (attempt {attempt + 1}/{self.max_retries}): {str(e)} for prompt {messages}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {str(e)}")
                    print(f"Failed after {self.max_retries} attempts: {str(e)}")
                    return None
                self._exponential_backoff(attempt)
                attempt += 1
        return ""
    
    def batch_generate(self, messages_list: List[List[Message]]) -> List[str]:
        responses = []
        for messages in messages_list:
            responses.append(self.generate(messages))
        return responses
    
    def _format_messages(self, messages: List[Message]) -> str:
        return "\n".join([f"{msg['content']}" for msg in messages])
