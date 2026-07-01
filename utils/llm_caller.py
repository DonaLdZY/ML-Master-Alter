import os
from typing import List, Dict, Union, Optional, Any
from openai import OpenAI
import time
import logging
logger = logging.getLogger("ml-master")
NETWORK_RETRY_MAX_SLEEP_SECONDS = 30.0


def _is_deepseek_model(model_name: str) -> bool:
    return str(model_name or "").strip().lower().startswith("deepseek")


def _normalize_reasoning_effort(effort: str | None) -> str | None:
    if effort is None:
        return None
    text = str(effort).strip().lower()
    if text in {"", "default", "none", "null"}:
        return None
    return {"low": "high", "medium": "high", "xhigh": "max"}.get(text, text)


def _openai_reasoning_effort(effort: str | None) -> str | None:
    text = _normalize_reasoning_effort(effort)
    if text == "max":
        return "high"
    return text


def _is_retryable_llm_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    if any(
        key in name
        for key in [
            "timeout",
            "connection",
            "ratelimit",
            "internalserver",
            "apierror",
            "apiconnection",
            "badgateway",
            "serviceunavailable",
            "gateway",
            "httpstatus",
        ]
    ):
        return True
    return any(
        key in msg
        for key in [
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "connection refused",
            "10061",
            "actively refused",
            "积极拒绝",
            "temporary failure",
            "temporarily unavailable",
            "bad gateway",
            "502",
            "503",
            "504",
            "rate limit",
            "too many requests",
            "getaddrinfo",
            "11001",
            "name resolution",
            "name or service not known",
            "server disconnected",
            "remote protocol error",
        ]
    )


def _retry_sleep(attempt: int, default_delay: float) -> float:
    return min(NETWORK_RETRY_MAX_SLEEP_SECONDS, max(float(default_delay), 5.0 * attempt))


class LLM:
    """
    Encapsulate the VLLM-based LLM class to invoke the self-hosted VLLM model via the OpenAI SDK.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy-key",
        model_name: str = "default-model",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        retry_time: int = 5,
        delay_time: int = 3,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """
        Initialize the VLLM LLM class.
        
        Args:
            base_url: The URL of the VLLM service.
            api_key: API key (generally not important when self-hosted).
            model_name: Name of the model (generally not important when self-hosted).
            temperature: Temperature parameter to control output randomness.
            max_tokens: Maximum number of tokens to generate.
            stop_tokens: List of tokens that signal the end of generation.
        """
        self.base_url = base_url
        self.api_key = api_key

        # ================= [新增代码: DeepSeek URL 修正] =================
        # 当检测到配置为 DeepSeek v1 或根域名时，强制指向 beta 接口以支持 reasoning 参数
        if self.base_url.rstrip("/") in ["https://api.deepseek.com/v1", "https://api.deepseek.com"]:
            logger.info(f"Redirecting DeepSeek API to Beta endpoint: {self.base_url} -> https://api.deepseek.com/beta")
            self.base_url = "https://api.deepseek.com/beta"
        # ==============================================================

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else None
        self.stop_tokens = stop_tokens
        self.retry_time = retry_time
        self.delay_time = delay_time
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        
        # initalize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _resolved_max_tokens(self, max_tokens: Optional[int]) -> Optional[int]:
        try:
            tokens = int(max_tokens) if max_tokens is not None else 0
        except (TypeError, ValueError):
            tokens = 0
        if tokens > 0:
            return tokens
        return self.max_tokens

    @staticmethod
    def _maybe_set_max_tokens(params: Dict[str, Any], tokens: Optional[int]) -> None:
        if tokens is not None and tokens > 0:
            params["max_tokens"] = tokens
    
    def generate(
        self, 
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Generate text
        
        Args:
            messages: List of conversation messages
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop sequences
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the OpenAI API
            
        Returns:
            If stream=False, returns the generated text  
            If stream=True, returns the streaming response object
        """
        
        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = self._resolved_max_tokens(max_tokens)
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        
        # create request parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "stream": stream,
            **kwargs
        }
        self._maybe_set_max_tokens(params, tokens)
        extra_body = self._chat_extra_body()
        if extra_body:
            params["extra_body"] = extra_body
        
        # add stop_tokens
        if stops is not None:
            params["stop"] = stops
            
        attempt = 0
        while attempt < self.retry_time:
            try:
                if  "gpt-5" in self.model_name:
                    response_params = self._responses_params(params["messages"])
                    response = self.client.responses.create(**response_params)
                    return response.output_text
                else:
                    response = self.client.chat.completions.create(**params)
                    return response.choices[0].message
            except Exception as e:
                attempt += 1
                retryable = _is_retryable_llm_error(e)
                logger.warning(f"calling llm failed, retryable={retryable}, attempt {attempt}/{self.retry_time}, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                if not retryable:
                    raise e
                time.sleep(_retry_sleep(attempt, self.delay_time))

    def stream_generate(
        self, 
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        delay_time:int = 1,
        **kwargs
    ) -> Union[str, Any]:
        """
        Generate text
        
        Args:
            messages: List of conversation messages
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop sequences
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the OpenAI API
            
        Returns:
            If stream=False, returns the generated text  
            If stream=True, returns the streaming response object
        """
        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = self._resolved_max_tokens(max_tokens)
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        stream = True
        
        # create request parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
            "stream": stream,
            **kwargs
        }
        self._maybe_set_max_tokens(params, tokens)
        extra_body = self._chat_extra_body()
        if extra_body:
            params["extra_body"] = extra_body
        
        # add stop_tokens
        if stops is not None:
            params["stop"] = stops
        attempt = 0
        while attempt < self.retry_time:
            try:
                if  "gpt-5" in self.model_name:
                    response_params = self._responses_params(params["messages"], stream=True)
                    response = self.client.responses.create(**response_params)
                    full_text = ""
                    for chunk in response:
                        if chunk.type=="response.output_text.delta":
                            full_text += chunk.delta
                    return full_text
                else:
                    response = self.client.chat.completions.create(**params)

                    full_text = ""
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            full_text += chunk.choices[0].delta.content
                    return full_text
            except Exception as e:
                attempt += 1
                retryable = _is_retryable_llm_error(e)
                logger.warning(f"calling llm failed, retryable={retryable}, attempt {attempt}/{self.retry_time}, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                if not retryable:
                    raise e
                time.sleep(_retry_sleep(attempt, self.delay_time))

    def _chat_extra_body(self) -> Dict[str, Any]:
        """Build provider extra_body only when the user selected a non-default thinking mode."""
        if self.enable_thinking is None:
            return {}
        if _is_deepseek_model(self.model_name):
            body: Dict[str, Any] = {"thinking": {"type": "enabled" if self.enable_thinking else "disabled"}}
            effort = _normalize_reasoning_effort(self.reasoning_effort)
            if self.enable_thinking and effort:
                body["reasoning_effort"] = effort
            return body
        return {
            "chat_template_kwargs": {"thinking": bool(self.enable_thinking)},
            "separate_reasoning": bool(self.enable_thinking),
        }

    def _responses_params(self, messages: List[Dict[str, str]], *, stream: bool = False) -> Dict[str, Any]:
        instructions = messages[0]["content"] if messages else ""
        user_input = messages[1]["content"] if len(messages) > 1 else ""
        params: Dict[str, Any] = {
            "model": self.model_name,
            "instructions": instructions,
            "input": user_input,
            "text": {"verbosity": "medium"},
        }
        effort = _openai_reasoning_effort(self.reasoning_effort)
        if self.enable_thinking is not None and effort:
            params["reasoning"] = {"effort": effort}
        if stream:
            params["stream"] = True
        return params
    
    def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Text Completion

        Args:
            prompt: Text prompt
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop tokens
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the OpenAI API

        Returns:
            If stream=False, returns the generated text  
            If stream=True, returns a streaming response object
        """

        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = self._resolved_max_tokens(max_tokens)
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        
        # create request parameters
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temp,
            "stream": stream,
            **kwargs
        }
        self._maybe_set_max_tokens(params, tokens)
        
        # add stop_tokens
        if stops is not None:
            params["stop"] = stops
            
        attempt = 0
        while attempt < self.retry_time:
            try:
                response = self.client.completions.create(**params)
                break
            except Exception as e:
                attempt += 1
                retryable = _is_retryable_llm_error(e)
                logger.warning(f"calling llm failed, retryable={retryable}, attempt {attempt}/{self.retry_time}, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                if not retryable:
                    raise e
                time.sleep(_retry_sleep(attempt, self.delay_time))
        
        if stream:
            return response
        
        return response.choices[0].text
    
    def stream_complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Union[str, Any]:
        """
        Text Completion

        Args:
            prompt: Text prompt
            temperature: Overrides the default temperature parameter
            max_tokens: Overrides the default maximum number of tokens
            stop_tokens: Overrides the default stop tokens
            stream: Whether to use streaming output
            **kwargs: Additional parameters passed to the OpenAI API

        Returns:
            If stream=False, returns the generated text  
            If stream=True, returns a streaming response object
        """

        # use function parameters or default values
        temp = temperature if temperature is not None else self.temperature
        tokens = self._resolved_max_tokens(max_tokens)
        stops = stop_tokens if stop_tokens is not None else self.stop_tokens
        stream = True
        # create request parameters
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temp,
            "stream": stream,
            **kwargs
        }
        self._maybe_set_max_tokens(params, tokens)
        
        # add stop_tokens
        if stops is not None:
            params["stop"] = stops
        
        attempt = 0
        while attempt < self.retry_time:
            try:
                response = self.client.completions.create(**params)
                
                full_text = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].text is not None:
                        full_text += chunk.choices[0].text
                return full_text
            except Exception as e:
                attempt += 1
                retryable = _is_retryable_llm_error(e)
                logger.warning(f"calling llm failed, retryable={retryable}, attempt {attempt}/{self.retry_time}, error message: {e}")
                if attempt >= self.retry_time:
                    logger.error("LLM call retry limit reached, throwing exception")
                    raise e
                if not retryable:
                    raise e
                time.sleep(_retry_sleep(attempt, self.delay_time))
    
