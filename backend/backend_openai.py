"""Backend for OpenAI API."""

import json
import logging
import time

from backend.backend_utils import (
    FunctionSpec,
    OutputType,
    opt_messages_to_list,
    backoff_create,
)
from funcy import notnone, once, select_values
import openai
from utils.config_mcts import Config

logger = logging.getLogger("ml-master")

_client: openai.OpenAI = None  # type: ignore


def _is_deepseek_model(model_name: str) -> bool:
    return str(model_name or "").strip().lower().startswith("deepseek")


def _normalize_reasoning_effort(effort: str | None) -> str | None:
    if effort is None:
        return None
    text = str(effort).strip().lower()
    if text in {"", "default", "none", "null"}:
        return None
    return {"low": "high", "medium": "high", "xhigh": "max"}.get(text, text)


def _feedback_max_tokens(cfg: Config) -> int | None:
    try:
        tokens = int(getattr(cfg.agent.feedback, "max_tokens", 0) or 0)
    except (TypeError, ValueError):
        return None
    return tokens if tokens > 0 else None


def _feedback_extra_body(cfg: Config, model: str) -> dict:
    enabled = getattr(cfg.agent.feedback, "enable_thinking", None)
    if enabled is None:
        return {}
    if _is_deepseek_model(model):
        body = {"thinking": {"type": "enabled" if enabled else "disabled"}}
        effort = _normalize_reasoning_effort(getattr(cfg.agent.feedback, "reasoning_effort", None))
        if enabled and effort:
            body["reasoning_effort"] = effort
        return body
    return {
        "chat_template_kwargs": {"thinking": bool(enabled)},
        "separate_reasoning": bool(enabled),
    }


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.APIStatusError,
)


@once
def _setup_openai_client(cfg:Config):
    global _client

    # ================= [新增代码: DeepSeek URL 修正] =================
    base_url = cfg.agent.feedback.base_url
    if base_url.rstrip("/") in ["https://api.deepseek.com/v1", "https://api.deepseek.com"]:
        logger.info(f"Redirecting Feedback Backend API to Beta: {base_url} -> https://api.deepseek.com/beta")
        base_url = "https://api.deepseek.com/beta"
    # ==============================================================

    _client = openai.OpenAI(
        base_url=base_url,
        api_key=cfg.agent.feedback.api_key,
        max_retries=0,
    )


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    convert_system_to_user: bool = False,
    cfg:Config=None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openai_client(cfg)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message, convert_system_to_user=convert_system_to_user)
    if not filtered_kwargs.get("max_tokens"):
        configured_max_tokens = _feedback_max_tokens(cfg)
        if configured_max_tokens is not None:
            filtered_kwargs["max_tokens"] = configured_max_tokens

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model the use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    extra_body = _feedback_extra_body(cfg, str(filtered_kwargs.get("model") or cfg.agent.feedback.model))
    if extra_body:
        filtered_kwargs["extra_body"] = extra_body

    t0 = time.time()
    message_print = messages[0]["content"]
    print(f"\033[31m{message_print}\033[0m")
    completion = backoff_create(
        _client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
        print(f"\033[32m{output}\033[0m")
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            output = json.loads(choice.message.tool_calls[0].function.arguments)
            print(f"\033[32m{output}\033[0m")
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
