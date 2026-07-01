import logging
from dataclasses import dataclass
from typing import Callable

import jsonschema
from dataclasses_json import DataClassJsonMixin
import traceback
PromptType = str | dict | list
FunctionCallType = dict
OutputType = str | FunctionCallType

import time

logger = logging.getLogger("ml-master")
NETWORK_RETRY_MAX_ATTEMPTS = 5
NETWORK_RETRY_MAX_SLEEP_SECONDS = 30.0


def _is_retryable_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    try:
        if status_code is not None:
            return int(status_code) in {429, 500, 502, 503, 504}
    except Exception:
        pass
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


def backoff_create(
    create_fn: Callable, retry_exceptions: list[Exception], *args, **kwargs
):
    last_exc: Exception | None = None
    for attempt in range(1, NETWORK_RETRY_MAX_ATTEMPTS + 1):
        try:
            return create_fn(*args, **kwargs)
        except tuple(retry_exceptions) as e:
            last_exc = e
            retryable = _is_retryable_exception(e)
            logger.warning(
                "LLM backend network error; retryable=%s reconnecting attempt %s/%s: %s\n%s",
                retryable,
                attempt,
                NETWORK_RETRY_MAX_ATTEMPTS,
                e,
                traceback.format_exc(),
            )
            if (not retryable) or attempt >= NETWORK_RETRY_MAX_ATTEMPTS:
                logger.error("LLM backend reconnect retry limit reached")
                raise
            time.sleep(min(NETWORK_RETRY_MAX_SLEEP_SECONDS, 5.0 * attempt))
    if last_exc is not None:
        raise last_exc


def opt_messages_to_list(
    system_message: str | None,
    user_message: str | None,
    convert_system_to_user: bool = False,
) -> list[dict[str, str]]:
    messages = []
    if system_message:
        if convert_system_to_user:
            messages.append({"role": "user", "content": system_message})
        else:
            messages.append({"role": "system", "content": system_message})
    if user_message:
        messages.append({"role": "user", "content": user_message})
    return messages


def compile_prompt_to_md(prompt: PromptType, _header_depth: int = 1) -> str:
    if isinstance(prompt, str):
        return prompt.strip() + "\n"
    elif isinstance(prompt, list):
        return "\n".join([f"- {s.strip()}" for s in prompt] + ["\n"])

    out = []
    header_prefix = "#" * _header_depth
    for k, v in prompt.items():
        out.append(f"{header_prefix} {k}\n")
        out.append(compile_prompt_to_md(v, _header_depth=_header_depth + 1))
    return "\n".join(out)


@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict  # JSON schema
    description: str

    def __post_init__(self):
        # validate the schema
        jsonschema.Draft7Validator.check_schema(self.json_schema)

    @property
    def as_openai_tool_dict(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
            "strict": True,
        }

    @property
    def openai_tool_choice_dict(self):
        return {
            "type": "function",
            "function": {"name": self.name},
        }
