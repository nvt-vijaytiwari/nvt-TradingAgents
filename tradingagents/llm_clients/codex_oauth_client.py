"""
OpenAI Codex OAuth client.

Uses the access token stored by the Codex CLI at ``~/.codex/auth.json`` to
authenticate against the ChatGPT-backed Codex Responses endpoint.

This is *not* the public ``api.openai.com/v1`` API path. Codex with a ChatGPT
account speaks to ``https://chatgpt.com/backend-api/codex/responses`` and
expects streaming Responses API payloads with separate ``instructions`` and
``input`` sections.
"""

from __future__ import annotations

import base64
import json
import random
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient
from .validators import validate_model

# Path where Codex CLI stores OAuth credentials.
_AUTH_FILE = Path.home() / ".codex" / "auth.json"

# OpenAI OAuth token refresh endpoint.
_REFRESH_URL = "https://auth.openai.com/oauth/token"

# Refresh the token this many seconds before it actually expires.
_REFRESH_BUFFER_SECS = 300  # 5 minutes
_RETRYABLE_HTTP_STATUS_CODES = {500, 502, 503, 504}
_MAX_CODEX_RESPONSE_ATTEMPTS = 4

_lock = threading.Lock()


def _read_auth() -> dict:
    """Read the Codex CLI auth file."""
    if not _AUTH_FILE.exists():
        raise FileNotFoundError(
            f"Codex CLI auth file not found at {_AUTH_FILE}. "
            "Run `codex` and log in first."
        )
    return json.loads(_AUTH_FILE.read_text())


def _write_auth(auth: dict) -> None:
    """Persist updated tokens back to the auth file."""
    _AUTH_FILE.write_text(json.dumps(auth, indent=2))


def _token_expiry(access_token: str) -> float:
    """Return the Unix timestamp when the JWT access token expires."""
    try:
        payload = access_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return float(claims["exp"])
    except Exception:
        # If we cannot decode, treat as already expired so refresh is attempted.
        return 0.0


def _refresh_access_token(auth: dict) -> dict:
    """Use the refresh_token to get a new access_token and persist it."""
    refresh_token = auth["tokens"]["refresh_token"]
    client_id = "app_EMoamEEZ73f0CkXaXp7hrann"  # Codex CLI client ID

    resp = requests.post(
        _REFRESH_URL,
        json={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    auth["tokens"]["access_token"] = data["access_token"]
    if "refresh_token" in data:
        auth["tokens"]["refresh_token"] = data["refresh_token"]
    if "id_token" in data:
        auth["tokens"]["id_token"] = data["id_token"]

    from datetime import datetime, timezone

    auth["last_refresh"] = datetime.now(timezone.utc).isoformat()
    _write_auth(auth)
    return auth


def get_valid_access_token() -> str:
    """Return a non-expired access token, refreshing if necessary."""
    with _lock:
        auth = _read_auth()
        access_token = auth["tokens"]["access_token"]
        expiry = _token_expiry(access_token)

        if time.time() >= expiry - _REFRESH_BUFFER_SECS:
            print("[Codex OAuth] Access token expiring soon - refreshing...")
            auth = _refresh_access_token(auth)
            access_token = auth["tokens"]["access_token"]
            print("[Codex OAuth] Token refreshed successfully.")

        return access_token


def _message_content_to_text(content: Any) -> str:
    """Extract plain text from a LangChain/OpenAI message content block."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in {"text", "input_text", "output_text"}:
                    text = block.get("text") or block.get("content")
                    if text:
                        parts.append(str(text))
                elif block_type == "non_standard" and block.get("value"):
                    parts.append(str(block["value"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def _build_placeholder_user_message() -> dict[str, Any]:
    """Create a minimal user message so the Codex endpoint always receives input."""
    return {
        "role": "user",
        "content": [{"type": "input_text", "text": "Proceed."}],
    }


def _split_codex_instructions_and_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Split system/developer text into instructions and keep the rest as input."""
    instructions: list[str] = []
    user_messages: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        if role in {"system", "developer"}:
            text = _message_content_to_text(message.get("content"))
            if text:
                instructions.append(text)
        else:
            user_messages.append(message)

    # Plain string prompts arrive as a single user message. If we do not have
    # explicit instructions, reuse that prompt as the instruction block and keep
    # a tiny user message so the endpoint still has an input turn.
    if not instructions and user_messages and user_messages[0].get("role") == "user":
        prompt_text = _message_content_to_text(user_messages[0].get("content"))
        if prompt_text:
            instructions.append(prompt_text)
            user_messages = user_messages[1:]

    if not instructions:
        instructions.append("You are a helpful assistant.")

    if not user_messages:
        user_messages = [_build_placeholder_user_message()]

    return "\n\n".join(instructions), user_messages


def _iter_codex_events(response: requests.Response) -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield SSE events from a streaming Codex response."""
    event_name: str | None = None
    data_lines: list[str] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue

        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8", errors="replace")

        line = raw_line.strip("\r")
        if not line:
            if event_name and data_lines:
                payload = json.loads("\n".join(data_lines))
                yield event_name, payload
            event_name = None
            data_lines = []
            continue

        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())


def _parse_codex_stream(response: requests.Response) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Convert the Codex SSE stream into a plain text result and tool calls."""
    text_items: dict[int, str] = {}
    tool_calls: dict[int, dict[str, Any]] = {}
    response_metadata: dict[str, Any] = {}

    for _event_name, payload in _iter_codex_events(response):
        event_type = payload.get("type")

        if event_type == "response.output_item.done":
            item = payload.get("item") or {}
            output_index = payload.get("output_index")

            if item.get("type") == "message":
                text = "".join(
                    block.get("text", "")
                    for block in item.get("content", [])
                    if isinstance(block, dict)
                    and block.get("type") in {"output_text", "text"}
                ).strip()
                if text and output_index is not None:
                    text_items[int(output_index)] = text

            elif item.get("type") == "function_call":
                raw_args = item.get("arguments") or ""
                try:
                    parsed_args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    parsed_args = {"__raw_arguments__": raw_args}

                if output_index is not None:
                    tool_calls[int(output_index)] = {
                        "name": item.get("name", ""),
                        "args": parsed_args,
                        "id": item.get("call_id") or item.get("id"),
                        "type": "tool_call",
                    }

        elif event_type == "response.completed":
            response_obj = payload.get("response") or {}
            response_metadata = {
                "response_id": response_obj.get("id"),
                "model": response_obj.get("model"),
                "usage": response_obj.get("usage"),
                "service_tier": response_obj.get("service_tier"),
            }

        elif event_type == "response.failed":
            error = payload.get("response", {}).get("error") or payload.get("error") or payload
            raise RuntimeError(f"Codex response failed: {error}")

    content = "\n".join(text_items[index] for index in sorted(text_items))
    ordered_tool_calls = [tool_calls[index] for index in sorted(tool_calls)]
    return content, ordered_tool_calls, response_metadata


def _retry_delay_seconds(attempt: int, response: requests.Response | None = None) -> float:
    """Compute a short backoff delay for transient Codex backend failures."""
    retry_after = None
    if response is not None:
        retry_after = response.headers.get("Retry-After")

    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass

    base_delay = min(30.0, 2**attempt)
    return base_delay + random.uniform(0.1, 0.5)


class CodexOAuthChatOpenAI(ChatOpenAI):
    """ChatOpenAI wrapper that talks to the ChatGPT-backed Codex endpoint."""

    def invoke(self, input, config=None, **kwargs):
        # Keep the token fresh and store it on the instance for transparency.
        self.openai_api_key = get_valid_access_token()

        stop = kwargs.pop("stop", None)
        payload = self._get_request_payload(input, stop=stop, **kwargs)
        # LangChain's structured-output wrapper injects internal metadata that
        # the Codex backend does not understand. Drop it before serializing.
        payload.pop("ls_structured_output_format", None)
        instructions, codex_input = _split_codex_instructions_and_input(
            payload.pop("input", [])
        )
        payload["instructions"] = instructions
        payload["input"] = codex_input
        payload["store"] = False
        payload["stream"] = True

        response = None
        last_error = None
        for attempt in range(_MAX_CODEX_RESPONSE_ATTEMPTS):
            try:
                response = requests.post(
                    f"{(getattr(self, 'openai_api_base', None) or 'https://chatgpt.com/backend-api/codex').rstrip('/')}/responses",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                    json=payload,
                    stream=True,
                    timeout=(15, 300),
                )
                response.raise_for_status()
                break
            except requests.HTTPError as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code in _RETRYABLE_HTTP_STATUS_CODES and attempt < _MAX_CODEX_RESPONSE_ATTEMPTS - 1:
                    delay = _retry_delay_seconds(attempt, getattr(exc, "response", None))
                    time.sleep(delay)
                    last_error = exc
                    continue
                raise
            except requests.RequestException as exc:
                if attempt < _MAX_CODEX_RESPONSE_ATTEMPTS - 1:
                    delay = _retry_delay_seconds(attempt)
                    time.sleep(delay)
                    last_error = exc
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error

        content, tool_calls, response_metadata = _parse_codex_stream(response)
        return AIMessage(
            content=content,
            tool_calls=tool_calls,
            response_metadata=response_metadata,
        )

    def with_structured_output(self, schema, *, method=None, **kwargs):
        if method is None:
            method = "function_calling"
        return super().with_structured_output(schema, method=method, **kwargs)


class CodexOAuthClient(BaseLLMClient):
    """LLM client that uses Codex CLI OAuth - no API key needed."""

    OPENAI_BASE_URL = "https://chatgpt.com/backend-api/codex"

    def validate_model(self) -> bool:
        """Validate models that are known to work with ChatGPT-backed Codex."""
        return validate_model("codex", self.model)

    def get_llm(self) -> Any:
        access_token = get_valid_access_token()

        return CodexOAuthChatOpenAI(
            model=self.model,
            base_url=self.OPENAI_BASE_URL,
            api_key=access_token,
            temperature=0,
            streaming=True,
            store=False,
            use_responses_api=True,
        )
