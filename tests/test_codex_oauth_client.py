import json
import unittest
from unittest.mock import patch

import requests

from tradingagents.llm_clients.codex_oauth_client import (
    CodexOAuthChatOpenAI,
    _parse_codex_stream,
    _split_codex_instructions_and_input,
)
from tradingagents.agents.schemas import ResearchPlan


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


class CodexOAuthClientTests(unittest.TestCase):
    def test_split_codex_instructions_uses_system_messages(self):
        messages = [
            {"role": "system", "content": "You are a trading analyst."},
            {"role": "user", "content": [{"type": "input_text", "text": "Analyze RELIANCE"}]},
        ]

        instructions, codex_input = _split_codex_instructions_and_input(messages)

        self.assertEqual(instructions, "You are a trading analyst.")
        self.assertEqual(
            codex_input,
            [{"role": "user", "content": [{"type": "input_text", "text": "Analyze RELIANCE"}]}],
        )

    def test_split_codex_instructions_falls_back_for_plain_prompt(self):
        messages = [{"role": "user", "content": "Analyze RELIANCE as of 2026-05-29"}]

        instructions, codex_input = _split_codex_instructions_and_input(messages)

        self.assertEqual(instructions, "Analyze RELIANCE as of 2026-05-29")
        self.assertEqual(
            codex_input,
            [{"role": "user", "content": [{"type": "input_text", "text": "Proceed."}]}],
        )

    def test_parse_codex_stream_extracts_text_and_tool_calls(self):
        text_done = {
            "type": "response.output_item.done",
            "item": {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "content": [
                    {"type": "output_text", "annotations": [], "logprobs": [], "text": "pong"},
                ],
                "phase": "final_answer",
                "role": "assistant",
            },
            "output_index": 0,
            "sequence_number": 7,
        }
        tool_done = {
            "type": "response.output_item.done",
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "status": "completed",
                "arguments": "{\"text\":\"Tool called as requested.\"}",
                "call_id": "call_123",
                "name": "echo",
            },
            "output_index": 1,
            "sequence_number": 13,
        }
        completed = {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-5.4-mini-2026-03-17",
                "service_tier": "default",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        }
        lines = [
            "event: response.output_item.done",
            f"data: {json.dumps(text_done)}",
            "",
            "event: response.output_item.done",
            f"data: {json.dumps(tool_done)}",
            "",
            "event: response.completed",
            f"data: {json.dumps(completed)}",
            "",
        ]

        content, tool_calls, metadata = _parse_codex_stream(_FakeResponse(lines))

        self.assertEqual(content, "pong")
        self.assertEqual(
            tool_calls,
            [
                {
                    "name": "echo",
                    "args": {"text": "Tool called as requested."},
                    "id": "call_123",
                    "type": "tool_call",
                }
            ],
        )
        self.assertEqual(metadata["response_id"], "resp_1")
        self.assertEqual(metadata["model"], "gpt-5.4-mini-2026-03-17")

    def test_invoke_hits_codex_backend_and_parses_message(self):
        text_done = {
            "type": "response.output_item.done",
            "item": {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "content": [
                    {"type": "output_text", "annotations": [], "logprobs": [], "text": "pong"},
                ],
                "phase": "final_answer",
                "role": "assistant",
            },
            "output_index": 0,
            "sequence_number": 7,
        }
        completed = {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-5.4-mini-2026-03-17",
                "service_tier": "default",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        }

        class _FakePostResponse:
            def raise_for_status(self):
                return None

            def iter_lines(self, decode_unicode=True):
                return iter(
                    [
                        "event: response.output_item.done",
                        f"data: {json.dumps(text_done)}",
                        "",
                        "event: response.completed",
                        f"data: {json.dumps(completed)}",
                        "",
                    ]
                )

        with patch(
            "tradingagents.llm_clients.codex_oauth_client.get_valid_access_token",
            return_value="test-token",
        ), patch(
            "tradingagents.llm_clients.codex_oauth_client.requests.post",
            return_value=_FakePostResponse(),
        ) as post_mock:
            llm = CodexOAuthChatOpenAI(
                model="gpt-5.4-mini",
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="test-token",
                streaming=True,
                store=False,
                use_responses_api=True,
                temperature=0,
            )

            message = llm.invoke("Reply with exactly: pong")

        self.assertEqual(message.content, "pong")
        self.assertEqual(message.response_metadata["response_id"], "resp_1")
        self.assertEqual(post_mock.call_args.args[0], "https://chatgpt.com/backend-api/codex/responses")

    def test_invoke_retries_transient_server_error(self):
        text_done = {
            "type": "response.output_item.done",
            "item": {
                "id": "msg_1",
                "type": "message",
                "status": "completed",
                "content": [
                    {"type": "output_text", "annotations": [], "logprobs": [], "text": "pong"},
                ],
                "phase": "final_answer",
                "role": "assistant",
            },
            "output_index": 0,
            "sequence_number": 7,
        }
        completed = {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-5.4-mini-2026-03-17",
                "service_tier": "default",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        }

        class _FakeResponse:
            def __init__(self, status_code, lines):
                self.status_code = status_code
                self._lines = lines
                self.headers = {}
                self.url = "https://chatgpt.com/backend-api/codex/responses"
                self.reason = "Service Unavailable"

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.HTTPError(
                        f"{self.status_code} {self.reason}",
                        response=self,
                    )

            def iter_lines(self, decode_unicode=True):
                return iter(self._lines)

        responses = [
            _FakeResponse(503, []),
            _FakeResponse(
                200,
                [
                    "event: response.output_item.done",
                    f"data: {json.dumps(text_done)}",
                    "",
                    "event: response.completed",
                    f"data: {json.dumps(completed)}",
                    "",
                ],
            ),
        ]

        def fake_post(*args, **kwargs):
            return responses.pop(0)

        with patch(
            "tradingagents.llm_clients.codex_oauth_client.get_valid_access_token",
            return_value="test-token",
        ), patch(
            "tradingagents.llm_clients.codex_oauth_client.requests.post",
            side_effect=fake_post,
        ) as post_mock, patch(
            "tradingagents.llm_clients.codex_oauth_client.time.sleep",
            return_value=None,
        ) as sleep_mock:
            llm = CodexOAuthChatOpenAI(
                model="gpt-5.4-mini",
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="test-token",
                streaming=True,
                store=False,
                use_responses_api=True,
                temperature=0,
            )

            message = llm.invoke("Reply with exactly: pong")

        self.assertEqual(message.content, "pong")
        self.assertEqual(post_mock.call_count, 2)
        self.assertTrue(sleep_mock.called)

    def test_structured_output_wrapper_returns_pydantic_object(self):
        tool_done = {
            "type": "response.output_item.done",
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "status": "completed",
                "arguments": json.dumps(
                    {
                        "recommendation": "Hold",
                        "rationale": "The debate is balanced.",
                        "strategic_actions": "Maintain exposure and wait for confirmation.",
                    }
                ),
                "call_id": "call_123",
                "name": "ResearchPlan",
            },
            "output_index": 0,
            "sequence_number": 13,
        }
        completed = {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-5.4-mini-2026-03-17",
                "service_tier": "default",
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        }

        class _FakePostResponse:
            def raise_for_status(self):
                return None

            def iter_lines(self, decode_unicode=True):
                return iter(
                    [
                        "event: response.output_item.done",
                        f"data: {json.dumps(tool_done)}",
                        "",
                        "event: response.completed",
                        f"data: {json.dumps(completed)}",
                        "",
                    ]
                )

        with patch(
            "tradingagents.llm_clients.codex_oauth_client.get_valid_access_token",
            return_value="test-token",
        ), patch(
            "tradingagents.llm_clients.codex_oauth_client.requests.post",
            return_value=_FakePostResponse(),
        ) as post_mock:
            llm = CodexOAuthChatOpenAI(
                model="gpt-5.4-mini",
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="test-token",
                streaming=True,
                store=False,
                use_responses_api=True,
                temperature=0,
            )

            structured = llm.with_structured_output(ResearchPlan)
            result = structured.invoke("Build a plan")

        self.assertIsInstance(result, ResearchPlan)
        self.assertEqual(result.recommendation.value, "Hold")
        self.assertNotIn("ls_structured_output_format", post_mock.call_args.kwargs["json"])
