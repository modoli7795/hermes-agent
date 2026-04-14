from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"tool {n}",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="Hello", finish_reason="stop"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


class TestExternalAdvisorIntegration:
    def test_run_conversation_injects_parallel_advisor_note(self, agent, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {"enabled": True, "mode": "external", "invocation": "hybrid", "call_mode": "single", "max_uses_per_turn": 1},
        )
        monkeypatch.setattr(
            "agent.advisor_policy.should_request_advice",
            lambda **kwargs: (True, "requested-by-user", "parallel"),
        )
        monkeypatch.setattr(
            "agent.advisor_config.resolve_advisor_runtimes",
            lambda parent_agent=None: [
                {"label": "claude", "provider": "anthropic", "model": "claude-opus-4-6"},
                {"label": "codex", "provider": "openai-codex", "model": "gpt-5.2-codex"},
            ],
        )
        monkeypatch.setattr(
            "agent.advisor_runner.request_parallel_advice",
            lambda **kwargs: {
                "status": "completed",
                "call_mode": "parallel",
                "results": [
                    {"label": "claude", "summary": "claude says A"},
                    {"label": "codex", "summary": "codex says B"},
                ],
                "summary": "claude: A\ncodex: B",
            },
        )

        captured = {}

        def fake_interruptible_api_call(api_kwargs):
            captured["messages"] = api_kwargs["messages"]
            return _mock_response(content="Final answer")

        monkeypatch.setattr(agent, "_interruptible_api_call", fake_interruptible_api_call)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda api_kwargs, on_first_delta=None: fake_interruptible_api_call(api_kwargs))

        result = agent.run_conversation("Please verify this with multiple advisors.")

        assert result["final_response"] == "Final answer"
        assistant_messages = [m for m in captured["messages"] if m.get("role") == "assistant"]
        assert any("Advisor summary mode=parallel" in (m.get("content") or "") for m in assistant_messages)
        assert any("claude says A" in (m.get("content") or "") for m in assistant_messages)

    def test_run_conversation_injects_debate_advisor_note(self, agent, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {"enabled": True, "mode": "external", "invocation": "hybrid", "call_mode": "single", "max_uses_per_turn": 1},
        )
        monkeypatch.setattr(
            "agent.advisor_policy.should_request_advice",
            lambda **kwargs: (True, "requested-by-user", "debate"),
        )
        monkeypatch.setattr(
            "agent.advisor_config.resolve_advisor_runtimes",
            lambda parent_agent=None: [
                {"label": "claude", "provider": "anthropic", "model": "claude-opus-4-6"},
                {"label": "codex", "provider": "openai-codex", "model": "gpt-5.2-codex"},
            ],
        )
        monkeypatch.setattr(
            "agent.advisor_runner.request_debate_advice",
            lambda **kwargs: {
                "status": "completed",
                "call_mode": "debate",
                "initial_results": [
                    {"label": "claude", "summary": "position A"},
                    {"label": "codex", "summary": "position B"},
                ],
                "summary": "final debated recommendation",
            },
        )

        captured = {}

        def fake_interruptible_api_call(api_kwargs):
            captured["messages"] = api_kwargs["messages"]
            return _mock_response(content="Final answer")

        monkeypatch.setattr(agent, "_interruptible_api_call", fake_interruptible_api_call)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda api_kwargs, on_first_delta=None: fake_interruptible_api_call(api_kwargs))

        agent.run_conversation("어드바이저들끼리 토론시켜서 답해줘.")

        assistant_messages = [m for m in captured["messages"] if m.get("role") == "assistant"]
        assert any("Advisor summary mode=debate" in (m.get("content") or "") for m in assistant_messages)
        assert any("final debated recommendation" in (m.get("content") or "") for m in assistant_messages)

    def test_advisor_failure_does_not_crash_turn(self, agent, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {"enabled": True, "mode": "external", "invocation": "hybrid", "call_mode": "single", "max_uses_per_turn": 1},
        )
        monkeypatch.setattr(
            "agent.advisor_policy.should_request_advice",
            lambda **kwargs: (True, "requested-by-user", "single"),
        )
        monkeypatch.setattr(
            "agent.advisor_config.resolve_advisor_runtime",
            lambda parent_agent=None: {"label": "claude", "provider": "anthropic", "model": "claude-opus-4-6"},
        )
        monkeypatch.setattr(
            "agent.advisor_runner.request_external_advice",
            lambda **kwargs: {"status": "error", "error": "boom", "call_mode": "single"},
        )

        captured = {}

        def fake_interruptible_api_call(api_kwargs):
            captured["messages"] = api_kwargs["messages"]
            return _mock_response(content="Still answered")

        monkeypatch.setattr(agent, "_interruptible_api_call", fake_interruptible_api_call)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda api_kwargs, on_first_delta=None: fake_interruptible_api_call(api_kwargs))

        result = agent.run_conversation("verify this")

        assert result["final_response"] == "Still answered"
        assistant_messages = [m for m in captured["messages"] if m.get("role") == "assistant"]
        assert not any("Advisor summary mode=" in (m.get("content") or "") for m in assistant_messages)

    def test_per_turn_limit_is_respected_in_integration(self, agent, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {"enabled": True, "mode": "external", "invocation": "hybrid", "call_mode": "single", "max_uses_per_turn": 1},
        )
        monkeypatch.setattr(
            "agent.advisor_policy.should_request_advice",
            lambda **kwargs: (False, "limit-reached", "single"),
        )

        called = {"single": 0}
        monkeypatch.setattr(
            "agent.advisor_runner.request_external_advice",
            lambda **kwargs: called.__setitem__("single", called["single"] + 1),
        )

        captured = {}

        def fake_interruptible_api_call(api_kwargs):
            captured["messages"] = api_kwargs["messages"]
            return _mock_response(content="No advisor used")

        monkeypatch.setattr(agent, "_interruptible_api_call", fake_interruptible_api_call)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda api_kwargs, on_first_delta=None: fake_interruptible_api_call(api_kwargs))

        result = agent.run_conversation("verify this")

        assert result["final_response"] == "No advisor used"
        assert called["single"] == 0
