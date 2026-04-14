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


class TestAdvisorRoutingIntegration:
    def test_advisor_disabled_keeps_turn_plain(self, agent, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {"enabled": False},
        )

        captured = {}

        def fake_interruptible_api_call(api_kwargs):
            captured["messages"] = api_kwargs["messages"]
            return _mock_response(content="No advisor path")

        monkeypatch.setattr(agent, "_interruptible_api_call", fake_interruptible_api_call)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda api_kwargs, on_first_delta=None: fake_interruptible_api_call(api_kwargs))

        result = agent.run_conversation("hello")

        assert result["final_response"] == "No advisor path"
        assistant_messages = [m for m in captured["messages"] if m.get("role") == "assistant"]
        assert not any("Advisor summary mode=" in (m.get("content") or "") for m in assistant_messages)

    def test_advisor_enabled_with_anthropic_provider(self, agent, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {
                "enabled": True,
                "mode": "external",
                "invocation": "hybrid",
                "call_mode": "single",
                "max_uses_per_turn": 1,
                "provider": "anthropic",
                "model": "claude-opus-4-6",
            },
        )
        monkeypatch.setattr(
            "agent.advisor_policy.should_request_advice",
            lambda **kwargs: (True, "requested-by-user", "single"),
        )
        monkeypatch.setattr(
            "agent.advisor_config.resolve_advisor_runtime",
            lambda parent_agent=None: {
                "provider": "anthropic",
                "model": "claude-opus-4-6",
                "label": "claude",
            },
        )
        monkeypatch.setattr(
            "agent.advisor_runner.request_external_advice",
            lambda **kwargs: {
                "status": "completed",
                "call_mode": "single",
                "advisor_provider": "anthropic",
                "advisor_model": "claude-opus-4-6",
                "summary": "anthropic recommendation",
            },
        )

        captured = {}

        def fake_interruptible_api_call(api_kwargs):
            captured["messages"] = api_kwargs["messages"]
            return _mock_response(content="Done")

        monkeypatch.setattr(agent, "_interruptible_api_call", fake_interruptible_api_call)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda api_kwargs, on_first_delta=None: fake_interruptible_api_call(api_kwargs))

        agent.run_conversation("verify this")

        assistant_messages = [m for m in captured["messages"] if m.get("role") == "assistant"]
        assert any("anthropic recommendation" in (m.get("content") or "") for m in assistant_messages)

    def test_advisor_enabled_with_codex_provider(self, agent, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {
                "enabled": True,
                "mode": "external",
                "invocation": "hybrid",
                "call_mode": "single",
                "max_uses_per_turn": 1,
                "provider": "openai-codex",
                "model": "gpt-5.2-codex",
            },
        )
        monkeypatch.setattr(
            "agent.advisor_policy.should_request_advice",
            lambda **kwargs: (True, "requested-by-user", "single"),
        )
        monkeypatch.setattr(
            "agent.advisor_config.resolve_advisor_runtime",
            lambda parent_agent=None: {
                "provider": "openai-codex",
                "model": "gpt-5.2-codex",
                "label": "codex",
            },
        )
        monkeypatch.setattr(
            "agent.advisor_runner.request_external_advice",
            lambda **kwargs: {
                "status": "completed",
                "call_mode": "single",
                "advisor_provider": "openai-codex",
                "advisor_model": "gpt-5.2-codex",
                "summary": "codex recommendation",
            },
        )

        captured = {}

        def fake_interruptible_api_call(api_kwargs):
            captured["messages"] = api_kwargs["messages"]
            return _mock_response(content="Done")

        monkeypatch.setattr(agent, "_interruptible_api_call", fake_interruptible_api_call)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda api_kwargs, on_first_delta=None: fake_interruptible_api_call(api_kwargs))

        agent.run_conversation("verify this")

        assistant_messages = [m for m in captured["messages"] if m.get("role") == "assistant"]
        assert any("codex recommendation" in (m.get("content") or "") for m in assistant_messages)

    def test_advisor_enabled_with_direct_base_url(self, agent, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {
                "enabled": True,
                "mode": "external",
                "invocation": "hybrid",
                "call_mode": "single",
                "max_uses_per_turn": 1,
                "base_url": "https://example.com/v1",
                "api_key": "example-key",
                "model": "example-model",
            },
        )
        monkeypatch.setattr(
            "agent.advisor_policy.should_request_advice",
            lambda **kwargs: (True, "requested-by-user", "single"),
        )
        monkeypatch.setattr(
            "agent.advisor_config.resolve_advisor_runtime",
            lambda parent_agent=None: {
                "provider": "custom",
                "base_url": "https://example.com/v1",
                "api_key": "example-key",
                "model": "example-model",
                "label": "custom-advisor",
            },
        )
        monkeypatch.setattr(
            "agent.advisor_runner.request_external_advice",
            lambda **kwargs: {
                "status": "completed",
                "call_mode": "single",
                "advisor_provider": "custom",
                "advisor_model": "example-model",
                "summary": "custom recommendation",
            },
        )

        captured = {}

        def fake_interruptible_api_call(api_kwargs):
            captured["messages"] = api_kwargs["messages"]
            return _mock_response(content="Done")

        monkeypatch.setattr(agent, "_interruptible_api_call", fake_interruptible_api_call)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda api_kwargs, on_first_delta=None: fake_interruptible_api_call(api_kwargs))

        agent.run_conversation("verify this")

        assistant_messages = [m for m in captured["messages"] if m.get("role") == "assistant"]
        assert any("custom recommendation" in (m.get("content") or "") for m in assistant_messages)

    def test_explicit_parallel_mode_works(self, agent, monkeypatch):
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
                    {"label": "claude", "summary": "A"},
                    {"label": "codex", "summary": "B"},
                ],
                "summary": "A vs B",
            },
        )

        captured = {}

        def fake_interruptible_api_call(api_kwargs):
            captured["messages"] = api_kwargs["messages"]
            return _mock_response(content="Done")

        monkeypatch.setattr(agent, "_interruptible_api_call", fake_interruptible_api_call)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda api_kwargs, on_first_delta=None: fake_interruptible_api_call(api_kwargs))

        agent.run_conversation("여러 어드바이저에게 물어봐")
        assistant_messages = [m for m in captured["messages"] if m.get("role") == "assistant"]
        assert any("Advisor summary mode=parallel" in (m.get("content") or "") for m in assistant_messages)

    def test_explicit_debate_mode_works(self, agent, monkeypatch):
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
                    {"label": "claude", "summary": "A"},
                    {"label": "codex", "summary": "B"},
                ],
                "summary": "debated final",
            },
        )

        captured = {}

        def fake_interruptible_api_call(api_kwargs):
            captured["messages"] = api_kwargs["messages"]
            return _mock_response(content="Done")

        monkeypatch.setattr(agent, "_interruptible_api_call", fake_interruptible_api_call)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda api_kwargs, on_first_delta=None: fake_interruptible_api_call(api_kwargs))

        agent.run_conversation("토론시켜서 결론 내줘")
        assistant_messages = [m for m in captured["messages"] if m.get("role") == "assistant"]
        assert any("Advisor summary mode=debate" in (m.get("content") or "") for m in assistant_messages)

    def test_autonomous_upgrade_to_parallel(self, agent, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {"enabled": True, "mode": "external", "invocation": "autonomous", "call_mode": "single", "max_uses_per_turn": 1},
        )
        monkeypatch.setattr(
            "agent.advisor_policy.should_request_advice",
            lambda **kwargs: (True, "keyword:architecture", "parallel"),
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
                    {"label": "claude", "summary": "A"},
                    {"label": "codex", "summary": "B"},
                ],
                "summary": "auto parallel summary",
            },
        )

        captured = {}

        def fake_interruptible_api_call(api_kwargs):
            captured["messages"] = api_kwargs["messages"]
            return _mock_response(content="Done")

        monkeypatch.setattr(agent, "_interruptible_api_call", fake_interruptible_api_call)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda api_kwargs, on_first_delta=None: fake_interruptible_api_call(api_kwargs))

        agent.run_conversation("We need an architecture review.")
        assistant_messages = [m for m in captured["messages"] if m.get("role") == "assistant"]
        assert any("Advisor summary mode=parallel" in (m.get("content") or "") for m in assistant_messages)

    def test_missing_credentials_fail_open(self, agent, monkeypatch):
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
            lambda parent_agent=None: (_ for _ in ()).throw(ValueError("missing credentials")),
        )

        captured = {}

        def fake_interruptible_api_call(api_kwargs):
            captured["messages"] = api_kwargs["messages"]
            return _mock_response(content="Still works")

        monkeypatch.setattr(agent, "_interruptible_api_call", fake_interruptible_api_call)
        monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda api_kwargs, on_first_delta=None: fake_interruptible_api_call(api_kwargs))

        result = agent.run_conversation("verify this")

        assert result["final_response"] == "Still works"
        assistant_messages = [m for m in captured["messages"] if m.get("role") == "assistant"]
        assert not any("Advisor summary mode=" in (m.get("content") or "") for m in assistant_messages)
