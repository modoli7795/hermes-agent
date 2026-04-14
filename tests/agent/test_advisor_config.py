import pytest


class TestAdvisorConfig:
    def test_advisor_enabled_false_when_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {"enabled": False},
        )

        from agent.advisor_config import advisor_enabled

        assert advisor_enabled() is False

    def test_resolve_advisor_runtime_from_provider(self, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {
                "enabled": True,
                "mode": "external",
                "strategy": "on_demand",
                "invocation": "hybrid",
                "call_mode": "single",
                "provider": "anthropic",
                "model": "claude-opus-4-6",
                "reasoning_effort": "high",
                "max_iterations": 12,
                "max_advice_chars": 4000,
                "max_uses_per_turn": 1,
                "toolsets": ["terminal", "file"],
            },
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested=None, **kwargs: {
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com",
                "api_key": "anth-key",
                "api_mode": "anthropic_messages",
            },
        )

        from agent.advisor_config import resolve_advisor_runtime

        runtime = resolve_advisor_runtime()

        assert runtime["enabled"] is True
        assert runtime["provider"] == "anthropic"
        assert runtime["model"] == "claude-opus-4-6"
        assert runtime["base_url"] == "https://api.anthropic.com"
        assert runtime["api_key"] == "anth-key"
        assert runtime["api_mode"] == "anthropic_messages"
        assert runtime["call_mode"] == "single"
        assert runtime["oauth_preferred"] is True

    def test_resolve_advisor_runtime_from_base_url(self, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {
                "enabled": True,
                "mode": "external",
                "call_mode": "single",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "codex-key",
                "model": "gpt-5.2-codex",
            },
        )

        from agent.advisor_config import resolve_advisor_runtime

        runtime = resolve_advisor_runtime()

        assert runtime["provider"] == "openai-codex"
        assert runtime["api_mode"] == "codex_responses"
        assert runtime["base_url"] == "https://chatgpt.com/backend-api/codex"
        assert runtime["api_key"] == "codex-key"

    def test_resolve_advisor_runtimes_from_providers_list(self, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {
                "enabled": True,
                "mode": "external",
                "invocation": "hybrid",
                "call_mode": "parallel",
                "providers": [
                    {
                        "provider": "anthropic",
                        "model": "claude-opus-4-6",
                        "label": "claude",
                        "oauth_preferred": True,
                    },
                    {
                        "provider": "openai-codex",
                        "model": "gpt-5.2-codex",
                        "label": "codex",
                        "oauth_preferred": True,
                    },
                ],
            },
        )

        def fake_resolve_runtime_provider(requested=None, **kwargs):
            if requested == "anthropic":
                return {
                    "provider": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "api_key": "anth-key",
                    "api_mode": "anthropic_messages",
                }
            if requested == "openai-codex":
                return {
                    "provider": "openai-codex",
                    "base_url": "https://chatgpt.com/backend-api/codex",
                    "api_key": "codex-key",
                    "api_mode": "codex_responses",
                }
            raise AssertionError(f"unexpected provider: {requested}")

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            fake_resolve_runtime_provider,
        )

        from agent.advisor_config import resolve_advisor_runtimes

        runtimes = resolve_advisor_runtimes()

        assert len(runtimes) == 2
        assert runtimes[0]["label"] == "claude"
        assert runtimes[0]["provider"] == "anthropic"
        assert runtimes[1]["label"] == "codex"
        assert runtimes[1]["provider"] == "openai-codex"

    def test_resolve_advisor_runtime_raises_clear_error_when_missing_api_key(self, monkeypatch):
        monkeypatch.setattr(
            "agent.advisor_config.load_advisor_config",
            lambda: {
                "enabled": True,
                "mode": "external",
                "call_mode": "single",
                "base_url": "https://example.com/v1",
                "api_key": "",
            },
        )

        from agent.advisor_config import resolve_advisor_runtime

        with pytest.raises(ValueError, match="no API key was found"):
            resolve_advisor_runtime()
