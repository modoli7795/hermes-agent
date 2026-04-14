from types import SimpleNamespace


class TestAdvisorRunner:
    def test_request_external_advice_passes_runtime_and_trims_summary(self, monkeypatch):
        runtime = {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "base_url": "https://api.anthropic.com",
            "api_key": "anth-key",
            "api_mode": "anthropic_messages",
            "max_iterations": 7,
            "max_advice_chars": 20,
            "toolsets": ["terminal", "file"],
        }
        parent = SimpleNamespace(
            model="gpt-5.4",
            provider="openai-codex",
            base_url=None,
            api_key="parent-key",
            api_mode="codex_responses",
            acp_command=None,
            acp_args=[],
            max_tokens=4000,
            reasoning_config=None,
            prefill_messages=None,
            platform="telegram",
            providers_allowed=None,
            providers_ignored=None,
            providers_order=None,
            provider_sort=None,
            _session_db=None,
            session_id="sess-1",
            enabled_toolsets=["terminal", "file"],
            valid_tool_names=[],
            _delegate_depth=0,
        )

        class FakeChild:
            def run_conversation(self, user_message):
                return {
                    "final_response": "abcdefghijklmnopqrstuvwxyz",
                    "completed": True,
                    "interrupted": False,
                    "api_calls": 3,
                    "messages": [],
                }

        monkeypatch.setattr(
            "agent.advisor_runner._build_advisor_child",
            lambda **kwargs: FakeChild(),
        )

        from agent.advisor_runner import request_external_advice

        result = request_external_advice(
            parent_agent=parent,
            goal="review this",
            context="important context",
            runtime=runtime,
        )

        assert result["status"] == "completed"
        assert result["advisor_provider"] == "anthropic"
        assert result["advisor_model"] == "claude-opus-4-6"
        assert result["summary"] == "abcdefghijklmnopqrst"
        assert result["call_mode"] == "single"

    def test_request_parallel_advice_combines_results(self, monkeypatch):
        parent = SimpleNamespace()
        runtimes = [
            {"label": "claude", "provider": "anthropic", "model": "claude-opus-4-6"},
            {"label": "codex", "provider": "openai-codex", "model": "gpt-5.2-codex"},
        ]

        monkeypatch.setattr(
            "agent.advisor_runner.request_external_advice",
            lambda **kwargs: {
                "status": "completed",
                "call_mode": "single",
                "advisor_provider": kwargs["runtime"]["provider"],
                "advisor_model": kwargs["runtime"]["model"],
                "summary": f"summary:{kwargs['runtime']['label']}",
                "full_text": f"full:{kwargs['runtime']['label']}",
                "label": kwargs["runtime"]["label"],
            },
        )

        from agent.advisor_runner import request_parallel_advice

        result = request_parallel_advice(
            parent_agent=parent,
            goal="compare",
            context="ctx",
            runtimes=runtimes,
        )

        assert result["status"] == "completed"
        assert result["call_mode"] == "parallel"
        assert len(result["results"]) == 2
        assert result["results"][0]["label"] == "claude"
        assert "summary:claude" in result["summary"]
        assert "summary:codex" in result["summary"]

    def test_request_debate_advice_runs_synthesis_pass(self, monkeypatch):
        parent = SimpleNamespace()
        runtimes = [
            {"label": "claude", "provider": "anthropic", "model": "claude-opus-4-6"},
            {"label": "codex", "provider": "openai-codex", "model": "gpt-5.2-codex"},
        ]

        calls = []

        def fake_request_external_advice(**kwargs):
            calls.append(kwargs["runtime"]["label"])
            if kwargs["runtime"]["label"] == "synthesizer":
                return {
                    "status": "completed",
                    "call_mode": "single",
                    "advisor_provider": "anthropic",
                    "advisor_model": "claude-opus-4-6",
                    "summary": "final debated recommendation",
                    "full_text": "final debated recommendation",
                    "label": "synthesizer",
                }
            return {
                "status": "completed",
                "call_mode": "single",
                "advisor_provider": kwargs["runtime"]["provider"],
                "advisor_model": kwargs["runtime"]["model"],
                "summary": f"position:{kwargs['runtime']['label']}",
                "full_text": f"position:{kwargs['runtime']['label']}",
                "label": kwargs["runtime"]["label"],
            }

        monkeypatch.setattr("agent.advisor_runner.request_external_advice", fake_request_external_advice)

        from agent.advisor_runner import request_debate_advice

        result = request_debate_advice(
            parent_agent=parent,
            goal="debate this",
            context="ctx",
            runtimes=runtimes,
        )

        assert result["status"] == "completed"
        assert result["call_mode"] == "debate"
        assert len(result["initial_results"]) == 2
        assert result["summary"] == "final debated recommendation"
        assert calls == ["claude", "codex", "synthesizer"]

    def test_request_external_advice_returns_structured_error(self, monkeypatch):
        runtime = {
            "provider": "anthropic",
            "model": "claude-opus-4-6",
            "max_advice_chars": 100,
            "toolsets": ["terminal", "file"],
        }
        parent = SimpleNamespace()

        class BrokenChild:
            def run_conversation(self, user_message):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            "agent.advisor_runner._build_advisor_child",
            lambda **kwargs: BrokenChild(),
        )

        from agent.advisor_runner import request_external_advice

        result = request_external_advice(
            parent_agent=parent,
            goal="review this",
            context="ctx",
            runtime=runtime,
        )

        assert result["status"] == "error"
        assert "boom" in result["error"]
