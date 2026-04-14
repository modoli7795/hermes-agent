import pytest


class TestAdvisorPolicy:
    def test_explicit_verification_request_triggers_single_mode(self):
        from agent.advisor_policy import should_request_advice

        cfg = {
            "enabled": True,
            "invocation": "hybrid",
            "call_mode": "single",
            "max_uses_per_turn": 1,
            "trigger_keywords": [],
            "autonomous_modes": {"default": "single"},
        }

        should, reason, mode = should_request_advice(
            user_text="Please verify this answer with an advisor.",
            cfg=cfg,
            turn_state={"uses_this_turn": 0},
        )

        assert should is True
        assert reason == "requested-by-user"
        assert mode == "single"

    def test_explicit_parallel_request_selects_parallel_mode(self):
        from agent.advisor_policy import should_request_advice

        cfg = {
            "enabled": True,
            "invocation": "hybrid",
            "call_mode": "single",
            "max_uses_per_turn": 1,
            "trigger_keywords": [],
            "autonomous_modes": {"default": "single"},
        }

        should, reason, mode = should_request_advice(
            user_text="Ask multiple advisors independently and compare the results.",
            cfg=cfg,
            turn_state={"uses_this_turn": 0},
        )

        assert should is True
        assert reason == "requested-by-user"
        assert mode == "parallel"

    def test_explicit_debate_request_selects_debate_mode(self):
        from agent.advisor_policy import should_request_advice

        cfg = {
            "enabled": True,
            "invocation": "hybrid",
            "call_mode": "single",
            "max_uses_per_turn": 1,
            "trigger_keywords": [],
            "autonomous_modes": {"default": "single"},
        }

        should, reason, mode = should_request_advice(
            user_text="Have multiple advisors debate this and then synthesize a final answer.",
            cfg=cfg,
            turn_state={"uses_this_turn": 0},
        )

        assert should is True
        assert reason == "requested-by-user"
        assert mode == "debate"

    def test_korean_parallel_trigger_selects_parallel_mode(self):
        from agent.advisor_policy import should_request_advice

        cfg = {
            "enabled": True,
            "invocation": "hybrid",
            "call_mode": "single",
            "max_uses_per_turn": 1,
            "trigger_keywords": [],
            "autonomous_modes": {"default": "single"},
        }

        should, reason, mode = should_request_advice(
            user_text="이건 복수의 어드바이저에게 각각 물어보고 비교해줘.",
            cfg=cfg,
            turn_state={"uses_this_turn": 0},
        )

        assert should is True
        assert reason == "requested-by-user"
        assert mode == "parallel"

    def test_korean_debate_trigger_selects_debate_mode(self):
        from agent.advisor_policy import should_request_advice

        cfg = {
            "enabled": True,
            "invocation": "hybrid",
            "call_mode": "single",
            "max_uses_per_turn": 1,
            "trigger_keywords": [],
            "autonomous_modes": {"default": "single"},
        }

        should, reason, mode = should_request_advice(
            user_text="어드바이저들끼리 토론시켜서 최종 결론을 내줘.",
            cfg=cfg,
            turn_state={"uses_this_turn": 0},
        )

        assert should is True
        assert reason == "requested-by-user"
        assert mode == "debate"

    def test_neutral_task_does_not_trigger_advice(self):
        from agent.advisor_policy import should_request_advice

        cfg = {
            "enabled": True,
            "invocation": "hybrid",
            "call_mode": "single",
            "max_uses_per_turn": 1,
            "trigger_keywords": [],
            "autonomous_modes": {"default": "single"},
        }

        should, reason, mode = should_request_advice(
            user_text="Summarize this file.",
            cfg=cfg,
            turn_state={"uses_this_turn": 0},
        )

        assert should is False
        assert reason == "no-trigger"
        assert mode == "single"

    def test_per_turn_limit_blocks_repeated_advice(self):
        from agent.advisor_policy import should_request_advice

        cfg = {
            "enabled": True,
            "invocation": "hybrid",
            "call_mode": "single",
            "max_uses_per_turn": 1,
            "trigger_keywords": ["architecture"],
            "autonomous_modes": {"default": "single", "architecture": "parallel"},
        }

        should, reason, mode = should_request_advice(
            user_text="Please review the architecture tradeoff.",
            cfg=cfg,
            turn_state={"uses_this_turn": 1},
        )

        assert should is False
        assert reason == "limit-reached"
        assert mode == "single"

    def test_custom_trigger_keyword_works(self):
        from agent.advisor_policy import should_request_advice

        cfg = {
            "enabled": True,
            "invocation": "hybrid",
            "call_mode": "single",
            "max_uses_per_turn": 1,
            "trigger_keywords": ["tradeoff"],
            "autonomous_modes": {"default": "single", "architecture": "parallel", "high_stakes": "debate"},
        }

        should, reason, mode = should_request_advice(
            user_text="We need to analyze a tradeoff here.",
            cfg=cfg,
            turn_state={"uses_this_turn": 0},
        )

        assert should is True
        assert reason == "keyword:tradeoff"
        assert mode == "single"

    def test_architecture_keyword_can_autonomously_upgrade_to_parallel(self):
        from agent.advisor_policy import should_request_advice

        cfg = {
            "enabled": True,
            "invocation": "autonomous",
            "call_mode": "single",
            "max_uses_per_turn": 1,
            "trigger_keywords": ["architecture"],
            "autonomous_modes": {"default": "single", "architecture": "parallel", "high_stakes": "debate"},
        }

        should, reason, mode = should_request_advice(
            user_text="We need an architecture review for this service boundary.",
            cfg=cfg,
            turn_state={"uses_this_turn": 0},
        )

        assert should is True
        assert reason == "keyword:architecture"
        assert mode == "parallel"

    def test_detect_explicit_request_parses_korean_single(self):
        from agent.advisor_policy import detect_explicit_advisor_request

        req = detect_explicit_advisor_request(
            "이건 어드바이저 한 명에게만 물어봐줘.",
            {"call_mode": "single"},
        )

        assert req["requested"] is True
        assert req["mode"] == "single"

    def test_build_advisor_goal_mentions_mode(self):
        from agent.advisor_policy import build_advisor_goal

        goal = build_advisor_goal(
            user_text="검토해줘",
            context_summary="현재 설계안 비교",
            reason="requested-by-user",
            mode="debate",
        )

        assert "Mode: debate" in goal
        assert "Reason: requested-by-user" in goal
        assert "현재 설계안 비교" in goal
