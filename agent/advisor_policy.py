"""Advisor trigger detection and mode selection policy."""

from __future__ import annotations

from typing import Any, Dict, Tuple


_EXPLICIT_SINGLE_PATTERNS = [
    "one advisor",
    "single advisor",
    "one expert",
    "한 명",
    "한명",
    "한 사람",
    "단일 어드바이저",
    "어드바이저 한 명",
    "어드바이저 1명",
]

_EXPLICIT_PARALLEL_PATTERNS = [
    "multiple advisors",
    "several advisors",
    "independently",
    "compare the results",
    "parallel advisors",
    "복수의 어드바이저",
    "여러 어드바이저",
    "각각 물어보고",
    "각자 물어보고",
    "비교해줘",
    "독립적으로",
    "병렬로",
]

_EXPLICIT_DEBATE_PATTERNS = [
    "debate",
    "discuss with each other",
    "argue both sides",
    "consensus after debate",
    "토론시켜",
    "토론 시켜",
    "토론하게",
    "논쟁시켜",
    "상호 토론",
    "서로 토론",
    "최종 결론",
]

_EXPLICIT_ADVISOR_PATTERNS = [
    "advisor",
    "adviser",
    "second opinion",
    "verify",
    "double-check",
    "review with",
    "어드바이저",
    "조언",
    "검토해줘",
    "검증해줘",
    "확인해줘",
    "다시 봐줘",
]

_AUTONOMOUS_DEBATE_HINTS = [
    "high stakes",
    "critical",
    "risky",
    "safety",
    "security",
    "고위험",
    "중요",
    "치명적",
    "보안",
    "위험",
]



def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())



def advisor_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("enabled")) and str(cfg.get("mode") or "external").strip().lower() != "off"



def detect_explicit_advisor_request(user_text: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    text = _normalize(user_text)
    default_mode = str(cfg.get("call_mode") or "single").strip() or "single"

    if not text:
        return {"requested": False, "mode": default_mode}

    has_advisor_intent = any(p in text for p in _EXPLICIT_ADVISOR_PATTERNS)
    mode = default_mode

    if any(p in text for p in _EXPLICIT_DEBATE_PATTERNS):
        return {"requested": True, "mode": "debate"}
    if any(p in text for p in _EXPLICIT_PARALLEL_PATTERNS):
        return {"requested": True, "mode": "parallel"}
    if any(p in text for p in _EXPLICIT_SINGLE_PATTERNS):
        return {"requested": True, "mode": "single"}
    if has_advisor_intent:
        return {"requested": True, "mode": default_mode}
    return {"requested": False, "mode": default_mode}



def select_advisor_mode(*, user_text: str, cfg: Dict[str, Any], turn_state: Dict[str, Any], explicit_request: Dict[str, Any] | None) -> Tuple[str, str]:
    text = _normalize(user_text)
    autonomous_modes = cfg.get("autonomous_modes") or {}
    default_mode = str(autonomous_modes.get("default") or cfg.get("call_mode") or "single").strip() or "single"

    if explicit_request and explicit_request.get("requested"):
        return str(explicit_request.get("mode") or default_mode), "requested-by-user"

    if any(hint in text for hint in _AUTONOMOUS_DEBATE_HINTS):
        return str(autonomous_modes.get("high_stakes") or "debate"), "autonomous:high-stakes"

    if "architecture" in text or "아키텍처" in text:
        return str(autonomous_modes.get("architecture") or default_mode), "keyword:architecture"

    trigger_keywords = [str(k).strip().lower() for k in (cfg.get("trigger_keywords") or []) if str(k).strip()]
    for keyword in trigger_keywords:
        if keyword and keyword in text:
            if keyword in {"architecture", "아키텍처"}:
                return str(autonomous_modes.get("architecture") or default_mode), f"keyword:{keyword}"
            return default_mode, f"keyword:{keyword}"

    return default_mode, "no-trigger"



def should_request_advice(*, user_text: str, cfg: Dict[str, Any], turn_state: Dict[str, Any]) -> Tuple[bool, str, str]:
    if not advisor_enabled(cfg):
        return False, "advisor-disabled", "single"

    invocation = str(cfg.get("invocation") or "hybrid").strip().lower() or "hybrid"
    explicit_request = detect_explicit_advisor_request(user_text, cfg)
    selected_mode, reason = select_advisor_mode(
        user_text=user_text,
        cfg=cfg,
        turn_state=turn_state,
        explicit_request=explicit_request,
    )

    uses_this_turn = int(turn_state.get("uses_this_turn") or 0)
    max_uses = int(cfg.get("max_uses_per_turn") or 1)
    if uses_this_turn >= max_uses:
        return False, "limit-reached", "single"

    if explicit_request.get("requested"):
        if invocation in {"explicit", "hybrid"}:
            return True, "requested-by-user", selected_mode
        return False, "invocation-disabled", selected_mode

    if reason != "no-trigger" and invocation in {"autonomous", "hybrid"}:
        return True, reason, selected_mode

    return False, "no-trigger", selected_mode



def build_advisor_goal(*, user_text: str, context_summary: str, reason: str, mode: str) -> str:
    return (
        f"Mode: {mode}\n"
        f"Reason: {reason}\n"
        f"User request: {user_text}\n"
        f"Context summary: {context_summary}\n\n"
        "Provide concise advisor guidance. Return:\n"
        "1. Best recommendation\n"
        "2. Key risks or uncertainty\n"
        "3. Concrete next step\n"
    )
