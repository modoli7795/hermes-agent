"""External advisor execution helpers."""

from __future__ import annotations

import time
from typing import Any, Dict, List


_DEFAULT_TOOLSETS = ["terminal", "file"]



def _trim_text(text: str, limit: int) -> str:
    text = text or ""
    limit = max(0, int(limit or 0))
    if not limit:
        return text
    return text[:limit]



def _build_advisor_prompt(goal: str, context: str, mode: str) -> str:
    return (
        f"Mode: {mode}\n"
        "You are an advisor model. Do not execute tools unless necessary.\n"
        "Return:\n"
        "1. Best recommendation\n"
        "2. Key risks or uncertainty\n"
        "3. Concrete next step\n"
        "Keep the response under 250 words.\n\n"
        f"Goal: {goal}\n"
        f"Context: {context}"
    )



def _make_stub_parent(runtime: Dict[str, Any]):
    """Create a minimal stub parent agent when no real parent is available (e.g. standalone tests)."""
    from run_agent import AIAgent
    return AIAgent(
        model=runtime.get("model") or "claude-haiku-4-5",
        provider=runtime.get("provider"),
        base_url=runtime.get("base_url"),
        api_key=runtime.get("api_key"),
        api_mode=runtime.get("api_mode"),
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _build_advisor_child(*, parent_agent, goal: str, context: str, runtime: Dict[str, Any], task_index: int = 0):
    from tools.delegate_tool import _build_child_agent

    effective_parent = parent_agent if parent_agent is not None else _make_stub_parent(runtime)

    return _build_child_agent(
        task_index=task_index,
        goal=_build_advisor_prompt(goal, context, runtime.get("call_mode") or "single"),
        context=context,
        toolsets=runtime.get("toolsets") or _DEFAULT_TOOLSETS,
        model=runtime.get("model"),
        max_iterations=int(runtime.get("max_iterations") or 12),
        parent_agent=effective_parent,
        override_provider=runtime.get("provider"),
        override_base_url=runtime.get("base_url"),
        override_api_key=runtime.get("api_key"),
        override_api_mode=runtime.get("api_mode"),
        override_acp_command=runtime.get("command"),
        override_acp_args=runtime.get("args"),
    )



def request_external_advice(*, parent_agent, goal: str, context: str, runtime: Dict[str, Any]) -> Dict[str, Any]:
    start = time.monotonic()
    try:
        child = _build_advisor_child(parent_agent=parent_agent, goal=goal, context=context, runtime=runtime)
        result = child.run_conversation(user_message=goal)
        full_text = result.get("final_response") or ""
        summary = _trim_text(full_text, int(runtime.get("max_advice_chars") or 4000))
        duration = round(time.monotonic() - start, 2)
        return {
            "status": "completed" if full_text else "failed",
            "call_mode": "single",
            "advisor_provider": runtime.get("provider"),
            "advisor_model": runtime.get("model"),
            "label": runtime.get("label") or runtime.get("provider") or "advisor",
            "summary": summary,
            "full_text": full_text,
            "duration_seconds": duration,
            "api_calls": result.get("api_calls", 0),
        }
    except Exception as exc:
        duration = round(time.monotonic() - start, 2)
        return {
            "status": "error",
            "call_mode": "single",
            "advisor_provider": runtime.get("provider"),
            "advisor_model": runtime.get("model"),
            "label": runtime.get("label") or runtime.get("provider") or "advisor",
            "summary": "",
            "full_text": "",
            "error": str(exc),
            "duration_seconds": duration,
        }



def request_parallel_advice(*, parent_agent, goal: str, context: str, runtimes: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = []
    for runtime in runtimes:
        results.append(
            request_external_advice(
                parent_agent=parent_agent,
                goal=goal,
                context=context,
                runtime={**runtime, "call_mode": "single"},
            )
        )
    pieces = []
    for item in results:
        label = item.get("label") or item.get("advisor_provider") or "advisor"
        pieces.append(f"{label}: {item.get('summary', '')}")
    return {
        "status": "completed" if results else "failed",
        "call_mode": "parallel",
        "results": results,
        "summary": "\n".join(pieces),
    }



def request_debate_advice(*, parent_agent, goal: str, context: str, runtimes: List[Dict[str, Any]]) -> Dict[str, Any]:
    initial = []
    for runtime in runtimes:
        initial.append(
            request_external_advice(
                parent_agent=parent_agent,
                goal=goal,
                context=context,
                runtime={**runtime, "call_mode": "single"},
            )
        )

    disagreement_summary = "\n".join(
        f"{item.get('label')}: {item.get('summary', '')}" for item in initial
    )
    synthesizer = dict(runtimes[0]) if runtimes else {"provider": "advisor", "model": "advisor"}
    synthesizer["label"] = "synthesizer"
    synth = request_external_advice(
        parent_agent=parent_agent,
        goal=f"Synthesize a final recommendation after comparing these advisor positions:\n{disagreement_summary}",
        context=context,
        runtime={**synthesizer, "call_mode": "single"},
    )
    return {
        "status": synth.get("status", "failed"),
        "call_mode": "debate",
        "initial_results": initial,
        "debate_summary": disagreement_summary,
        "summary": synth.get("summary", ""),
        "final_result": synth,
    }
