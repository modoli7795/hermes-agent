"""Advisor configuration loading and runtime resolution."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List


def _default_advisor_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import DEFAULT_CONFIG

        advisor = deepcopy(DEFAULT_CONFIG.get("advisor") or {})
        if advisor:
            return advisor
    except Exception:
        pass
    return {
        "enabled": False,
        "mode": "external",
        "strategy": "on_demand",
        "invocation": "hybrid",
        "call_mode": "single",
        "provider": "",
        "model": "",
        "base_url": "",
        "api_key": "",
        "reasoning_effort": "",
        "max_iterations": 12,
        "max_advice_chars": 4000,
        "max_uses_per_turn": 1,
        "toolsets": ["terminal", "file"],
        "advisor_count": 2,
        "debate_rounds": 1,
        "providers": [],
        "autonomous_modes": {
            "default": "single",
            "architecture": "parallel",
            "high_stakes": "debate",
        },
        "trigger_keywords": [],
    }


def load_advisor_config() -> Dict[str, Any]:
    """Load advisor config from runtime CLI config or persistent config."""
    defaults = _default_advisor_config()

    try:
        from cli import CLI_CONFIG  # type: ignore

        cfg = CLI_CONFIG.get("advisor", {})
        if isinstance(cfg, dict) and cfg:
            merged = deepcopy(defaults)
            merged.update(cfg)
            return merged
    except Exception:
        pass

    try:
        from hermes_cli.config import load_config

        full = load_config()
        cfg = full.get("advisor", {}) if isinstance(full, dict) else {}
        if isinstance(cfg, dict):
            merged = deepcopy(defaults)
            merged.update(cfg)
            return merged
    except Exception:
        pass

    return defaults


def advisor_enabled(cfg: Dict[str, Any] | None = None) -> bool:
    cfg = cfg or load_advisor_config()
    return bool(cfg.get("enabled")) and str(cfg.get("mode") or "external").strip().lower() != "off"


def _normalize_runtime_from_entry(entry: Dict[str, Any], shared: Dict[str, Any], *, index: int = 0) -> Dict[str, Any]:
    provider = str(entry.get("provider") or shared.get("provider") or "").strip()
    model = str(entry.get("model") or shared.get("model") or "").strip()
    base_url = str(entry.get("base_url") or shared.get("base_url") or "").strip()
    api_key = str(entry.get("api_key") or shared.get("api_key") or "").strip()
    label = str(entry.get("label") or provider or f"advisor-{index + 1}").strip()
    oauth_preferred = bool(entry.get("oauth_preferred", shared.get("oauth_preferred", True)))

    runtime = {
        "enabled": True,
        "mode": str(shared.get("mode") or "external").strip() or "external",
        "strategy": str(shared.get("strategy") or "on_demand").strip() or "on_demand",
        "invocation": str(shared.get("invocation") or "hybrid").strip() or "hybrid",
        "call_mode": str(shared.get("call_mode") or "single").strip() or "single",
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "reasoning_effort": str(shared.get("reasoning_effort") or "").strip(),
        "max_iterations": int(shared.get("max_iterations") or 12),
        "max_advice_chars": int(shared.get("max_advice_chars") or 4000),
        "max_uses_per_turn": int(shared.get("max_uses_per_turn") or 1),
        "toolsets": list(shared.get("toolsets") or ["terminal", "file"]),
        "advisor_count": int(shared.get("advisor_count") or 2),
        "debate_rounds": int(shared.get("debate_rounds") or 1),
        "autonomous_modes": deepcopy(shared.get("autonomous_modes") or {}),
        "trigger_keywords": list(shared.get("trigger_keywords") or []),
        "label": label,
        "weight": int(entry.get("weight") or 1),
        "oauth_preferred": oauth_preferred,
    }

    if base_url:
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "Advisor base_url is configured but no API key was found. Set advisor.api_key or OPENAI_API_KEY."
            )
        provider_name = "custom"
        api_mode = "chat_completions"
        lowered = base_url.lower()
        if "chatgpt.com/backend-api/codex" in lowered:
            provider_name = "openai-codex"
            api_mode = "codex_responses"
        elif "api.anthropic.com" in lowered:
            provider_name = "anthropic"
            api_mode = "anthropic_messages"
        runtime.update(
            {
                "provider": provider_name,
                "base_url": base_url,
                "api_key": api_key,
                "api_mode": api_mode,
            }
        )
        return runtime

    if not provider:
        raise ValueError("Advisor is enabled but no provider or base_url is configured.")

    from hermes_cli.runtime_provider import resolve_runtime_provider

    try:
        resolved = resolve_runtime_provider(requested=provider)
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve advisor provider '{provider}': {exc}. Check that the provider is configured or set advisor.base_url/advisor.api_key."
        ) from exc

    resolved_api_key = str(resolved.get("api_key") or "").strip()
    if not resolved_api_key:
        raise ValueError(
            f"Advisor provider '{provider}' resolved but no API key was found. Set the appropriate environment variable or run 'hermes auth'."
        )

    runtime.update(
        {
            "provider": str(resolved.get("provider") or provider),
            "base_url": str(resolved.get("base_url") or "").rstrip("/"),
            "api_key": resolved_api_key,
            "api_mode": str(resolved.get("api_mode") or "chat_completions"),
        }
    )
    if resolved.get("command"):
        runtime["command"] = resolved.get("command")
    if resolved.get("args") is not None:
        runtime["args"] = list(resolved.get("args") or [])
    return runtime



def resolve_advisor_runtime(parent_agent=None) -> Dict[str, Any]:
    cfg = load_advisor_config()
    if not advisor_enabled(cfg):
        return {
            **cfg,
            "enabled": False,
        }
    return _normalize_runtime_from_entry({}, cfg)



def resolve_advisor_runtimes(parent_agent=None) -> List[Dict[str, Any]]:
    cfg = load_advisor_config()
    if not advisor_enabled(cfg):
        return []

    providers = cfg.get("providers") or []
    if isinstance(providers, list) and providers:
        runtimes = []
        for idx, entry in enumerate(providers):
            if not isinstance(entry, dict):
                continue
            runtimes.append(_normalize_runtime_from_entry(entry, cfg, index=idx))
        return runtimes

    return [resolve_advisor_runtime(parent_agent=parent_agent)]


def native_advisor_applicable(cfg: Dict[str, Any], provider: str = "") -> bool:
    """
    Returns True if native advisor_20260301 tool should be used for this request.
    mode "native": requires provider == "anthropic"; raises ValueError otherwise.
    mode "auto":   True only when provider == "anthropic", False otherwise.
    other modes:   always False.
    """
    if not cfg.get("enabled"):
        return False
    mode = str(cfg.get("mode") or "external").strip().lower()
    is_anthropic = str(provider or "").strip().lower() == "anthropic"
    if mode == "native":
        if not is_anthropic:
            raise ValueError(
                f"advisor.mode='native' requires provider='anthropic', got '{provider}'. "
                "Use mode='auto' to fall back to external on non-Anthropic providers."
            )
        return True
    if mode == "auto":
        return is_anthropic
    return False
