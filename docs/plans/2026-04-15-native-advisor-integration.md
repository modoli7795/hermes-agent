# Native Advisor Integration (Phase 3) Implementation Plan — Rev 2

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Hermes의 `anthropic_messages` 모드에서 Anthropic 네이티브 `advisor_20260301` 툴을 사용해 executor 모델이 생성 도중 서버 측에서 Opus에게 직접 조언을 구할 수 있게 한다.

**Opus 코드리뷰로 발견된 Rev 1 문제점 (전부 수정됨):**
1. `convert_tools_to_anthropic()`가 `advisor_20260301` 타입을 일반 function 포맷으로 잘못 변환 — 별도 bypass 필요
2. `normalize_anthropic_response()`가 `server_tool_use` / `advisor_tool_result` 블록을 조용히 드롭 — 응답 파싱 추가 필요
3. beta 헤더를 `build_anthropic_kwargs()`의 `extra_headers`로 넘겨야 함 (`betas` 파라미터 아님)
4. `build_anthropic_kwargs()`가 `tools` 파라미터로 `self.tools`를 받는데 여기서 advisor 툴 주입 — 별도 kwarg patch 필요
5. Task 2 테스트: `ValueError` raise인데 `False` 반환을 assert — 수정됨
6. `_advisor_cfg_cache` 속성 없음 — `load_advisor_config()` 직접 호출로 교체
7. context compressor가 `advisor_tool_result` 블록을 인식 못 하면 멀티턴 제약 위반 — passthrough 추가 필요

**Architecture:**
- OpenAI SDK / `chat_completions` 경로 완전 불변
- `api_mode == "anthropic_messages"` + `advisor.mode in ("native", "auto")` 일 때만 동작하는 분기
- advisor 툴은 `self.tools`가 아닌 `build_anthropic_kwargs()` 호출 직전 api_kwargs에 패치
- beta 헤더는 `extra_headers` 경유 (기존 fast_mode beta 패턴과 동일)
- `advisor_tool_result` 블록은 Anthropic 네이티브 포맷으로 히스토리 보존
- fallback 발동 시 advisor 블록 제거 (non-Anthropic provider 안전)

**Touch points:**
1. `hermes_cli/config.py` — mode에 `native` / `auto` + `max_uses` 필드 추가
2. `agent/advisor_config.py` — `native_advisor_applicable()` 추가
3. `agent/anthropic_adapter.py` — (A) `convert_tools_to_anthropic()` advisor 타입 bypass, (B) `normalize_anthropic_response()` advisor 블록 파싱, (C) `build_anthropic_kwargs()` beta 헤더 파라미터 추가
4. `run_agent.py` — `_build_api_kwargs()`에서 advisor 툴 + beta 헤더 주입, 멀티턴 강제 검사, fallback 시 advisor 블록 제거

**변경하지 않는 파일:** `advisor_runner.py`, `advisor_policy.py`, `context_compressor.py`

---

## Task 1: Config에 native / auto 모드 + max_uses 추가

**Objective:** `advisor.mode` 값으로 `"native"` / `"auto"` 인식 + `max_uses` 필드 추가. config version bump.

**Files:**
- Modify: `hermes_cli/config.py`
- Test: `tests/hermes_cli/test_config_env_expansion.py`

**Step 1: 실패 테스트 작성**

```python
def test_advisor_config_has_native_mode_and_max_uses():
    from hermes_cli.config import DEFAULT_CONFIG
    advisor = DEFAULT_CONFIG["advisor"]
    assert advisor["mode"] == "external"          # 기본값은 그대로
    assert "max_uses" in advisor                  # 신규 필드
    assert isinstance(advisor["max_uses"], int)
```

**Step 2: 테스트 실행 → FAIL 확인**

```bash
cd ~/.hermes/hermes-agent && source venv/bin/activate
python -m pytest tests/hermes_cli/test_config_env_expansion.py::test_advisor_config_has_native_mode_and_max_uses -v
```
Expected: FAIL — `KeyError: 'max_uses'`

**Step 3: config.py 수정**

`hermes_cli/config.py`의 `"advisor"` 블록에서:

```python
# 변경 전
"mode": "external",              # external | off

# 변경 후
"mode": "external",              # external | native | auto | off
                                  #   native: advisor_20260301 tool, Anthropic only
                                  #   auto:   native when provider==anthropic, else external
"max_uses": 1,                   # max advisor calls per request (native mode)
```

`_config_version` 18 → 19

**Step 4: 테스트 PASS 확인**

```bash
python -m pytest tests/hermes_cli/test_config_env_expansion.py -v
```

**Step 5: Commit**

```bash
git add hermes_cli/config.py tests/hermes_cli/test_config_env_expansion.py
git commit -m "feat: extend advisor config with native/auto mode and max_uses field"
```

---

## Task 2: advisor_config.py에 native_advisor_applicable() 추가

**Objective:** `native_advisor_applicable(cfg, provider)` 함수가 mode에 따라 올바르게 동작한다.

**주의: `mode="native"` + non-Anthropic 일 때 반환값은 `False`가 아니라 `ValueError` raise임.**

**Files:**
- Modify: `agent/advisor_config.py`
- Test: `tests/agent/test_advisor_config.py`

**Step 1: 실패 테스트 작성**

```python
import pytest

def test_native_advisor_applicable_true_when_native_and_anthropic():
    from agent.advisor_config import native_advisor_applicable
    cfg = {"enabled": True, "mode": "native", "model": "claude-opus-4-6"}
    assert native_advisor_applicable(cfg, provider="anthropic") is True

def test_native_advisor_applicable_raises_when_native_non_anthropic():
    from agent.advisor_config import native_advisor_applicable
    cfg = {"enabled": True, "mode": "native", "model": "claude-opus-4-6"}
    with pytest.raises(ValueError, match="provider"):
        native_advisor_applicable(cfg, provider="openai")

def test_native_advisor_applicable_auto_with_anthropic():
    from agent.advisor_config import native_advisor_applicable
    cfg = {"enabled": True, "mode": "auto"}
    assert native_advisor_applicable(cfg, provider="anthropic") is True

def test_native_advisor_applicable_auto_with_non_anthropic():
    from agent.advisor_config import native_advisor_applicable
    cfg = {"enabled": True, "mode": "auto"}
    assert native_advisor_applicable(cfg, provider="openai") is False

def test_native_advisor_applicable_false_when_disabled():
    from agent.advisor_config import native_advisor_applicable
    cfg = {"enabled": False, "mode": "native"}
    assert native_advisor_applicable(cfg, provider="anthropic") is False

def test_native_advisor_applicable_false_for_external_mode():
    from agent.advisor_config import native_advisor_applicable
    cfg = {"enabled": True, "mode": "external"}
    assert native_advisor_applicable(cfg, provider="anthropic") is False
```

**Step 2: 테스트 FAIL 확인**

```bash
python -m pytest tests/agent/test_advisor_config.py -k "native" -v
```

**Step 3: advisor_config.py에 함수 추가**

```python
def native_advisor_applicable(cfg: Dict[str, Any], provider: str = "") -> bool:
    """
    Returns True if native advisor_20260301 tool should be used for this request.

    mode "native": requires provider == "anthropic"; raises ValueError otherwise.
    mode "auto":   True only when provider == "anthropic", False otherwise (no error).
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
```

**Step 4: 테스트 PASS 확인**

```bash
python -m pytest tests/agent/test_advisor_config.py -v
```

**Step 5: Commit**

```bash
git add agent/advisor_config.py tests/agent/test_advisor_config.py
git commit -m "feat: add native_advisor_applicable() to advisor_config"
```

---

## Task 3: convert_tools_to_anthropic() — advisor 타입 bypass

**Objective:** `advisor_20260301` 타입 툴은 OpenAI function 포맷 변환 없이 그대로 통과시킨다.

**현재 문제:** `convert_tools_to_anthropic()`은 모든 툴을 `fn = t.get("function", {})` 패턴으로 변환하므로 `advisor_20260301` 타입이 `{"name": "", "description": "", "input_schema": {}}` 로 망가진다.

**Files:**
- Modify: `agent/anthropic_adapter.py` — `convert_tools_to_anthropic()`
- Test: `tests/agent/test_anthropic_adapter.py` (기존 파일에 추가)

**Step 1: 실패 테스트 작성**

```python
def test_convert_tools_preserves_advisor_tool_type():
    from agent.anthropic_adapter import convert_tools_to_anthropic
    tools = [
        {"function": {"name": "terminal", "description": "run cmd", "parameters": {}}},
        {
            "type": "advisor_20260301",
            "name": "advisor",
            "model": "claude-opus-4-6",
            "max_uses": 1,
        },
    ]
    result = convert_tools_to_anthropic(tools)
    # advisor_20260301 툴은 원형 그대로여야 함
    advisor = next(t for t in result if t.get("type") == "advisor_20260301")
    assert advisor["model"] == "claude-opus-4-6"
    assert advisor["name"] == "advisor"
    assert advisor["max_uses"] == 1
```

**Step 2: 테스트 FAIL 확인**

```bash
python -m pytest tests/agent/test_anthropic_adapter.py::test_convert_tools_preserves_advisor_tool_type -v
```

**Step 3: convert_tools_to_anthropic() 수정**

```python
# _SERVER_TOOL_TYPES: Anthropic native server-side tools — pass through as-is
_SERVER_TOOL_TYPES = {"advisor_20260301", "web_search_20250305", "computer_use_20241022"}

def convert_tools_to_anthropic(tools: List[Dict]) -> List[Dict]:
    """Convert OpenAI tool definitions to Anthropic format.
    Server-side native tool types (e.g. advisor_20260301) are passed through unchanged.
    """
    if not tools:
        return []
    result = []
    for t in tools:
        if t.get("type") in _SERVER_TOOL_TYPES:
            result.append(dict(t))  # native server tool — pass through
            continue
        fn = t.get("function", {})
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result
```

**Step 4: 테스트 PASS 확인**

```bash
python -m pytest tests/agent/test_anthropic_adapter.py -v
```

**Step 5: Commit**

```bash
git add agent/anthropic_adapter.py tests/agent/test_anthropic_adapter.py
git commit -m "fix: pass through advisor_20260301 and other server tool types in convert_tools_to_anthropic"
```

---

## Task 4: normalize_anthropic_response() — advisor 블록 파싱

**Objective:** `server_tool_use` / `advisor_tool_result` / `advisor_tool_result_error` 블록을 파싱해 히스토리에 보존한다. 단, OpenAI-format 응답(tool_calls)에는 영향 없음.

**현재 문제:** `normalize_anthropic_response()`는 `text`, `thinking`, `tool_use` 만 처리하고 advisor 블록 타입은 조용히 드롭된다.

**Files:**
- Modify: `agent/anthropic_adapter.py` — `normalize_anthropic_response()`
- Test: `tests/agent/test_anthropic_adapter.py`

**Step 1: 실패 테스트 작성**

```python
from types import SimpleNamespace

def _make_mock_response_with_advisor():
    """Simulate Anthropic SDK response with advisor blocks."""
    block1 = SimpleNamespace(type="text", text="I'll consult the advisor.")
    block2 = SimpleNamespace(type="server_tool_use", id="srv_01", name="advisor", input={})
    inner = SimpleNamespace(type="advisor_result", text="Use insertion sort for nearly-sorted arrays.")
    block3 = SimpleNamespace(type="advisor_tool_result", tool_use_id="srv_01", content=inner)
    block4 = SimpleNamespace(type="text", text="Based on advisor: insertion sort.")
    response = SimpleNamespace(
        content=[block1, block2, block3, block4],
        stop_reason="end_turn",
    )
    return response

def test_normalize_preserves_advisor_blocks():
    from agent.anthropic_adapter import normalize_anthropic_response
    response = _make_mock_response_with_advisor()
    msg, finish = normalize_anthropic_response(response)
    # text 합쳐짐
    assert "consult" in msg.content
    assert "insertion sort" in msg.content
    # advisor_native_blocks에 server_tool_use + advisor_tool_result 보존
    assert hasattr(msg, "advisor_native_blocks")
    block_types = [b["type"] for b in msg.advisor_native_blocks]
    assert "server_tool_use" in block_types
    assert "advisor_tool_result" in block_types

def test_normalize_advisor_result_text_extracted():
    from agent.anthropic_adapter import normalize_anthropic_response
    response = _make_mock_response_with_advisor()
    msg, _ = normalize_anthropic_response(response)
    advisor_block = next(b for b in msg.advisor_native_blocks if b["type"] == "advisor_tool_result")
    assert advisor_block["content"]["type"] == "advisor_result"
    assert "insertion sort" in advisor_block["content"]["text"]
```

**Step 2: 테스트 FAIL 확인**

```bash
python -m pytest tests/agent/test_anthropic_adapter.py -k "advisor" -v
```

**Step 3: normalize_anthropic_response() 수정**

기존 for loop에 advisor 블록 처리 추가:

```python
def normalize_anthropic_response(response, strip_tool_prefix: bool = False):
    text_parts = []
    reasoning_parts = []
    reasoning_details = []
    tool_calls = []
    advisor_native_blocks = []   # ← 신규

    for block in response.content:
        btype = block.type

        if btype == "text":
            text_parts.append(block.text)

        elif btype == "thinking":
            reasoning_parts.append(block.thinking)
            block_dict = _to_plain_data(block)
            if isinstance(block_dict, dict):
                reasoning_details.append(block_dict)

        elif btype == "tool_use":
            name = block.name
            if strip_tool_prefix and name.startswith(_MCP_TOOL_PREFIX):
                name = name[len(_MCP_TOOL_PREFIX):]
            tool_calls.append(SimpleNamespace(
                id=block.id,
                type="function",
                function=SimpleNamespace(name=name, arguments=json.dumps(block.input)),
            ))

        elif btype == "server_tool_use":
            advisor_native_blocks.append({
                "type": "server_tool_use",
                "id": block.id,
                "name": block.name,
                "input": getattr(block, "input", {}),
            })

        elif btype == "advisor_tool_result":
            inner = block.content
            inner_type = getattr(inner, "type", None)
            if inner_type == "advisor_result":
                inner_dict = {"type": "advisor_result", "text": inner.text}
            elif inner_type == "advisor_redacted_result":
                inner_dict = {
                    "type": "advisor_redacted_result",
                    "encrypted_content": getattr(inner, "encrypted_content", ""),
                }
            else:
                inner_dict = {"type": str(inner_type) if inner_type else "unknown"}
            advisor_native_blocks.append({
                "type": "advisor_tool_result",
                "tool_use_id": block.tool_use_id,
                "content": inner_dict,
            })

        elif btype == "advisor_tool_result_error":
            advisor_native_blocks.append({
                "type": "advisor_tool_result_error",
                "tool_use_id": getattr(block, "tool_use_id", ""),
                "error_code": getattr(block, "error_code", "unknown"),
            })

    stop_reason_map = {
        "end_turn": "stop", "tool_use": "tool_calls",
        "max_tokens": "length", "stop_sequence": "stop",
    }
    finish_reason = stop_reason_map.get(response.stop_reason, "stop")

    return (
        SimpleNamespace(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            reasoning="\n\n".join(reasoning_parts) if reasoning_parts else None,
            reasoning_content=None,
            reasoning_details=reasoning_details or None,
            advisor_native_blocks=advisor_native_blocks or None,  # ← 신규
        ),
        finish_reason,
    )
```

**Step 4: 테스트 PASS 확인**

```bash
python -m pytest tests/agent/test_anthropic_adapter.py -v
```

**Step 5: Commit**

```bash
git add agent/anthropic_adapter.py tests/agent/test_anthropic_adapter.py
git commit -m "feat: parse advisor_tool_result blocks in normalize_anthropic_response"
```

---

## Task 5: build_anthropic_kwargs() — advisor beta 헤더 파라미터 추가

**Objective:** `build_anthropic_kwargs()`에 `native_advisor: bool = False` 파라미터를 추가해, True일 때 `extra_headers`에 `advisor-tool-2026-03-01` beta를 주입한다. 기존 fast_mode beta 패턴과 동일한 방식.

**Files:**
- Modify: `agent/anthropic_adapter.py` — `build_anthropic_kwargs()` 시그니처 + 본체
- Test: `tests/agent/test_anthropic_adapter.py`

**Step 1: 실패 테스트 작성**

```python
def test_build_anthropic_kwargs_adds_advisor_beta():
    from agent.anthropic_adapter import build_anthropic_kwargs
    kwargs = build_anthropic_kwargs(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-6", "max_uses": 1}],
        max_tokens=100,
        reasoning_config=None,
        native_advisor=True,
    )
    beta_header = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
    assert "advisor-tool-2026-03-01" in beta_header
    # advisor 툴이 tools 배열에 그대로 있어야 함
    tools = kwargs.get("tools", [])
    assert any(t.get("type") == "advisor_20260301" for t in tools)
```

**Step 2: 테스트 FAIL 확인**

```bash
python -m pytest tests/agent/test_anthropic_adapter.py::test_build_anthropic_kwargs_adds_advisor_beta -v
```

**Step 3: build_anthropic_kwargs() 수정**

시그니처에 `native_advisor: bool = False` 추가 후 함수 하단(fast_mode 블록 근처)에 추가:

```python
# ── Native advisor beta header ──────────────────────────────────────
if native_advisor and not _is_third_party_anthropic_endpoint(base_url):
    _ADVISOR_BETA = "advisor-tool-2026-03-01"
    existing_headers = kwargs.get("extra_headers", {})
    existing_beta = existing_headers.get("anthropic-beta", "")
    if _ADVISOR_BETA not in existing_beta:
        betas_list = [b for b in existing_beta.split(",") if b] if existing_beta else []
        # 공통 betas 포함 (fast_mode가 없을 경우 extra_headers 미생성 상태일 수 있음)
        if not betas_list:
            betas_list = list(_common_betas_for_base_url(base_url))
            if is_oauth:
                betas_list.extend(_OAUTH_ONLY_BETAS)
        betas_list.append(_ADVISOR_BETA)
        kwargs["extra_headers"] = {
            **existing_headers,
            "anthropic-beta": ",".join(betas_list),
        }
```

**Step 4: 테스트 PASS 확인**

```bash
python -m pytest tests/agent/test_anthropic_adapter.py -v
```

**Step 5: Commit**

```bash
git add agent/anthropic_adapter.py tests/agent/test_anthropic_adapter.py
git commit -m "feat: add native_advisor param to build_anthropic_kwargs for beta header injection"
```

---

## Task 6: run_agent.py — advisor 툴 주입 + _build_api_kwargs() 연결

**Objective:** `_build_api_kwargs()`에서 native advisor 활성 시 advisor 툴을 `build_anthropic_kwargs()`에 전달하고 beta 헤더를 주입한다. `self.tools`는 오염시키지 않는다.

**Files:**
- Modify: `run_agent.py` — `_build_api_kwargs()`
- Test: `tests/agent/test_native_advisor.py` (신규)

**Step 1: 실패 테스트 작성**

`tests/agent/test_native_advisor.py` 신규 생성:

```python
import pytest
from unittest.mock import MagicMock, patch

def _make_minimal_agent():
    from run_agent import AIAgent
    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"
    agent.model = "claude-haiku-4-5-20251001"
    agent.max_tokens = 1024
    agent.reasoning_config = None
    agent._is_anthropic_oauth = False
    agent._anthropic_base_url = None
    agent.tools = [{"function": {"name": "terminal", "description": "run", "parameters": {}}}]
    agent.request_overrides = {}
    agent.context_compressor = None
    agent._ephemeral_max_output_tokens = None
    return agent

def test_build_api_kwargs_injects_advisor_tool_when_native():
    agent = _make_minimal_agent()
    advisor_cfg = {"enabled": True, "mode": "native", "model": "claude-opus-4-6", "max_uses": 1}

    with patch("agent.advisor_config.load_advisor_config", return_value=advisor_cfg):
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])

    tools = kwargs.get("tools", [])
    assert any(t.get("type") == "advisor_20260301" for t in tools), "advisor tool not injected"
    beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
    assert "advisor-tool-2026-03-01" in beta, "advisor beta header not present"

def test_build_api_kwargs_no_advisor_when_external_mode():
    agent = _make_minimal_agent()
    advisor_cfg = {"enabled": True, "mode": "external", "model": "claude-opus-4-6"}

    with patch("agent.advisor_config.load_advisor_config", return_value=advisor_cfg):
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])

    tools = kwargs.get("tools", [])
    assert not any(t.get("type") == "advisor_20260301" for t in tools)

def test_build_api_kwargs_self_tools_not_mutated():
    """advisor 툴 주입이 self.tools를 변경하지 않아야 함."""
    agent = _make_minimal_agent()
    original_tools_count = len(agent.tools)
    advisor_cfg = {"enabled": True, "mode": "native", "model": "claude-opus-4-6", "max_uses": 1}

    with patch("agent.advisor_config.load_advisor_config", return_value=advisor_cfg):
        agent._build_api_kwargs([{"role": "user", "content": "hi"}])

    assert len(agent.tools) == original_tools_count, "self.tools was mutated"
```

**Step 2: 테스트 FAIL 확인**

```bash
python -m pytest tests/agent/test_native_advisor.py -v
```

**Step 3: run_agent.py `_build_api_kwargs()` 수정**

`anthropic_messages` 분기 안에서 `build_anthropic_kwargs()` 호출 직전:

```python
def _build_api_kwargs(self, api_messages: list) -> dict:
    if self.api_mode == "anthropic_messages":
        from agent.anthropic_adapter import build_anthropic_kwargs
        from agent.advisor_config import load_advisor_config, native_advisor_applicable

        anthropic_messages = self._prepare_anthropic_messages_for_api(api_messages)
        ctx_len = getattr(self, "context_compressor", None)
        ctx_len = ctx_len.context_length if ctx_len else None
        ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
        if ephemeral_out is not None:
            self._ephemeral_max_output_tokens = None

        # --- native advisor 툴 주입 (self.tools 불변) ---
        tools_for_call = list(self.tools or [])
        native_advisor = False
        try:
            adv_cfg = load_advisor_config()
            if native_advisor_applicable(adv_cfg, provider=self.provider):
                native_advisor = True
                advisor_model = str(adv_cfg.get("model") or "claude-opus-4-6")
                max_uses = int(adv_cfg.get("max_uses") or 1)
                # 멀티턴 강제 검사 + 신규 주입 둘 다 처리
                has_advisor_in_history = any(
                    isinstance(msg.get("content"), list) and
                    any(isinstance(b, dict) and b.get("type") == "advisor_tool_result"
                        for b in msg["content"])
                    for msg in api_messages
                )
                if not any(t.get("type") == "advisor_20260301" for t in tools_for_call):
                    tools_for_call = tools_for_call + [{
                        "type": "advisor_20260301",
                        "name": "advisor",
                        "model": advisor_model,
                        "max_uses": max_uses,
                    }]
                elif has_advisor_in_history:
                    # 이미 있음 — multi-turn constraint 자동 충족
                    pass
        except ValueError as exc:
            logger.warning("Native advisor config error (skipping): %s", exc)
        except Exception as exc:
            logger.debug("Native advisor setup skipped: %s", exc)
        # ------------------------------------------------

        return build_anthropic_kwargs(
            model=self.model,
            messages=anthropic_messages,
            tools=tools_for_call,       # ← self.tools 대신 tools_for_call
            max_tokens=ephemeral_out if ephemeral_out is not None else self.max_tokens,
            reasoning_config=self.reasoning_config,
            is_oauth=self._is_anthropic_oauth,
            preserve_dots=self._anthropic_preserve_dots(),
            context_length=ctx_len,
            base_url=getattr(self, "_anthropic_base_url", None),
            fast_mode=(self.request_overrides or {}).get("speed") == "fast",
            native_advisor=native_advisor,   # ← 신규
        )
    # ... 나머지 분기 불변
```

**Step 4: 테스트 PASS 확인**

```bash
python -m pytest tests/agent/test_native_advisor.py -v
```

**Step 5: Commit**

```bash
git add run_agent.py tests/agent/test_native_advisor.py
git commit -m "feat: inject native advisor tool in _build_api_kwargs without mutating self.tools"
```

---

## Task 7: run_agent.py — advisor 블록 히스토리 보존 + fallback 시 제거

**Objective:** `normalize_anthropic_response()`가 반환한 `advisor_native_blocks`를 메시지 히스토리에 보존하고, fallback 발동 시 비-Anthropic provider를 위해 제거한다.

**Files:**
- Modify: `run_agent.py`
- Test: `tests/agent/test_native_advisor.py`

**Step 1: 히스토리 저장 위치 파악**

```bash
grep -n "normalize_anthropic_response\|assistant_message\|messages.append.*role.*assistant" \
  ~/.hermes/hermes-agent/run_agent.py | head -20
```

**Step 2: 실패 테스트 작성**

```python
def test_advisor_native_blocks_stored_in_history():
    """normalize_anthropic_response가 반환한 advisor_native_blocks가 히스토리 메시지에 포함되는지."""
    from types import SimpleNamespace
    # advisor_native_blocks를 가진 mock assistant_message
    mock_msg = SimpleNamespace(
        content="Based on advisor: insertion sort.",
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
        advisor_native_blocks=[
            {"type": "server_tool_use", "id": "srv_01", "name": "advisor", "input": {}},
            {"type": "advisor_tool_result", "tool_use_id": "srv_01",
             "content": {"type": "advisor_result", "text": "Use insertion sort."}},
        ],
    )
    agent = _make_minimal_agent()
    messages = [{"role": "user", "content": "best sort?"}]
    agent._append_assistant_turn_to_history(messages, mock_msg, finish_reason="stop")

    assert len(messages) == 2
    content = messages[1].get("content")
    if isinstance(content, list):
        types = [b.get("type") for b in content if isinstance(b, dict)]
        assert "advisor_tool_result" in types
    else:
        # text-only 저장 시 최소한 content가 있어야 함
        assert content
```

**Step 3: `_append_assistant_turn_to_history()` 헬퍼 추가 또는 기존 히스토리 저장 로직 수정**

기존 코드에서 assistant 메시지를 messages에 어펜드하는 지점을 찾아 수정:

```python
# advisor_native_blocks가 있으면 content를 list 형태로 확장
adv_blocks = getattr(assistant_message, "advisor_native_blocks", None)
if adv_blocks and self.api_mode == "anthropic_messages":
    text_content = assistant_message.content or ""
    content_list = []
    if text_content:
        content_list.append({"type": "text", "text": text_content})
    # server_tool_use 블록 삽입 (advisor_tool_result 앞에 와야 함)
    for blk in adv_blocks:
        content_list.append(blk)
    # 최종 text가 advisor_tool_result 이후에 있으면 별도 text 블록으로 추가
    # (이미 text_content에 합쳐져 있으므로 중복 삽입 없음)
    messages.append({"role": "assistant", "content": content_list})
else:
    # 기존 경로 — 변경 없음
    messages.append({"role": "assistant", "content": assistant_message.content or ""})
```

**fallback 시 advisor 블록 제거:**

```python
def _strip_advisor_blocks_from_history(self, messages: list) -> list:
    """
    fallback 발동 시 비-Anthropic provider를 위해 advisor_tool_result 블록 제거.
    advisor_tool_result / server_tool_use 블록을 포함한 content list를
    text-only 형태로 평탄화한다.
    """
    cleaned = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = [
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if text_parts:
                cleaned.append({**msg, "content": "\n".join(text_parts)})
            elif any(isinstance(b, dict) and b.get("type") not in ("server_tool_use", "advisor_tool_result", "advisor_tool_result_error") for b in content):
                cleaned.append({**msg, "content": [
                    b for b in content
                    if isinstance(b, dict) and b.get("type") not in
                    ("server_tool_use", "advisor_tool_result", "advisor_tool_result_error")
                ]})
            else:
                cleaned.append({**msg, "content": ""})
        else:
            cleaned.append(msg)
    return cleaned
```

fallback 발동 코드 근처(`_fallback_activated` 관련)에서 호출:

```python
if self._fallback_activated:
    messages = self._strip_advisor_blocks_from_history(messages)
```

**Step 4: 테스트 PASS 확인**

```bash
python -m pytest tests/agent/test_native_advisor.py -v
```

**Step 5: Commit**

```bash
git add run_agent.py tests/agent/test_native_advisor.py
git commit -m "feat: preserve advisor_native_blocks in history, strip on provider fallback"
```

---

## Task 8: 전체 테스트 스위트 + smoke test

**Objective:** 기존 테스트가 깨지지 않았는지 확인하고, config 활성화 후 실제 동작 검증.

**Step 1: 전체 테스트 스위트**

```bash
cd ~/.hermes/hermes-agent && source venv/bin/activate
python -m pytest tests/ -q --timeout=60 2>&1 | tail -30
```

Expected: 기존 실패 6개 그대로, 신규 실패 없음.

**Step 2: config 활성화**

`~/.hermes/config.yaml`:

```yaml
advisor:
  enabled: true
  mode: native
  model: claude-opus-4-6
  max_uses: 1
```

**Step 3: CLI smoke test**

```bash
# provider가 anthropic인지 확인
hermes /advisor status

# advisor 블록 로그 확인용 (debug 레벨)
HERMES_LOG_LEVEL=debug hermes "Design a lock-free ring buffer in C++ with wait-free reads"
```

기대 동작: 응답 생성 중 advisor 개입, 로그에 `advisor_tool_result` 블록 확인.

**Step 4: smoke test 후 config 원복**

```yaml
advisor:
  enabled: false
  mode: external
```

**Step 5: Final commit**

```bash
git commit -m "test: native advisor Phase 3 integration verified"
```

---

## 변경 파일 요약

| 파일 | 변경 내용 |
|------|-----------|
| `hermes_cli/config.py` | `mode` 주석에 native/auto 추가, `max_uses: 1` 필드 추가, version 18→19 |
| `agent/advisor_config.py` | `native_advisor_applicable()` 추가 |
| `agent/anthropic_adapter.py` | `convert_tools_to_anthropic()` server tool bypass, `normalize_anthropic_response()` advisor 블록 파싱, `build_anthropic_kwargs()` `native_advisor` 파라미터 + beta 헤더 주입 |
| `run_agent.py` | `_build_api_kwargs()` advisor 툴 주입, `_strip_advisor_blocks_from_history()` 추가, fallback 시 호출, 히스토리 advisor 블록 보존 |
| `tests/hermes_cli/test_config_env_expansion.py` | Task 1 테스트 추가 |
| `tests/agent/test_advisor_config.py` | Task 2 테스트 추가 |
| `tests/agent/test_anthropic_adapter.py` | Task 3, 4, 5 테스트 추가 |
| `tests/agent/test_native_advisor.py` | 신규 — Task 6, 7 통합 테스트 |

**변경하지 않는 파일:** `advisor_runner.py`, `advisor_policy.py`, `context_compressor.py`, `gateway/`
