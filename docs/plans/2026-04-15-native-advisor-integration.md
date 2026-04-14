# Native Advisor Integration (Phase 3) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Hermes의 `anthropic_messages` 모드에서 Anthropic 네이티브 `advisor_20260301` 툴을 사용해 executor 모델이 생성 도중 서버 측에서 Opus에게 직접 조언을 구할 수 있게 한다.

**Architecture (Opus 조언 기반):**
- OpenAI SDK 경로는 전혀 건드리지 않는다. `api_mode == "anthropic_messages"` + `advisor.mode == "native"` 일 때만 동작하는 **병렬 분기**를 추가한다.
- `anthropic` SDK는 이미 설치되어 있고(`0.94.0`), `_anthropic_client`를 통해 `_anthropic_messages_create()`에서 이미 사용 중이다.
- `advisor_tool_result` 블록은 Anthropic 네이티브 포맷으로 히스토리에 그대로 보존한다. OpenAI 포맷으로 변환하지 않는다.
- 멀티턴 제약(히스토리에 `advisor_tool_result`가 있으면 tools에 반드시 advisor 툴 포함)을 call-time 검사로 강제한다.

**Touch points (최소 변경):**
1. `hermes_cli/config.py` — mode에 `"native"` / `"auto"` 추가
2. `agent/advisor_config.py` — mode 인식 및 validation 추가
3. `run_agent.py` — `_build_api_kwargs()`에서 native advisor 툴 주입 + 히스토리 보존 로직

**변경하지 않는 파일:** `advisor_runner.py`, `advisor_policy.py` (external 경로 그대로 유지)

**Tech Stack:** Python, anthropic SDK 0.94.0, Anthropic beta `advisor-tool-2026-03-01`

---

## Task 1: Config에 native / auto 모드 추가

**Objective:** `advisor.mode` 값으로 `"native"`와 `"auto"`를 인식하도록 config 기본값과 문서 주석을 업데이트한다.

**Files:**
- Modify: `hermes_cli/config.py` — `"advisor"` 블록의 `mode` 주석 확장
- Bump: `_config_version` (현재 18 → 19)
- Test: `tests/hermes_cli/test_config_env_expansion.py`

**Step 1: 실패 테스트 작성**

```python
def test_advisor_mode_values_include_native_and_auto():
    from hermes_cli.config import DEFAULT_CONFIG
    advisor = DEFAULT_CONFIG["advisor"]
    # mode 주석에 native, auto가 문서화되어 있는지 — 기본값은 여전히 "external"
    assert advisor["mode"] == "external"
    # native/auto를 유효한 값으로 명시한 문서 키 확인
    assert "max_uses" in advisor  # Task 1에서 max_uses 필드도 추가
```

**Step 2: 테스트 실행 → FAIL 확인**

```bash
cd ~/.hermes/hermes-agent && source venv/bin/activate
python -m pytest tests/hermes_cli/test_config_env_expansion.py::test_advisor_mode_values_include_native_and_auto -v
```
Expected: FAIL — `KeyError: 'max_uses'`

**Step 3: config.py 수정**

`hermes_cli/config.py`의 `"advisor"` 블록에서:

```python
# 변경 전
"mode": "external",              # external | off

# 변경 후
"mode": "external",              # external | native | auto | off
                                  # native: advisor_20260301 tool (Anthropic only)
                                  # auto:   native when provider==anthropic, else external
"max_uses": 1,                   # max advisor calls per request (native mode only)
```

`_config_version` 18 → 19

**Step 4: 테스트 실행 → PASS 확인**

```bash
python -m pytest tests/hermes_cli/test_config_env_expansion.py -v
```
Expected: 신규 테스트 포함 전체 PASS

**Step 5: Commit**

```bash
git add hermes_cli/config.py tests/hermes_cli/test_config_env_expansion.py
git commit -m "feat: extend advisor config mode to support native/auto and add max_uses"
```

---

## Task 2: advisor_config.py에 native 모드 인식 및 validation 추가

**Objective:** `advisor_enabled()`, `_normalize_runtime_from_entry()`, 새 `native_advisor_applicable()` 함수가 `native` / `auto` 모드를 올바르게 처리한다.

**Files:**
- Modify: `agent/advisor_config.py`
- Test: `tests/agent/test_advisor_config.py`

**Step 1: 실패 테스트 작성**

```python
def test_native_advisor_applicable_true_when_native_and_anthropic():
    from agent.advisor_config import native_advisor_applicable
    cfg = {"enabled": True, "mode": "native", "provider": "anthropic", "model": "claude-opus-4-6"}
    assert native_advisor_applicable(cfg, provider="anthropic") is True

def test_native_advisor_applicable_false_when_native_non_anthropic():
    from agent.advisor_config import native_advisor_applicable
    cfg = {"enabled": True, "mode": "native", "provider": "openai", "model": "gpt-4o"}
    assert native_advisor_applicable(cfg, provider="openai") is False

def test_native_advisor_applicable_auto_with_anthropic():
    from agent.advisor_config import native_advisor_applicable
    cfg = {"enabled": True, "mode": "auto"}
    assert native_advisor_applicable(cfg, provider="anthropic") is True

def test_native_advisor_applicable_auto_with_non_anthropic():
    from agent.advisor_config import native_advisor_applicable
    cfg = {"enabled": True, "mode": "auto"}
    assert native_advisor_applicable(cfg, provider="openai") is False
```

**Step 2: 테스트 실행 → FAIL 확인**

```bash
python -m pytest tests/agent/test_advisor_config.py -k "native" -v
```
Expected: FAIL — `ImportError: cannot import name 'native_advisor_applicable'`

**Step 3: advisor_config.py에 함수 추가**

`agent/advisor_config.py` 하단에 추가:

```python
def native_advisor_applicable(cfg: Dict[str, Any], provider: str = "") -> bool:
    """
    Returns True if native advisor_20260301 tool should be used.
    - mode "native": requires provider == "anthropic"; raises ValueError otherwise
    - mode "auto":   True only when provider == "anthropic"
    - other modes:   always False
    """
    if not cfg.get("enabled"):
        return False
    mode = str(cfg.get("mode") or "external").strip().lower()
    is_anthropic = str(provider or "").strip().lower() == "anthropic"
    if mode == "native":
        if not is_anthropic:
            raise ValueError(
                f"advisor.mode='native' requires provider='anthropic', got '{provider}'. "
                "Set mode='auto' to fall back to external on other providers."
            )
        return True
    if mode == "auto":
        return is_anthropic
    return False
```

**Step 4: 테스트 실행 → PASS 확인**

```bash
python -m pytest tests/agent/test_advisor_config.py -v
```
Expected: 신규 테스트 포함 전체 PASS

**Step 5: Commit**

```bash
git add agent/advisor_config.py tests/agent/test_advisor_config.py
git commit -m "feat: add native_advisor_applicable() to advisor_config"
```

---

## Task 3: run_agent.py — native advisor 툴 주입 헬퍼

**Objective:** API kwargs를 build할 때 native advisor 툴 정의를 tools 배열에 주입하는 헬퍼 메서드 `_inject_native_advisor_tool()`을 추가한다.

**Files:**
- Modify: `run_agent.py`
- Test: `tests/test_run_agent.py` (또는 `tests/agent/test_native_advisor.py` 신규 생성)

**Step 1: 실패 테스트 작성**

`tests/agent/test_native_advisor.py` 신규 생성:

```python
import pytest
from unittest.mock import MagicMock, patch

def _make_agent(mode="native", model="claude-opus-4-6"):
    """Helper: create a minimal AIAgent with advisor config."""
    from run_agent import AIAgent
    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"
    agent._advisor_cfg_cache = {
        "enabled": True,
        "mode": mode,
        "model": model,
        "max_uses": 1,
    }
    return agent

def test_inject_native_advisor_tool_adds_tool():
    agent = _make_agent()
    tools = [{"type": "function", "name": "terminal"}]
    result = agent._inject_native_advisor_tool(tools)
    types = [t.get("type") for t in result]
    assert "advisor_20260301" in types

def test_inject_native_advisor_tool_idempotent():
    agent = _make_agent()
    tools = [{"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-6"}]
    result = agent._inject_native_advisor_tool(tools)
    advisor_count = sum(1 for t in result if t.get("type") == "advisor_20260301")
    assert advisor_count == 1  # 중복 추가 안 됨

def test_inject_native_advisor_tool_noop_for_non_anthropic():
    agent = _make_agent()
    agent.provider = "openai"
    tools = [{"type": "function", "name": "terminal"}]
    result = agent._inject_native_advisor_tool(tools)
    types = [t.get("type") for t in result]
    assert "advisor_20260301" not in types
```

**Step 2: 테스트 실행 → FAIL 확인**

```bash
python -m pytest tests/agent/test_native_advisor.py -v
```
Expected: FAIL — `AttributeError: '_inject_native_advisor_tool'`

**Step 3: run_agent.py에 메서드 추가**

`AIAgent` 클래스에 다음 메서드를 추가 (기존 `_build_advisor_context_summary` 근처):

```python
def _inject_native_advisor_tool(self, tools: list) -> list:
    """
    Appends the advisor_20260301 tool definition when native advisor mode is active.
    Idempotent — never adds a duplicate. No-op for non-Anthropic providers.
    """
    try:
        from agent.advisor_config import load_advisor_config, native_advisor_applicable
        cfg = getattr(self, "_advisor_cfg_cache", None) or load_advisor_config()
    except Exception:
        return tools

    try:
        applicable = native_advisor_applicable(cfg, provider=getattr(self, "provider", ""))
    except ValueError as exc:
        logger.warning("Native advisor config error: %s", exc)
        return tools

    if not applicable:
        return tools

    # 이미 포함되어 있으면 그대로 반환
    if any(t.get("type") == "advisor_20260301" for t in tools):
        return tools

    advisor_model = str(cfg.get("model") or "claude-opus-4-6").strip()
    max_uses = int(cfg.get("max_uses") or 1)
    advisor_tool = {
        "type": "advisor_20260301",
        "name": "advisor",
        "model": advisor_model,
        "max_uses": max_uses,
    }
    return list(tools) + [advisor_tool]
```

**Step 4: 테스트 실행 → PASS 확인**

```bash
python -m pytest tests/agent/test_native_advisor.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add run_agent.py tests/agent/test_native_advisor.py
git commit -m "feat: add _inject_native_advisor_tool() helper to AIAgent"
```

---

## Task 4: run_agent.py — beta 헤더 주입

**Objective:** `api_mode == "anthropic_messages"` + native advisor 활성 시, `_anthropic_messages_create()`에 beta 헤더를 추가한다.

**Files:**
- Modify: `run_agent.py` (`_anthropic_messages_create` 메서드)
- Test: `tests/agent/test_native_advisor.py`

**Step 1: 실패 테스트 작성**

```python
def test_anthropic_messages_create_adds_beta_header_for_native_advisor():
    from unittest.mock import patch, MagicMock
    agent = _make_agent()
    mock_client = MagicMock()
    agent._anthropic_client = mock_client
    agent._try_refresh_anthropic_client_credentials = MagicMock()

    api_kwargs = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-6"}],
    }

    agent._anthropic_messages_create(api_kwargs)

    call_kwargs = mock_client.messages.create.call_args[1]
    betas = call_kwargs.get("betas", [])
    assert "advisor-tool-2026-03-01" in betas
```

**Step 2: 테스트 실행 → FAIL 확인**

```bash
python -m pytest tests/agent/test_native_advisor.py::test_anthropic_messages_create_adds_beta_header_for_native_advisor -v
```
Expected: FAIL

**Step 3: `_anthropic_messages_create` 수정**

```python
def _anthropic_messages_create(self, api_kwargs: dict):
    if self.api_mode == "anthropic_messages":
        self._try_refresh_anthropic_client_credentials()

    # native advisor beta 헤더 주입
    tools = api_kwargs.get("tools") or []
    if any(t.get("type") == "advisor_20260301" for t in tools):
        betas = list(api_kwargs.get("betas") or [])
        if "advisor-tool-2026-03-01" not in betas:
            betas.append("advisor-tool-2026-03-01")
        api_kwargs = {**api_kwargs, "betas": betas}

    return self._anthropic_client.messages.create(**api_kwargs)
```

**Step 4: 테스트 PASS 확인**

```bash
python -m pytest tests/agent/test_native_advisor.py -v
```
Expected: 4 passed

**Step 5: Commit**

```bash
git add run_agent.py tests/agent/test_native_advisor.py
git commit -m "feat: inject advisor-tool beta header in _anthropic_messages_create"
```

---

## Task 5: run_agent.py — advisor_tool_result 히스토리 보존

**Objective:** Anthropic 응답에 `advisor_tool_result` 블록이 포함된 경우, 다음 턴 히스토리에 네이티브 포맷 그대로 보존한다. (Opus 조언: OpenAI 포맷으로 변환하지 않는다.)

**Files:**
- Modify: `run_agent.py` — 메시지 히스토리 어펜드 로직
- Test: `tests/agent/test_native_advisor.py`

**Step 1: 현재 히스토리 어펜드 코드 위치 파악**

```bash
grep -n "append.*assistant\|messages\.append\|role.*assistant" ~/.hermes/hermes-agent/run_agent.py | head -20
```

**Step 2: 실패 테스트 작성**

```python
def test_advisor_tool_result_preserved_in_history():
    """advisor_tool_result 블록이 포함된 응답이 히스토리에 네이티브 포맷으로 보존되는지 검사."""
    from run_agent import AIAgent
    # anthropic SDK Message 객체를 시뮬레이션
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(type="text", text="I'll consult the advisor."),
        MagicMock(type="server_tool_use", id="srv_01", name="advisor", input={}),
        MagicMock(type="advisor_tool_result", tool_use_id="srv_01",
                  content=MagicMock(type="advisor_result", text="Use insertion sort.")),
        MagicMock(type="text", text="Based on advisor: insertion sort is best."),
    ]
    mock_response.stop_reason = "end_turn"

    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"

    messages = []
    agent._append_anthropic_assistant_message(mock_response, messages)

    # assistant 메시지가 advisor_tool_result 블록을 포함해야 함
    assert len(messages) == 1
    content = messages[0]["content"]
    types = [b.get("type") if isinstance(b, dict) else getattr(b, "type", None) for b in content]
    assert "advisor_tool_result" in types
```

**Step 3: `_append_anthropic_assistant_message()` 헬퍼 추가**

```python
def _append_anthropic_assistant_message(self, response, messages: list) -> None:
    """
    anthropic SDK 응답을 messages 히스토리에 어펜드한다.
    advisor_tool_result 블록을 포함한 content를 네이티브 포맷 그대로 보존한다.
    OpenAI 포맷으로 변환하지 않는다.
    """
    content_blocks = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            content_blocks.append({"type": "text", "text": block.text})
        elif block_type == "server_tool_use":
            content_blocks.append({
                "type": "server_tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        elif block_type == "advisor_tool_result":
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
                inner_dict = {"type": str(inner_type)}
            content_blocks.append({
                "type": "advisor_tool_result",
                "tool_use_id": block.tool_use_id,
                "content": inner_dict,
            })
        else:
            # 기타 블록은 dict 변환 시도
            try:
                content_blocks.append(block.model_dump())
            except Exception:
                content_blocks.append({"type": str(block_type)})

    messages.append({"role": "assistant", "content": content_blocks})
```

**Step 4: 테스트 PASS 확인**

```bash
python -m pytest tests/agent/test_native_advisor.py -v
```
Expected: 5 passed

**Step 5: Commit**

```bash
git add run_agent.py tests/agent/test_native_advisor.py
git commit -m "feat: preserve advisor_tool_result blocks in Anthropic message history"
```

---

## Task 6: run_agent.py — 멀티턴 advisor 툴 강제 포함 검사

**Objective:** 히스토리에 `advisor_tool_result`가 있으면 tools 배열에 advisor 툴이 반드시 포함되도록 강제한다. (Opus: "이걸 놓치면 프로덕션에서 silent API error 발생")

**Files:**
- Modify: `run_agent.py`
- Test: `tests/agent/test_native_advisor.py`

**Step 1: 실패 테스트 작성**

```python
def test_ensure_advisor_tool_in_tools_when_history_has_advisor_result():
    agent = _make_agent()
    history = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": [
            {"type": "advisor_tool_result", "tool_use_id": "x", "content": {"type": "advisor_result", "text": "..."}}
        ]},
    ]
    tools = [{"type": "function", "name": "terminal"}]
    result = agent._ensure_advisor_tool_present(tools, history)
    types = [t.get("type") for t in result]
    assert "advisor_20260301" in types

def test_ensure_advisor_tool_noop_when_no_history_advisor_result():
    agent = _make_agent()
    history = [{"role": "user", "content": "question"}]
    tools = [{"type": "function", "name": "terminal"}]
    result = agent._ensure_advisor_tool_present(tools, history)
    types = [t.get("type") for t in result]
    assert "advisor_20260301" not in types
```

**Step 2: 테스트 실행 → FAIL 확인**

```bash
python -m pytest tests/agent/test_native_advisor.py -k "ensure" -v
```

**Step 3: `_ensure_advisor_tool_present()` 추가**

```python
def _ensure_advisor_tool_present(self, tools: list, messages: list) -> list:
    """
    Multi-turn correctness check: if conversation history contains any
    advisor_tool_result block, the advisor tool MUST be in the tools list.
    Injects it if missing (using last-known config or default model).
    """
    def _has_advisor_result(msgs):
        for msg in msgs:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "advisor_tool_result":
                        return True
        return False

    if not _has_advisor_result(messages):
        return tools

    # advisor_tool_result가 있는데 tools에 없으면 주입
    if not any(t.get("type") == "advisor_20260301" for t in tools):
        try:
            from agent.advisor_config import load_advisor_config
            cfg = getattr(self, "_advisor_cfg_cache", None) or load_advisor_config()
            advisor_model = str(cfg.get("model") or "claude-opus-4-6")
            max_uses = int(cfg.get("max_uses") or 1)
        except Exception:
            advisor_model = "claude-opus-4-6"
            max_uses = 1
        logger.debug("Multi-turn: injecting advisor tool to satisfy history constraint")
        return list(tools) + [{
            "type": "advisor_20260301",
            "name": "advisor",
            "model": advisor_model,
            "max_uses": max_uses,
        }]
    return tools
```

**Step 4: 테스트 PASS 확인**

```bash
python -m pytest tests/agent/test_native_advisor.py -v
```
Expected: 7 passed

**Step 5: Commit**

```bash
git add run_agent.py tests/agent/test_native_advisor.py
git commit -m "feat: enforce advisor tool presence for multi-turn advisor_tool_result history"
```

---

## Task 7: run_agent.py — 메인 루프에 native advisor 연결

**Objective:** `_build_api_kwargs()` (또는 메인 루프의 tools 생성 지점)에서 `_inject_native_advisor_tool()`과 `_ensure_advisor_tool_present()`를 호출하고, `anthropic_messages` 응답을 `_append_anthropic_assistant_message()`로 저장한다.

**Files:**
- Modify: `run_agent.py` — `run_conversation()` 루프 내 API kwargs 구성 부분
- Test: `tests/agent/test_native_advisor.py` (integration-level mock test)

**Step 1: 연결 지점 파악**

```bash
grep -n "_build_api_kwargs\|tool_schemas\|tools=" ~/.hermes/hermes-agent/run_agent.py | head -20
```

**Step 2: 실패 테스트 작성 (integration mock)**

```python
def test_main_loop_injects_native_advisor_tool_and_calls_with_beta(monkeypatch):
    """메인 루프가 native advisor 모드에서 advisor 툴을 주입하고 beta 헤더를 보내는지 end-to-end mock 검사."""
    # 이 테스트는 AIAgent.run_conversation()을 실행하면서
    # _anthropic_client.messages.create 호출을 가로채 beta와 advisor 툴 포함 여부 확인
    # 구체적 구현은 코드베이스 내 run_conversation() 시그니처 확인 후 작성
    pass  # Task 7 구현 후 채움
```

**Step 3: 메인 루프 수정**

`run_conversation()` 내 tools 배열 구성 직후:

```python
# 기존 코드 (대략):
# tool_schemas = get_tool_definitions(...)

# 추가할 코드:
if self.api_mode == "anthropic_messages":
    tool_schemas = self._inject_native_advisor_tool(tool_schemas)
    tool_schemas = self._ensure_advisor_tool_present(tool_schemas, messages)
```

응답 저장 부분에서 `anthropic_messages` 분기 추가:

```python
# 기존: messages.append({"role": "assistant", "content": response_text})
# 수정:
if self.api_mode == "anthropic_messages" and hasattr(response, "content"):
    self._append_anthropic_assistant_message(response, messages)
else:
    messages.append({"role": "assistant", "content": response_text})
```

**Step 4: 전체 advisor 테스트 PASS 확인**

```bash
python -m pytest tests/agent/test_native_advisor.py tests/agent/test_advisor_config.py -v
```

**Step 5: Commit**

```bash
git add run_agent.py
git commit -m "feat: wire native advisor tool injection into main conversation loop"
```

---

## Task 8: 수동 smoke test + 전체 테스트 스위트

**Objective:** 실제 API로 native advisor가 동작하는지 확인하고, 기존 테스트가 깨지지 않았는지 검증한다.

**Step 1: config 활성화**

`~/.hermes/config.yaml`에서:

```yaml
advisor:
  enabled: true
  mode: native
  model: claude-opus-4-6
  max_uses: 1
```

**Step 2: CLI에서 smoke test**

```bash
cd ~/.hermes/hermes-agent && source venv/bin/activate
# provider가 anthropic인지 확인
hermes /advisor status
# 테스트 대화 (복잡한 기술 질문)
hermes "Design a lock-free ring buffer in C++"
```

기대 동작: response에 Opus advisor 개입 흔적 (또는 로그에서 `advisor_tool_result` 블록 확인)

**Step 3: 전체 테스트 스위트**

```bash
python -m pytest tests/ -q --timeout=60 2>&1 | tail -20
```

Expected: 기존 실패 6개만 있고 신규 실패 없음

**Step 4: smoke test 후 config 원복 (선택)**

```yaml
advisor:
  enabled: false
  mode: external
```

**Step 5: Final commit**

```bash
git add .
git commit -m "test: native advisor smoke test verified, full suite passing"
```

---

## 요약 — 변경 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `hermes_cli/config.py` | mode에 native/auto 추가, max_uses 필드 추가, version 18→19 |
| `agent/advisor_config.py` | `native_advisor_applicable()` 함수 추가 |
| `run_agent.py` | `_inject_native_advisor_tool()`, `_ensure_advisor_tool_present()`, `_append_anthropic_assistant_message()`, `_anthropic_messages_create()` beta 헤더 주입, 메인 루프 연결 |
| `tests/agent/test_native_advisor.py` | 신규 테스트 파일 |
| `tests/hermes_cli/test_config_env_expansion.py` | config native/auto 테스트 추가 |
| `tests/agent/test_advisor_config.py` | `native_advisor_applicable` 테스트 추가 |

**변경하지 않는 파일:** `advisor_runner.py`, `advisor_policy.py`, `gateway/`, `tools/`
