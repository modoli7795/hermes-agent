# External Advisor Routing Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a first practical advisory system to Hermes so the main agent can escalate difficult reasoning-heavy subtasks to stronger external advisor models/providers only when needed, with three advisory modes: single optimized advisor, multiple independent advisors, and multiple debating advisors.

**Architecture:** Build Phase 1 on top of the existing `delegate_task` and provider/credential-pool infrastructure instead of waiting for Anthropic native advisor support. Introduce an explicit advisor policy layer that decides when to escalate, routes to configured advisor providers/models, and returns compact advisory summaries to the main agent. Keep this phase provider-agnostic so Claude, Codex, and Gemini can all participate as external advisors. Prefer subscription/OAuth-backed account registration wherever the provider supports it, and only fall back to API-key billing for providers that do not have a supported OAuth/subscription path in Hermes. Support both explicit user-invoked advisory mode selection and agent-autonomous advisor escalation.

**Tech Stack:** Python, existing Hermes `AIAgent`, `delegate_task`, runtime provider resolution, credential pools, YAML config, pytest.

---

## Scope of this plan

This plan covers only Phase 1:
- external advisor routing
- provider/model/account policy
- safe escalation heuristics
- multi-advisor orchestration modes
- explicit and autonomous invocation policy
- config and tests

This plan does not yet implement:
- Anthropic native `advisor_20260301`
- `advisor_tool_result` transcript round-tripping
- beta header plumbing for the Anthropic advisor tool

Those belong to later phases after this routing layer works.

## Target advisory modes

Phase 1 should support these three runtime modes:

1. `single`
   - one optimized advisor is selected and asked once
   - cheapest and default path

2. `parallel`
   - multiple advisors are asked independently in parallel
   - parent agent receives separate answers plus a synthesized summary

3. `debate`
   - multiple advisors first answer independently
   - then one synthesis/debate pass compares disagreements and produces a final recommendation
   - most expensive, reserved for explicit request or high-stakes autonomous escalation

Phase 1 should also support two invocation sources:

1. `explicit`
   - the user directly requests advisor use or a specific mode

2. `autonomous`
   - Hermes decides to escalate based on policy and confidence heuristics

Suggested precedence:
- explicit user request always wins over automatic policy
- if no explicit request exists, policy may choose `single`, `parallel`, or `debate`
- default autonomous mode should start conservative: `single`

---

### Task 1: Add advisor config schema to persistent config

**Objective:** Create a dedicated config block for advisor behavior without overloading the existing `delegation` block.

**Files:**
- Modify: `hermes_cli/config.py`
- Test: `tests/hermes_cli/test_config_env_expansion.py`

**Step 1: Add default config section**

Add a new top-level `advisor` block near the existing `delegation` section in `DEFAULT_CONFIG`.

Use this shape:

```python
"advisor": {
    "enabled": False,
    "mode": "external",              # external | off
    "strategy": "on_demand",         # on_demand | always_verify | manual_only
    "invocation": "hybrid",          # explicit | autonomous | hybrid
    "call_mode": "single",           # single | parallel | debate
    "provider": "",                  # default provider for single mode
    "model": "",                     # default model for single mode
    "base_url": "",                  # optional direct endpoint
    "api_key": "",                   # optional direct API key
    "reasoning_effort": "",          # override for advisor child
    "max_iterations": 12,
    "max_advice_chars": 4000,
    "max_uses_per_turn": 1,
    "toolsets": ["terminal", "file"],
    "advisor_count": 2,
    "debate_rounds": 1,
    "providers": [
        {
            "provider": "",
            "model": "",
            "label": "",
            "base_url": "",
            "api_key": "",
            "weight": 1,
            "oauth_preferred": True,
        }
    ],
    "autonomous_modes": {
        "default": "single",
        "architecture": "parallel",
        "high_stakes": "debate",
    },
    "trigger_keywords": [
        "double-check",
        "second opinion",
        "verify",
        "hard bug",
        "architecture",
        "tradeoff",
    ],
}
```

**Step 2: Bump config version**

Increment `_config_version` in `hermes_cli/config.py` so existing installs receive the new defaults.

**Step 3: Add explanatory comments**

Document that:
- `delegation` is still general-purpose subagent routing
- `advisor` is a separate high-capability consultation path
- Phase 1 uses external delegation, not Anthropic native advisor
- `call_mode` controls one-advisor vs multi-advisor behavior
- `invocation` controls explicit-only, autonomous-only, or hybrid behavior

**Step 4: Add/adjust config tests**

Add or extend a config-loading test to verify the new `advisor` section exists with the expected defaults.

**Step 5: Run test**

Run: `source venv/bin/activate && python -m pytest tests/hermes_cli/test_config_env_expansion.py -q`
Expected: PASS

**Step 6: Commit**

```bash
git add hermes_cli/config.py tests/hermes_cli/test_config_env_expansion.py
git commit -m "feat: add advisor config defaults"
```

---

### Task 2: Add a reusable advisor config loader and runtime resolver

**Objective:** Create a single place that resolves advisor runtime settings, including single-advisor and multi-advisor topologies, using the same provider resolution model as delegation.

**Files:**
- Create: `agent/advisor_config.py`
- Test: `tests/agent/test_advisor_config.py`

**Step 1: Create config loader module**

Create `agent/advisor_config.py` with functions like:

```python
def load_advisor_config() -> dict: ...
def resolve_advisor_runtime(parent_agent=None) -> dict: ...
def resolve_advisor_runtimes(parent_agent=None) -> list[dict]: ...
def advisor_enabled(cfg: dict | None = None) -> bool: ...
```

**Step 2: Mirror delegation resolution rules**

`resolve_advisor_runtime()` / `resolve_advisor_runtimes()` should follow the same resolution order used by `tools/delegate_tool.py`:
- direct `advisor.base_url` + `advisor.api_key`
- configured `advisor.provider` resolved via `hermes_cli.runtime_provider.resolve_runtime_provider()`
- configured entries under `advisor.providers[]` resolved one-by-one for parallel/debate modes
- otherwise inherit nothing and return a disabled/empty runtime

**Step 3: Return a normalized runtime dict**

Use a consistent shape:

```python
{
    "enabled": True,
    "invocation": "hybrid",
    "call_mode": "single",
    "provider": "anthropic",
    "model": "claude-opus-4-6",
    "base_url": "https://api.anthropic.com",
    "api_key": "***",
    "api_mode": "anthropic_messages",
    "reasoning_effort": "high",
    "max_iterations": 12,
    "max_advice_chars": 4000,
    "max_uses_per_turn": 1,
    "toolsets": ["terminal", "file"],
    "strategy": "on_demand",
    "mode": "external",
    "oauth_preferred": True,
}
```

For parallel/debate modes, also return a normalized list of advisor runtimes with labels and ordering metadata.

**Step 4: Add focused unit tests**

Test cases:
- disabled config returns `enabled=False`
- direct `base_url` path resolves correctly
- provider-based path resolves through runtime provider
- multi-provider entries resolve into multiple normalized runtimes
- missing key/provider yields a clear error message

**Step 5: Run tests**

Run: `source venv/bin/activate && python -m pytest tests/agent/test_advisor_config.py -q`
Expected: PASS

**Step 6: Commit**

```bash
git add agent/advisor_config.py tests/agent/test_advisor_config.py
git commit -m "feat: add advisor runtime resolver"
```

---

### Task 3: Add advisor trigger heuristics and policy evaluation

**Objective:** Define when the main agent should ask for advice, whether the request was explicit or autonomous, and which of the three advisor modes should be used.

**Files:**
- Create: `agent/advisor_policy.py`
- Test: `tests/agent/test_advisor_policy.py`

**Step 1: Implement a small pure-policy module**

Create functions like:

```python
def should_request_advice(*, user_text: str, cfg: dict, turn_state: dict) -> tuple[bool, str, str]: ...
def detect_explicit_advisor_request(user_text: str, cfg: dict) -> dict: ...
def select_advisor_mode(*, user_text: str, cfg: dict, turn_state: dict, explicit_request: dict | None) -> tuple[str, str]: ...
def build_advisor_goal(*, user_text: str, context_summary: str, reason: str, mode: str) -> str: ...
```

**Step 2: Start with conservative heuristics**

Phase 1 heuristics should be explicit and easy to reason about:
- user directly asks for verification, second opinion, architecture review, tradeoff analysis
- user directly asks for one advisor, multiple advisors, consensus, debate, or discussion
- task text includes configured trigger keywords
- task is marked as difficult by the caller
- architecture/tradeoff/high-stakes tasks may autonomously choose `parallel` or `debate`
- do not exceed `max_uses_per_turn`

Do not add opaque scoring yet.

**Step 3: Keep results explainable**

Return both boolean and reason, for example:
- `("requested-by-user")`
- `("keyword:architecture")`
- `("limit-reached")`
- `("advisor-disabled")`

Include the chosen mode in the result, for example:
- `(True, "requested-by-user", "parallel")`
- `(True, "autonomous:high-stakes", "debate")`
- `(False, "limit-reached", "single")`

**Step 4: Add policy tests**

Test:
- explicit verification request triggers advice
- explicit request for multiple advisors selects `parallel`
- explicit request for advisor debate selects `debate`
- neutral task does not trigger advice
- per-turn limit blocks repeated advice
- custom trigger keywords work

**Step 5: Run tests**

Run: `source venv/bin/activate && python -m pytest tests/agent/test_advisor_policy.py -q`
Expected: PASS

**Step 6: Commit**

```bash
git add agent/advisor_policy.py tests/agent/test_advisor_policy.py
git commit -m "feat: add advisor trigger policy"
```

---

### Task 4: Add external advisor execution helper on top of delegate_task semantics

**Objective:** Create execution helpers for all three advisory modes: single, parallel, and debate.

**Files:**
- Create: `agent/advisor_runner.py`
- Modify: `tools/delegate_tool.py` only if a small shared helper extraction is needed
- Test: `tests/agent/test_advisor_runner.py`

**Step 1: Build a dedicated runner instead of duplicating all delegation code inline**

Create functions like:

```python
def request_external_advice(
    *,
    parent_agent,
    goal: str,
    context: str,
    runtime: dict,
) -> dict:
    ...

def request_parallel_advice(
    *,
    parent_agent,
    goal: str,
    context: str,
    runtimes: list[dict],
) -> dict:
    ...

def request_debate_advice(
    *,
    parent_agent,
    goal: str,
    context: str,
    runtimes: list[dict],
) -> dict:
    ...
```

Return shape:

```python
{
    "status": "completed",
    "call_mode": "single",
    "advisor_provider": "anthropic",
    "advisor_model": "claude-opus-4-6",
    "summary": "...",
    "full_text": "...",
    "duration_seconds": 4.2,
    "reason": "requested-by-user",
}
```

For `parallel` mode, return:

```python
{
    "status": "completed",
    "call_mode": "parallel",
    "results": [
        {"label": "claude", "summary": "..."},
        {"label": "codex", "summary": "..."},
    ],
    "summary": "Cross-advisor synthesis...",
}
```

For `debate` mode, return:

```python
{
    "status": "completed",
    "call_mode": "debate",
    "initial_results": [...],
    "debate_summary": "...",
    "summary": "Final debated recommendation...",
}
```

**Step 2: Reuse existing child-agent building patterns**

Use the same concepts already present in `tools/delegate_tool.py`:
- child inherits safe tool constraints
- advisor runtime can override provider/model/base_url/api_key/api_mode
- child gets fresh iteration budget
- child skips memory and context files

Prefer extracting a small shared helper from `tools/delegate_tool.py` only if it avoids copy-paste without causing a large refactor.

For `parallel`, use concurrent independent child agents.

For `debate`, use this minimal Phase 1 protocol:
- ask each advisor independently for an initial answer
- build a compact disagreement summary
- run one final synthesis pass, preferably on the highest-priority advisor runtime
- do not create recursive free-form debates yet; keep it to one compare-and-synthesize round

**Step 3: Limit advisor verbosity**

The advisor prompt should request:
- concise recommendation
- uncertainty notes
- next-step suggestions
- capped output length

Example prompt contract:

```text
You are an advisor model. Do not execute tools unless necessary.
Return:
1. Best recommendation
2. Key risks or uncertainty
3. Concrete next step
Keep the response under 250 words.
```

For parallel/debate, require each advisor to keep its own answer short so the synthesis step stays cheap.

**Step 4: Trim final output for parent injection**

If full advice is too long, trim to `max_advice_chars` while preserving the first recommendation block.

**Step 5: Add tests**

Mock the child agent execution path and verify:
- runtime overrides are passed through
- summary trimming works
- failure returns structured error
- parallel mode launches multiple independent advisor calls
- debate mode performs initial passes then one synthesis pass

**Step 6: Run tests**

Run: `source venv/bin/activate && python -m pytest tests/agent/test_advisor_runner.py -q`
Expected: PASS

**Step 7: Commit**

```bash
git add agent/advisor_runner.py tools/delegate_tool.py tests/agent/test_advisor_runner.py
git commit -m "feat: add external advisor runner"
```

---

### Task 5: Wire advisor support into `AIAgent` turn execution

**Objective:** Let the main agent actually consult the advisor when policy says it should, with support for explicit and autonomous invocation plus all three advisory modes.

**Files:**
- Modify: `run_agent.py`
- Modify: `agent/prompt_builder.py` only if the system prompt needs a short advisor hint
- Test: `tests/run_agent/test_external_advisor.py`

**Step 1: Add a minimal advisor turn-state counter**

Track per-turn advisor usage, for example:

```python
advisor_state = {
    "uses_this_turn": 0,
    "last_mode": None,
    "explicit_request": None,
}
```

Reset this at the start of each top-level user turn, not every tool call.

**Step 2: Evaluate advisor policy before high-cost reasoning branches**

Initial integration target:
- before handling a difficult delegated reasoning task
- before final answer generation when user explicitly requested verification
- before final answer generation when the user explicitly requested `single`, `parallel`, or `debate`

Do not over-integrate yet. Keep Phase 1 narrow and deterministic.

**Step 3: Inject advisor result into the model conversation in a controlled way**

When advice is obtained, append a compact assistant-visible note such as:

```text
[Advisor summary mode=parallel]
Advisor claude-opus-4-6: ...
Advisor codex: ...
Synthesized recommendation: ...
```

Keep this as plain text context, not a fake tool transcript.

**Step 4: Fail open**

If advisor routing fails:
- log it
- do not abort the main turn
- continue with the parent model

**Step 5: Add tests**

Test:
- explicit verification request triggers advisor call
- explicit `parallel` request triggers parallel mode
- explicit `debate` request triggers debate mode
- autonomous path defaults to `single` unless policy upgrades mode
- advisor result is injected into downstream context
- advisor failure does not crash response generation
- per-turn usage cap is respected

**Step 6: Run tests**

Run: `source venv/bin/activate && python -m pytest tests/run_agent/test_external_advisor.py -q`
Expected: PASS

**Step 7: Commit**

```bash
git add run_agent.py agent/prompt_builder.py tests/run_agent/test_external_advisor.py
git commit -m "feat: wire external advisor into agent loop"
```

---

### Task 6: Add credential-pool guidance and provider-specific UX

**Objective:** Make it obvious how users register Anthropic, Gemini, and Codex credentials for advisor routing, with OAuth/subscription-backed registration preferred whenever available.

**Files:**
- Modify: `hermes_cli/auth_commands.py`
- Modify: `hermes_cli/tips.py`
- Test: `tests/hermes_cli/test_auth_commands.py`

**Step 1: Improve provider messaging**

Add help text or printed hints so users understand the recommended registration commands:
- `hermes auth add anthropic`
- `hermes auth add openai-codex`
- `hermes auth add gemini --auth-type api_key`
- `hermes auth list`

Make the guidance explicit:
- Anthropic and Codex should be presented as preferred OAuth/subscription advisor accounts
- Gemini should be clearly marked as API-key-only in the current Hermes implementation

**Step 2: Add advisor-oriented tips**

Examples:
- “Use pooled Anthropic credentials for Opus advisor rotation.”
- “Gemini and Codex can be external advisors, but not Anthropic native advisor models.”

**Step 3: Add tests for messages**

Keep this small: verify no regressions in auth command output and provider handling.

**Step 4: Run tests**

Run: `source venv/bin/activate && python -m pytest tests/hermes_cli/test_auth_commands.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add hermes_cli/auth_commands.py hermes_cli/tips.py tests/hermes_cli/test_auth_commands.py
git commit -m "docs: improve advisor auth guidance"
```

---

### Task 7: Add end-to-end config + integration tests

**Objective:** Prove the feature works across config loading, runtime resolution, and advisory invocation.

**Files:**
- Create: `tests/integration/test_advisor_routing.py`

**Step 1: Create integration scenarios**

Cover:
- advisor disabled
- advisor enabled with anthropic provider
- advisor enabled with codex provider
- advisor enabled with direct base_url
- advisor enabled with explicit `parallel` mode
- advisor enabled with explicit `debate` mode
- autonomous upgrade from `single` to `parallel`/`debate`
- missing credentials fail open

**Step 2: Verify provider isolation**

Ensure advisor runtime overrides do not mutate the parent runtime.

**Step 3: Verify credential-pool compatibility**

Mock pooled credentials and verify the advisor runtime can select them cleanly.

**Step 4: Run tests**

Run: `source venv/bin/activate && python -m pytest tests/integration/test_advisor_routing.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/integration/test_advisor_routing.py
git commit -m "test: cover external advisor routing flow"
```

---

### Task 8: Run targeted suite, then full suite

**Objective:** Verify the new advisor layer does not break the rest of Hermes.

**Files:**
- No code changes expected

**Step 1: Run targeted tests**

Run:

```bash
source venv/bin/activate && python -m pytest \
  tests/agent/test_advisor_config.py \
  tests/agent/test_advisor_policy.py \
  tests/agent/test_advisor_runner.py \
  tests/run_agent/test_external_advisor.py \
  tests/integration/test_advisor_routing.py \
  tests/hermes_cli/test_auth_commands.py \
  tests/hermes_cli/test_config_env_expansion.py -q
```

Expected: PASS

**Step 2: Run full suite**

Run:

```bash
source venv/bin/activate && python -m pytest tests/ -q
```

Expected: PASS

**Step 3: Commit final polish if needed**

```bash
git add -A
git commit -m "feat: add external advisor routing"
```

---

## Verification checklist

- [ ] `advisor` config block loads correctly
- [ ] advisor runtime can resolve Anthropic, Gemini, and Codex credentials
- [ ] advisor routing is optional and disabled by default
- [ ] explicit user verification requests can trigger advisor consultation
- [ ] explicit mode requests can select `single`, `parallel`, or `debate`
- [ ] autonomous policy can conservatively choose `single` and selectively upgrade to `parallel`/`debate`
- [ ] failures in advisor routing do not break the main answer path
- [ ] credential pools remain usable for advisor providers
- [ ] all targeted tests pass
- [ ] full test suite passes

---

## Phase 2 and Phase 3 follow-up

After this plan is implemented:

Phase 2:
- add provider/account selection policy
- smarter escalation heuristics
- advisor cost accounting and telemetry
- optional slash command or config UX for advisor inspection

Phase 3:
- add Anthropic native advisor support
- `advisor_20260301` tool definition
- beta header support
- `advisor_tool_result` history preservation rules
- executor/advisor pair validation

---

Plan complete and saved. Ready to execute using subagent-driven-development task-by-task, or I can start implementing Task 1 immediately in this session.