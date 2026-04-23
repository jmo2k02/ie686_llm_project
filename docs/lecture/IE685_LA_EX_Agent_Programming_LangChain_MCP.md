# IE685 Exercise: Agent Programming with ReAct, LangChain, and MCP

In the first part, we build and stress-test our own tools to understand what makes ReAct agents strong.
In the final part we connect an external MCP server (Google Keep).

## What this lab emphasizes

1. **Tool engineering quality** (descriptions, signatures, safe side effects).
2. **ReAct control loop behavior** (thought-action-observation cycles).
3. **Practical agent capabilities**: multi-step tool use, streaming traces, and memory.
4. **MCP interoperability**: plugging in third-party tools at the end.

## Working References (Part 1: ReAct + Tool Design)

- Agents overview: [https://docs.langchain.com/oss/python/langchain-agents](https://docs.langchain.com/oss/python/langchain-agents)
- `create_agent` API docs: [https://docs.langchain.com/oss/python/langchain/agents](https://docs.langchain.com/oss/python/langchain/agents)
- Tools docs: [https://docs.langchain.com/oss/python/langchain/tools](https://docs.langchain.com/oss/python/langchain/tools)
- Streaming docs: [https://docs.langchain.com/oss/python/langchain/streaming](https://docs.langchain.com/oss/python/langchain/streaming)

We add MCP-specific references later when we transition to the capstone.

## How To Use This Notebook

- Run cells top-to-bottom the first time.
- For each `Try It Out` block, change the prompt and inspect the trace.
- Keep an eye on **which tools were called**, not only the final answer.
- Treat this as an experimental lab: compare prompts and observe behavior changes.

## 1) Setup and Cross-Platform Preflight

This notebook is designed to run on **Windows and macOS**.

Requirements:
- `OPENAI_API_KEY`
- Keep MCP authentication variables (only needed in the final MCP section)

```python
%pip install -qU langchain==1.2.10 langgraph==1.0.9 langchain-openai==1.1.10 langchain-mcp-adapters mcp python-dotenv pydantic gpsoauth keep-mcp

# If keep-mcp is not installed yet, uncomment one of these:
# %pip install -qU keep-mcp
# %pip install -qU git+https://github.com/feuerdev/keep-mcp.git
```

```python
import os
import sys
import json
import random
import shutil
import platform
import subprocess
import importlib.util

from pathlib import Path
from datetime import datetime, date
from typing import Any

from dotenv import load_dotenv

load_dotenv()

print('Python executable:', sys.executable)
print('OS:', platform.system(), platform.release())
print('Working directory:', Path.cwd())
```

```python
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


def create_chat_model(model_name="gpt-5-mini") -> BaseChatModel:
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY is missing. Add it to environment or .env file.')

    return ChatOpenAI(model=model_name, temperature=0)


chat_model = create_chat_model()
print('Model backend:', chat_model.__class__.__name__)
```

```python
# Connectivity smoke test
smoke = chat_model.invoke('Reply with exactly six words: ReAct workshop is ready.')
print(smoke.content)
```

## 2) Quick ReAct Recap

A ReAct agent loops until completion:
1. Read current context.
2. Decide whether to call a tool.
3. Execute tool call.
4. Integrate observation.
5. Repeat or finalize.

The quality of this loop strongly depends on:
- tool descriptions
- argument schema clarity
- meaningful tool return values

### ReAct + Agent Loop Visuals

**ReAct reasoning/action loop**

![ReAct diagram](https://react-lm.github.io/files/diagram.png)

Source: ReAct project page ([https://react-lm.github.io/](https://react-lm.github.io/))

**General intelligent-agent perception/action loop**

![Agent loop](https://upload.wikimedia.org/wikipedia/commons/a/af/Artificial_Intelligent_Agent.png)

Source: Wikimedia Commons (CC0): [https://commons.wikimedia.org/wiki/File:Artificial_Intelligent_Agent.png](https://commons.wikimedia.org/wiki/File:Artificial_Intelligent_Agent.png)

## 3) Build a Rich Local Toolset 

These tools are intentionally varied:
- numeric helpers
- scheduling helpers
- content-analysis helpers
- stateful local storage helpers
- playful campus-planning helpers

This gives the agent interesting choices and multi-step trajectories.

```python
from langchain_core.tools import tool

TASKS_FILE = Path('outputs') / 'react_agent_tasks.json'
TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_tasks() -> list[dict[str, Any]]:
    if not TASKS_FILE.exists():
        return []
    try:
        return json.loads(TASKS_FILE.read_text(encoding='utf-8'))
    except Exception:
        return []


def _save_tasks(tasks: list[dict[str, Any]]) -> None:
    TASKS_FILE.write_text(json.dumps(tasks, indent=2), encoding='utf-8')


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def get_current_date() -> str:
    """Return today's date in YYYY-MM-DD format."""
    return date.today().isoformat()


@tool
def days_until(target_date: str) -> int:
    """Return number of days from today to target_date (YYYY-MM-DD)."""
    t = date.fromisoformat(target_date)
    return (t - date.today()).days


@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9.0 / 5.0) + 32.0


@tool
def estimate_reading_time_minutes(text: str, words_per_minute: int = 220) -> int:
    """Estimate reading time in minutes using word count and words-per-minute."""
    words = len([w for w in text.split() if w.strip()])
    return max(1, round(words / max(80, words_per_minute)))


@tool
def text_statistics(text: str) -> dict:
    """Return character count, word count, and estimated sentence count."""
    characters = len(text)
    words = len([w for w in text.split() if w.strip()])
    sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
    return {
        'characters': characters,
        'words': words,
        'sentences_estimate': sentences,
    }


@tool
def campus_weather_stub(day: str) -> str:
    """Return a playful deterministic weather stub for campus planning by day name."""
    mapping = {
        'monday': 'Cloudy, 11C, light wind',
        'tuesday': 'Sunny, 16C, clear sky',
        'wednesday': 'Rainy, 9C, umbrella recommended',
        'thursday': 'Partly cloudy, 14C',
        'friday': 'Sunny, 18C, nice for outdoor breaks',
        'saturday': 'Warm, 21C',
        'sunday': 'Cool, 12C',
    }
    return mapping.get(day.strip().lower(), 'Unknown day. Try Monday-Sunday.')


@tool
def cafeteria_menu_stub(day: str) -> str:
    """Return a sample cafeteria menu by day name."""
    menus = {
        'monday': 'Pasta arrabbiata, salad bowl, lentil soup',
        'tuesday': 'Veggie curry, rice, yogurt bowl',
        'wednesday': 'Chicken wrap, roasted veggies, fruit cup',
        'thursday': 'Falafel plate, hummus, tabbouleh',
        'friday': 'Fish and potatoes, mixed salad, apple pie',
    }
    return menus.get(day.strip().lower(), 'Weekend menu: brunch options and sandwiches.')


@tool
def travel_time_stub(origin: str, destination: str, mode: str = 'walk') -> int:
    """Estimate travel time in minutes between sample campus locations."""
    edges = {
        ('library', 'cafeteria'): 8,
        ('library', 'main_hall'): 5,
        ('main_hall', 'cafeteria'): 6,
        ('dorm', 'library'): 12,
        ('dorm', 'main_hall'): 10,
    }
    o = origin.strip().lower().replace(' ', '_')
    d = destination.strip().lower().replace(' ', '_')
    base = edges.get((o, d), edges.get((d, o), 15))
    mode = mode.strip().lower()
    if mode == 'bike':
        return max(2, round(base * 0.45))
    if mode == 'bus':
        return max(4, round(base * 0.65 + 3))
    return base


@tool
def save_local_task(title: str, due_date: str, priority: str = 'medium') -> str:
    """Persist a local task to outputs/react_agent_tasks.json and return a confirmation string."""
    tasks = _load_tasks()
    task = {
        'id': len(tasks) + 1,
        'title': title,
        'due_date': due_date,
        'priority': priority.lower(),
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    tasks.append(task)
    _save_tasks(tasks)
    return f"Saved task #{task['id']} to {TASKS_FILE.resolve()}"


@tool
def list_local_tasks() -> list[dict]:
    """Return all saved local tasks from outputs/react_agent_tasks.json."""
    return _load_tasks()


@tool
def clear_local_tasks(confirm: bool = False) -> str:
    """Clear all local tasks only when confirm=True."""
    if not confirm:
        return 'Refused: set confirm=True to clear tasks.'
    _save_tasks([])
    return 'All local tasks cleared.'


local_tools = [
    add_numbers,
    multiply_numbers,
    get_current_date,
    days_until,
    celsius_to_fahrenheit,
    estimate_reading_time_minutes,
    text_statistics,
    campus_weather_stub,
    cafeteria_menu_stub,
    travel_time_stub,
    save_local_task,
    list_local_tasks,
    clear_local_tasks,
]

print('Local tools loaded:', len(local_tools))
print([t.name for t in local_tools])
```

```python
# See the tool schemas exactly as the model sees them
for t in local_tools:
    print('=' * 90)
    print('Tool:', t.name)
    print('Description:', (t.description or '').strip()[:260])
    print('Args schema:', getattr(t, 'args', None))
```

## 4) Build ReAct Agent Helpers

We define reusable helpers:
- `build_agent`
- `run_agent`
- `print_tool_trace`
- `stream_agent_updates`

These make it easy to run many experiments quickly.

```python
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents import create_agent


def build_agent(
    model: BaseChatModel,
    tools: list[Any],
    checkpointer: Any = None,
    middleware: list[Any] | None = None,
):
    prompt = (
        'You are a careful ReAct-style assistant. '
        'Use tools when they improve correctness. '
        'For side-effect actions, explain briefly what you will do. '
        'Never clear tasks unless the user explicitly requests it.'
    )
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt,
        checkpointer=checkpointer,
        middleware=middleware or [],
    )


def _extract_messages(result: Any) -> list[Any]:
    if isinstance(result, dict) and 'messages' in result:
        return result['messages']
    if isinstance(result, list):
        return result
    return [result]


def run_agent(agent: Any, query: str, config: dict | None = None) -> dict[str, Any]:
    try:
        if config is None:
            result = agent.invoke({'messages': [{'role': 'user', 'content': query}]})
        else:
            result = agent.invoke({'messages': [{'role': 'user', 'content': query}]}, config=config)
    except NotImplementedError as exc:
        raise RuntimeError(
            'This agent is using async-only tools. Use: await run_agent_async(agent, query, config=...)'
        ) from exc

    messages = _extract_messages(result)
    final_text = ''
    for msg in reversed(messages):
        content = getattr(msg, 'content', '')
        if isinstance(content, str) and content.strip():
            final_text = content
            break
    return {'raw': result, 'messages': messages, 'final': final_text}


async def run_agent_async(agent: Any, query: str, config: dict | None = None) -> dict[str, Any]:
    if config is None:
        result = await agent.ainvoke({'messages': [{'role': 'user', 'content': query}]})
    else:
        result = await agent.ainvoke({'messages': [{'role': 'user', 'content': query}]}, config=config)

    messages = _extract_messages(result)
    final_text = ''
    for msg in reversed(messages):
        content = getattr(msg, 'content', '')
        if isinstance(content, str) and content.strip():
            final_text = content
            break
    return {'raw': result, 'messages': messages, 'final': final_text}


def print_tool_trace(messages: list[Any]) -> None:
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            print(f'[{i}] AIMessage: {str(msg.content)[:180]}')
            tool_calls = getattr(msg, 'tool_calls', None) or []
            for tc in tool_calls:
                print(f"    tool_call -> {tc.get('name')} args={tc.get('args')}")
        elif isinstance(msg, ToolMessage):
            print(f"[{i}] ToolMessage ({getattr(msg, 'name', 'tool')}): {str(msg.content)[:260]}")
        else:
            role = type(msg).__name__
            content = getattr(msg, 'content', msg)
            print(f'[{i}] {role}: {str(content)[:180]}')


def stream_agent_updates(agent: Any, query: str, config: dict | None = None) -> None:
    kwargs = {'stream_mode': 'updates'}
    if config is None:
        stream = agent.stream({'messages': [{'role': 'user', 'content': query}]}, **kwargs)
    else:
        stream = agent.stream({'messages': [{'role': 'user', 'content': query}]}, config=config, **kwargs)

    for event in stream:
        print('--- update ---')
        if isinstance(event, dict):
            for key, value in event.items():
                print('node:', key)
                print('payload keys:', list(value.keys()) if isinstance(value, dict) else type(value).__name__)
        else:
            print(event)
```

```python
react_agent = build_agent(chat_model, local_tools)
print('ReAct agent ready.')
```

## 5) Core ReAct Demos with Local Tools

In this section, we move from isolated tool calls to realistic agent workflows.

Each demo is designed to highlight a different part of the ReAct loop:
- composing multiple tool calls in sequence,
- handling side effects (writing state),
- validating outcomes with follow-up retrieval.

As you run the cells, focus on both:
1. the final answer quality, and
2. the tool-call trace that led to it.

```python
# Demo A: mixed numeric + date reasoning
q1 = (
    'Compute 18.5 * 4, then add 12.75. '
    'Also tell me the number of days until 2026-12-24 and return both results clearly.'
)
r1 = run_agent(react_agent, q1)

print('Final answer:')
print(r1['final'])
print()
print('Trace:')
print_tool_trace(r1['messages'])
```

```python
# Demo B: campus planning + side effects
q2 = (
    'Plan a focused study afternoon for Tuesday. '
    'Use weather, cafeteria, and travel-time tools (library to cafeteria by walk). '
    'Then save two concrete local tasks with due date 2026-03-10.'
)
r2 = run_agent(react_agent, q2)

print('Final answer:')
print(r2['final'])
print()
print('Trace:')
print_tool_trace(r2['messages'])
```

```python
# Verify persisted side effects
q3 = 'List all local tasks and summarize priority distribution.'
r3 = run_agent(react_agent, q3)

print('Final answer:')
print(r3['final'])
print()
print('Trace:')
print_tool_trace(r3['messages'])
```

### Try It Out 1

Create a prompt that forces the agent to use at least **four** local tools.

Suggestion:
- mix date math, text analysis, planning, and side effects.

```python
# TODO: Replace with your own prompt
try_prompt_1 = (
    'Analyze this text: "Agent engineering requires structured experimentation and good tooling." '
    'Estimate reading time, compute 9.5*8, get today date, and save one local task that references all results.'
)

try_run_1 = run_agent(react_agent, try_prompt_1)
print('Final answer:')
print(try_run_1['final'])
print()
print('Trace:')
print_tool_trace(try_run_1['messages'])
```

## 6) Fun ReAct Extensions 

Based on current LangChain ecosystem patterns, useful showcase directions include:
- streaming internal updates
- stateful memory across turns
- guardrails for risky tool actions
- structured outputs and middleware (v1 `create_agent` path)

In this notebook, we implement two directly with ReAct: **streaming** and **memory**.

```python
# Extension A: stream node updates during execution
stream_query = (
    'Check Tuesday weather, estimate travel time from dorm to library by bike, '
    'and suggest one task to save with due date 2026-03-11.'
)

stream_agent_updates(react_agent, stream_query)
```

```python
# Extension B: short-term memory across turns using checkpointer
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
memory_agent = build_agent(chat_model, local_tools, checkpointer=memory)
thread_cfg = {'configurable': {'thread_id': 'ie685-demo-thread'}}

m1 = run_agent(memory_agent, 'Create two tasks: revise ReAct notes and test MCP setup for 2026-03-12.', config=thread_cfg)
print('Turn 1 final:')
print(m1['final'])

m2 = run_agent(memory_agent, 'What are the two tasks I just asked you to create? Answer briefly.', config=thread_cfg)
print()
print('Turn 2 final:')
print(m2['final'])
```

### Try It Out 2

Use the memory-enabled agent and run a 3-turn interaction:
1. Ask it to create tasks.
2. Ask for a filtered summary.
3. Ask it to add one more task based on what is missing.

```python
# TODO: iterate with your own three-turn memory experiment
m_try_1 = run_agent(memory_agent, 'Create one task about benchmarking agent tool calls for 2026-03-15.', config=thread_cfg)
m_try_2 = run_agent(memory_agent, 'Now list only high priority tasks.', config=thread_cfg)

print('Turn A final:')
print(m_try_1['final'])
print()
print('Turn B final:')
print(m_try_2['final'])
```

## 7) Safety with LangChain Human-in-the-Loop Middleware

LangChain already ships a built-in pattern for tool approval:
- `HumanInTheLoopMiddleware`

In this setup:
- normal tools execute automatically,
- selected sensitive tools (here: `clear_local_tasks`) trigger an interrupt,
- you approve/reject/edit the action before execution resumes.

This is cleaner than implementing a custom manual approval loop.

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        'clear_local_tasks': {
            'allowed_decisions': ['approve', 'reject', 'edit']
        },
        # Example: explicitly allow this one without interruption
        'save_local_task': False,
    },
    description_prefix='Tool execution pending approval',
)

hitl_agent = build_agent(
    chat_model,
    local_tools,
    checkpointer=InMemorySaver(),
    middleware=[hitl_middleware],
)

hitl_config = {'configurable': {'thread_id': 'ie685-hitl-demo'}}


def resume_with_console_review(
    agent: Any,
    result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Resume interrupted HITL runs by collecting console decisions.

    Returns:
        {
            'result': <final invoke result>,
            'decision_log': [ ... ]
        }
    """
    decision_log: list[dict[str, Any]] = []

    while '__interrupt__' in result and result['__interrupt__']:
        interrupt_obj = result['__interrupt__'][0]
        payload = getattr(interrupt_obj, 'value', {}) or {}

        action_requests = payload.get('action_requests', [])
        review_configs = payload.get('review_configs', [])

        if not action_requests:
            print('Interrupt received but no action_requests found.')
            break

        print()
        print('Agent paused for review. Pending tool action(s):')
        decisions = []

        for i, req in enumerate(action_requests):
            tool_name = req.get('name')
            args = req.get('arguments', {})

            allowed = ['approve', 'reject', 'edit']
            if i < len(review_configs):
                allowed = review_configs[i].get('allowed_decisions', allowed)

            print(f"- Tool: {tool_name}")
            print(f"  Args: {args}")
            print(f"  Allowed decisions: {allowed}")

            decision = input('Choose decision (approve/reject/edit): ').strip().lower()
            if decision not in allowed:
                decision = allowed[0]
                print(f"  -> invalid input, defaulting to '{decision}'")

            log_entry: dict[str, Any] = {
                'tool': tool_name,
                'requested_args': args,
                'decision': decision,
            }

            if decision == 'approve':
                decisions.append({'type': 'approve'})
            elif decision == 'reject':
                reason = input('Optional rejection reason: ').strip() or 'Rejected by user.'
                decisions.append({'type': 'reject', 'message': reason})
                log_entry['message'] = reason
            elif decision == 'edit':
                print('Provide edited args as JSON (or press Enter to keep original args).')
                edited_raw = input('edited args JSON: ').strip()
                if edited_raw:
                    try:
                        edited_args = json.loads(edited_raw)
                    except json.JSONDecodeError:
                        edited_args = args
                        print('  -> invalid JSON, using original args')
                else:
                    edited_args = args
                decisions.append({
                    'type': 'edit',
                    'edited_action': {'name': tool_name, 'args': edited_args},
                })
                log_entry['edited_args'] = edited_args

            decision_log.append(log_entry)

        result = agent.invoke(Command(resume={'decisions': decisions}), config=config)

    return {'result': result, 'decision_log': decision_log}
```

```python
default_safety_query = (
    'List all local tasks, then clear all local tasks, then list tasks again and explain what happened.'
)

print('Enter a SAFETY SCENARIO query.')
print('Do not type approve/reject/edit here; those come in the next prompt.')
raw_query = input('Safety scenario (press Enter for default): ').strip()

decision_words = {'approve', 'reject', 'edit', 'yes', 'no', 'y', 'n'}
if raw_query.lower() in decision_words:
    print("Detected decision keyword as query. Using default safety scenario instead.")
    safety_query = default_safety_query
else:
    safety_query = raw_query or default_safety_query

print()
print('Running HITL safety demo...')
print('Query:', safety_query)

tasks_before = _load_tasks()

first_result = hitl_agent.invoke(
    {'messages': [{'role': 'user', 'content': safety_query}]},
    config=hitl_config,
)

if '__interrupt__' in first_result and first_result['__interrupt__']:
    review_out = resume_with_console_review(hitl_agent, first_result, hitl_config)
    final_result = review_out['result']
    decision_log = review_out['decision_log']
else:
    final_result = first_result
    decision_log = []
    print()
    print('No interrupt was triggered. Try a more explicit destructive request if you want approval flow.')

tasks_after = _load_tasks()
messages = final_result.get('messages', [])

print()
print('Decision log:')
if decision_log:
    for i, entry in enumerate(decision_log, start=1):
        print(f"{i}. {entry}")
else:
    print('(No approval decisions were collected)')

print()
print('Deterministic state check:')
print(f"tasks_before={len(tasks_before)}, tasks_after={len(tasks_after)}")

print()
print('Final answer:')
if messages:
    print(getattr(messages[-1], 'content', ''))
else:
    print('(no final message)')

print()
print('Trace:')
print_tool_trace(messages)
```

## 8) MCP 

At this point, we already have a strong local ReAct agent.
Now we switch the source of tools: from self-written tools to MCP-provided tools.

### MCP references 
- MCP architecture: [https://modelcontextprotocol.io/docs/learn/architecture](https://modelcontextprotocol.io/docs/learn/architecture)
- MCP specification: [https://modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification)
- Anthropic MCP docs: [https://docs.anthropic.com/en/docs/mcp](https://docs.anthropic.com/en/docs/mcp)
- Keep MCP repository: [https://github.com/feuerdev/keep-mcp](https://github.com/feuerdev/keep-mcp)

### MCP visual

![MCP component diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Model_Context_Protocol_Component_diagram.svg/500px-Model_Context_Protocol_Component_diagram.svg.png)

Source: Wikimedia Commons (MIT): [https://commons.wikimedia.org/wiki/File:Model_Context_Protocol_Component_diagram.svg](https://commons.wikimedia.org/wiki/File:Model_Context_Protocol_Component_diagram.svg)

## 9) Keep MCP Setup (Capstone)

Design constraints for this setup:
- Windows + macOS compatibility
- stdio transport
- command + args list (no shell command strings)

### Step A: Install `keep-mcp`

Pick one option:

```bash
# Option 1 (recommended for quick run)
pip install keep-mcp

# Option 2 (isolated runtime, no global install)
uvx keep-mcp --help
```

If the package is not available in your environment, install directly from GitHub:

```bash
pip install git+https://github.com/feuerdev/keep-mcp.git
```

### Step B: Get Google credentials for keep-mcp

`keep-mcp` expects these env vars:
- `GOOGLE_EMAIL`
- `GOOGLE_MASTER_TOKEN`

#### How to get a master token (quick practical flow)

1. Open [https://accounts.google.com/EmbeddedSetup](https://accounts.google.com/EmbeddedSetup) and sign in.
2. Open browser developer tools and inspect cookies for `accounts.google.com`.
3. Copy the `oauth_token` cookie value.
4. Convert `oauth_token` to a master token with `gpsoauth`:

### If you see `Failed to parse JSONRPC message from server`

This usually means the launcher printed non-JSON text to `stdout` (for example `NOTE: running app ...`), which breaks MCP stdio parsing.

Use an explicit app invocation that keeps stdout clean:

```bash
uvx --from keep-mcp mcp --help
```

If that works, the notebook should auto-select the same pattern.

```python
import gpsoauth, secrets
email = input('Google email: ').strip()
oauth_token = input('oauth_token: ').strip()
android_id = secrets.token_hex(8)
res = gpsoauth.exchange_token(email, oauth_token, android_id)
print('android_id =', android_id)
print('GOOGLE_MASTER_TOKEN =', res['Token'])
```

Then set in `.env`:

```env
GOOGLE_EMAIL="your_google_account@example.com"
GOOGLE_MASTER_TOKEN="your_master_token"
KEEP_MCP_AUTH_READY=1
```

### Security note

Treat `GOOGLE_MASTER_TOKEN` like a password:
- never commit it,
- do not share it in notebooks/slides,
- rotate/change account credentials if exposed.

### Sources

- keep-mcp README: [https://github.com/feuerdev/keep-mcp](https://github.com/feuerdev/keep-mcp)
- keep-mcp config (`GOOGLE_EMAIL` / `GOOGLE_MASTER_TOKEN`): [README env section](https://github.com/feuerdev/keep-mcp#configuration)
- gkeepapi auth docs (`gpsoauth`, token flow): [https://gkeepapi.readthedocs.io/en/latest/](https://gkeepapi.readthedocs.io/en/latest/)
- oauth token from EmbeddedSetup + cookies: [gpsoauth-java README](https://github.com/oddgny/gpsoauth-java/blob/main/README.md#receiving-an-authentication-token)

```python
# Keep auth preflight
required_keep_env = ['GOOGLE_EMAIL', 'GOOGLE_MASTER_TOKEN']
optional_keep_env = [
    'KEEP_MCP_AUTH_READY',
    'KEEP_EMAIL',
    'KEEP_PASSWORD',
    'KEEP_TOKEN',
    'GOOGLE_KEEP_EMAIL',
    'GOOGLE_KEEP_PASSWORD',
    'GOOGLE_PASSWORD',
]

present_required = [k for k in required_keep_env if os.getenv(k)]
missing_required = [k for k in required_keep_env if not os.getenv(k)]
present_optional = [k for k in optional_keep_env if os.getenv(k)]

print('Required Keep env vars present:', present_required)
print('Optional Keep env hints present:', present_optional)

if missing_required:
    print()
    print('Missing required Keep env vars:', missing_required)
    print('Follow setup instructions in section 9 and re-run this cell.')
else:
    print()
    print('Keep auth preflight looks good.')
```

```python
def _safe_find_spec(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def resolve_keep_stdio_command() -> tuple[str, list[str], list[dict[str, Any]]]:
    candidates: list[tuple[str, list[str]]] = []

    # Prefer local installs first (usually clean stdio output)
    if shutil.which('keep-mcp'):
        candidates.append(('keep-mcp', []))
    if shutil.which('mcp'):
        # Some keep-mcp distributions expose the app as `mcp`
        candidates.append(('mcp', []))

    # Python module variants
    if _safe_find_spec('keep_mcp'):
        candidates.append((sys.executable, ['-m', 'keep_mcp']))
    if _safe_find_spec('keep_mcp.server'):
        candidates.append((sys.executable, ['-m', 'keep_mcp.server']))

    # Runner-based fallbacks
    if shutil.which('pipx'):
        # Prefer explicit app name to avoid wrapper notes
        candidates.append(('pipx', ['run', '--spec', 'keep-mcp', 'mcp']))
    if shutil.which('uvx'):
        # IMPORTANT: explicit app `mcp` avoids uvx note lines on stdout
        candidates.append(('uvx', ['--from', 'keep-mcp', 'mcp']))
        # last-resort legacy form (can emit NOTE lines)
        candidates.append(('uvx', ['keep-mcp']))
    if shutil.which('npx'):
        candidates.append(('npx', ['-y', 'keep-mcp']))

    if not candidates:
        raise RuntimeError(
            'No keep-mcp command candidate found. Install keep-mcp in the SAME Python env as this kernel.'
        )

    probe_results: list[dict[str, Any]] = []

    for command, args in candidates:
        cmd = [command, *args, '--help']
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            detail = (completed.stdout or completed.stderr or '').strip()[:260]
            status = 'ok' if completed.returncode == 0 else f'returncode_{completed.returncode}'

            # Reject obvious noisy wrappers that print NOTE banners to stdout
            if 'NOTE: running app' in (completed.stdout or ''):
                status = 'stdout_noise'

            probe_results.append({'command': command, 'args': args, 'status': status, 'detail': detail})
            if status == 'ok':
                return command, args, probe_results
        except subprocess.TimeoutExpired:
            probe_results.append({
                'command': command,
                'args': args,
                'status': 'ok_timeout',
                'detail': 'Command exists and appears long-running.',
            })
            return command, args, probe_results
        except Exception as exc:
            probe_results.append({'command': command, 'args': args, 'status': 'error', 'detail': str(exc)})

    raise RuntimeError(f'No clean keep-mcp command found. Probe results: {probe_results}')


keep_command, keep_args, keep_probe = resolve_keep_stdio_command()
print('Selected command:', keep_command, keep_args)
print('Probe summary:')
for row in keep_probe:
    print(' -', row)
```

```python
from langchain_mcp_adapters.client import MultiServerMCPClient


def _mask_secret(value: str | None, left: int = 3, right: int = 3) -> str:
    if not value:
        return '(missing)'
    if len(value) <= left + right:
        return '*' * len(value)
    return value[:left] + '*' * (len(value) - left - right) + value[-right:]


def build_keep_server_env() -> dict[str, str]:
    env = dict(os.environ)
    # Ensure the critical values are present in the child process environment.
    if os.getenv('GOOGLE_EMAIL'):
        env['GOOGLE_EMAIL'] = os.getenv('GOOGLE_EMAIL', '')
    if os.getenv('GOOGLE_MASTER_TOKEN'):
        env['GOOGLE_MASTER_TOKEN'] = os.getenv('GOOGLE_MASTER_TOKEN', '')
    return env


def build_keep_mcp_client(command: str, args: list[str], env: dict[str, str]) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            'keep': {
                'command': command,
                'args': args,
                'transport': 'stdio',
                'env': env,
            }
        }
    )


keep_env = build_keep_server_env()
print('Keep env diagnostics:')
print(' - GOOGLE_EMAIL:', _mask_secret(keep_env.get('GOOGLE_EMAIL')))
print(' - GOOGLE_MASTER_TOKEN:', _mask_secret(keep_env.get('GOOGLE_MASTER_TOKEN')))

keep_client = build_keep_mcp_client(keep_command, keep_args, keep_env)

try:
    keep_tools = await keep_client.get_tools()
except Exception as exc:
    print('Failed to initialize keep-mcp tools.')
    print('Selected command:', keep_command, keep_args)
    print('Error type:', type(exc).__name__)
    print('Error:', exc)
    print('Common fixes:')
    print('1) Ensure GOOGLE_EMAIL and GOOGLE_MASTER_TOKEN are set in this kernel environment.')
    print('2) If using uvx, run once in terminal to prewarm/download: uvx --from keep-mcp keep-mcp --help')
    print('3) Prefer local install if uvx keeps failing: pip install keep-mcp')
    raise

print('Keep tools loaded:', len(keep_tools))
for t in keep_tools:
    print('-', t.name)

if len(keep_tools) == 0:
    raise RuntimeError('Keep MCP returned zero tools. Check keep-mcp setup/auth.')
```

```python
# Inspect Keep tool interfaces
for t in keep_tools:
    print('=' * 90)
    print('Tool:', t.name)
    print('Description:', (t.description or '').strip()[:600])
    print('Args schema:', getattr(t, 'args', None))
```

## 10) Final ReAct Capstone: Local + Keep MCP Tools

Same agent paradigm, broader tool universe.
This is the core interoperability takeaway.

Note: MCP tools loaded through `langchain_mcp_adapters` can be async-only.
So in this section we call the agent with:
- `await run_agent_async(...)`
instead of synchronous `run_agent(...)`.

```python
all_tools = [*local_tools, *keep_tools]
capstone_agent = build_agent(chat_model, all_tools)

print('Total tools available:', len(all_tools))
print('First 15 tool names:', [t.name for t in all_tools][:15])
```

```python
# Capstone A: read Keep notes
cap_a_query = 'List my most recent Keep notes related to llm and agents.'
cap_a = await run_agent_async(capstone_agent, cap_a_query)

print('Final answer:')
print(cap_a['final'])
print()
print('Trace:')
print_tool_trace(cap_a['messages'])
```

```python
# Capstone B: create Keep note as final send-off
stamp = datetime.now().strftime('%Y-%m-%d %H:%M')
keep_title = f'IE685 ReAct MCP Send-Off {stamp}'
keep_body = 'Built custom tools, explored ReAct traces, then connected Keep MCP in one unified agent loop.'

cap_b_query = (
    'Create a new Google Keep note using this title and body. '
    f'Title: {keep_title}. Body: {keep_body}'
)
cap_b = await run_agent_async(capstone_agent, cap_b_query)

print('Final answer:')
print(cap_b['final'])
print()
print('Trace:')
print_tool_trace(cap_b['messages'])
```

```python
# Capstone C: verify the note exists
cap_c_query = f"Find and summarize the Keep note with title containing '{keep_title}'."
cap_c = await run_agent_async(capstone_agent, cap_c_query)

print('Final answer:')
print(cap_c['final'])
print()
print('Trace:')
print_tool_trace(cap_c['messages'])
```

### Try It Out 3 (Capstone Challenge)

Write one prompt that **must** use:
1. at least one local tool
2. at least one Keep MCP tool

Example:
- compute values
- include date
- save result in Keep
- then verify retrieval

```python
# TODO: Replace with your own mixed-tools capstone prompt
try_prompt_3 = (
    'Compute 44.4 + 55.6, include today\'s date, create a Keep note titled '
    '"IE685 mixed-tools demo" with those values, then summarize what you stored.'
)

try_run_3 = await run_agent_async(capstone_agent, try_prompt_3)
print('Final answer:')
print(try_run_3['final'])
print()
print('Trace:')
print_tool_trace(try_run_3['messages'])
```

## Reflection Questions

1. Which tool descriptions most improved the agent's action selection?
2. Where did the agent overuse tools, and how would you fix that?
3. How did streaming and memory improve transparency and usability?
4. What changed conceptually when you moved from local tools to MCP tools?
5. If deploying this in production, what action guardrails would be mandatory?

## Troubleshooting (Windows + macOS)

### Common issues
- Import errors after install: restart kernel and rerun from top.
- Missing `OPENAI_API_KEY`: model init will fail early.
- Keep tools empty: keep-mcp likely started but auth is incomplete.

### Windows notes
- `sys.executable -m keep_mcp` is often most reliable.
- Avoid shell command strings; command+args is safer for quoting/path handling.

### macOS notes
- If using `uvx keep-mcp`, ensure `uvx` is in the Jupyter kernel PATH.
- Complete any external auth prompts, then rerun Keep tool loading cell.

```python
# Optional cleanup
# if hasattr(keep_client, 'aclose'):
#     await keep_client.aclose()
```

## 12) Interactive Keep Chat (Memory)

This final cell gives you a persistent back-and-forth chat with a Keep-only agent.

How to use:
- Type normal requests (e.g. "create a note", "list my latest notes").
- The agent keeps short-term memory within the same `thread_id`.
- Type `exit` or `quit` to stop.

Tip: If you want a fresh conversation, change `thread_id` below.

```python
from langgraph.checkpoint.memory import InMemorySaver

chat_model = create_chat_model("gpt-5.2")

# Keep-only conversational agent with memory
keep_chat_memory = InMemorySaver()
keep_chat_agent = build_agent(
    chat_model,
    keep_tools,
    checkpointer=keep_chat_memory,
)

thread_id = 'ie685-keep-chat-live'
keep_chat_config = {'configurable': {'thread_id': thread_id}}

print('Interactive Keep chat is ready.')
print(f"thread_id={thread_id}")
print("Type your message. Type 'exit' or 'quit' to stop.")

while True:
    user_text = input('You: ').strip()
    if user_text.lower() in {'exit', 'quit'}:
        print('Session ended.')
        break
    if not user_text:
        continue

    reply = await run_agent_async(keep_chat_agent, user_text, config=keep_chat_config)
    print('Agent:', reply['final'])
    print()
```