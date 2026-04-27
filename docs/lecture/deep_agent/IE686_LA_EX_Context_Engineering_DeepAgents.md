# IE686 Capstone: Context Engineering with Deep Agents and Skill Files

This notebook builds a proposal-writing agent for the IE686 student projects.

- **Target course:** **IE686 Large Language Models and Agents (FSS 2026)**
- **Capstone focus:** **skill files, context engineering, deep agents, subagent isolation, and LangSmith traces**
- **End result:** a deep agent that turns a suggested topic into a research-backed project proposal draft

**Working references:**
- [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/index)
- [Deep Agents skills](https://docs.langchain.com/oss/python/deepagents/skills)
- [Deep Agents subagents](https://docs.langchain.com/oss/python/deepagents/subagents)
- [Trace Deep Agents in LangSmith](https://docs.langchain.com/langsmith/trace-deep-agents)

**Local course materials used for the rubric:**
- `IE686_LA_06_IntroStudentProjects.pdf`
- `IE685_LA_06_ContextEngineering.pptx`

---

## What this lab emphasizes

1. **Skill files as natural-language control surfaces**: students inspect and edit on-disk `SKILL.md` files.
2. **Context engineering as architecture**: write context, select context, compress context, isolate context.
3. **Progressive build-up**: research memo → outline → critique → final orchestrated proposal.
4. **Traceability**: use LangSmith to inspect delegation, tool calls, and revision loops.

---

## How to use this notebook

- Run top-to-bottom the first time.
- Keep the checked-in skill files open in another editor tab while you experiment.
- For each comparison block, keep the prompt fixed and change only the skill configuration.
- The default example topic is **creating a secretary agent using OpenClaw**, but the final cells let you swap in your own project topic.

---

## 1. Setup and preflight

This notebook assumes live web research and LangSmith tracing.

**Required environment variables:**
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `LANGSMITH_API_KEY`

**Optional:**
- `DEEPAGENT_MODEL` to override the default model (`openai:gpt-5-mini`)
- `LANGSMITH_PROJECT` to set a custom tracing project name

### Install dependencies

```python
%pip install -qU deepagents==0.4.11 langchain==1.2.12 langchain-openai==1.1.11 langsmith==0.7.6 tavily-python==0.7.23 python-dotenv pydantic requests beautifulsoup4
```

### Initialize imports and paths

```python
import json
import os
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from IPython.display import Markdown, display
from langchain_core.tools import tool
from langchain_core.tracers.context import collect_runs
from langchain_openai import ChatOpenAI
from langsmith import Client
from pydantic import BaseModel, Field
from tavily import TavilyClient

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

load_dotenv()

ROOT = Path.cwd()
LAB_ROOT = ROOT / "deepagents_lab"
SKILLS_ROOT = LAB_ROOT / "skills"
WORKSPACE_ROOT = LAB_ROOT / "workspace"

SHARED_SKILLS = "/skills/shared"
MAIN_SKILLS = "/skills/main"
RESEARCH_SKILLS = "/skills/research"
REVIEW_SKILLS = "/skills/review"

MODEL_NAME = os.getenv("DEEPAGENT_MODEL", "openai:gpt-5-mini")
BASELINE_MODEL_NAME = MODEL_NAME.split(":", 1)[1] if MODEL_NAME.startswith("openai:") else MODEL_NAME

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "IE686-DeepAgents-Proposal-Lab")

print("Project root:", ROOT)
print("Lab root:", LAB_ROOT)
print("Skills root:", SKILLS_ROOT)
print("Workspace root:", WORKSPACE_ROOT)
print("Deep agent model:", MODEL_NAME)
print("LangSmith project:", os.getenv("LANGSMITH_PROJECT"))
```

### Environment preflight check

```python
required_env = ["OPENAI_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY"]
env_rows = [
    {
        "variable": name,
        "present": bool(os.getenv(name)),
        "preview": (os.getenv(name)[:6] + "...") if os.getenv(name) else "",
    }
    for name in required_env
]

skill_rows = []
for skill_file in sorted(SKILLS_ROOT.rglob("SKILL.md")):
    skill_rows.append(
        {
            "skill": skill_file.parent.name,
            "path": str(skill_file.relative_to(ROOT)),
            "bytes": skill_file.stat().st_size,
        }
    )

print("Environment preflight")
display(pd.DataFrame(env_rows))
print("\nChecked-in skill files")
display(pd.DataFrame(skill_rows))

if not all(row["present"] for row in env_rows):
    raise ValueError("Missing one or more required API keys. Add them to your environment or .env file before continuing.")
```

We intentionally use a **filesystem-backed deep agent workspace** rooted at `deepagents_lab/`.

**Why:**
- skills live on disk and can be edited directly,
- agents can write research memos and drafts to `/workspace/...`,
- `virtual_mode=True` gives stable POSIX-style paths (`/skills/...`, `/workspace/...`) across macOS and Windows.

---

## 2. Inspect the checked-in skill library

Before running any agent, look at the skills that will steer it.
The point of this exercise is that **natural-language instructions on disk** can change agent behavior without changing Python code.

### List all available skills

```python
def list_skills() -> pd.DataFrame:
    rows = []
    for skill_file in sorted(SKILLS_ROOT.rglob("SKILL.md")):
        text = skill_file.read_text(encoding="utf-8")
        name_match = re.search(r"^name:\s*(.+)$", text, flags=re.MULTILINE)
        desc_match = re.search(r"^description:\s*(.+)$", text, flags=re.MULTILINE)
        rows.append(
            {
                "lane": skill_file.parent.parent.name,
                "skill": name_match.group(1).strip() if name_match else skill_file.parent.name,
                "description": desc_match.group(1).strip() if desc_match else "",
                "path": str(skill_file.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


list_skills()
```

### Display a specific skill file

```python
def show_skill(skill_name: str) -> None:
    matches = list(SKILLS_ROOT.rglob(f"{skill_name}/SKILL.md"))
    if not matches:
        raise FileNotFoundError(f"No skill named {skill_name!r} found.")

    skill_file = matches[0]
    print(f"=== {skill_file.relative_to(ROOT)} ===")
    print(skill_file.read_text(encoding="utf-8"))


show_skill("benchmark-research")
```

> **Try it out:**
> - Open `deepagents_lab/skills/research/benchmark-research/SKILL.md`.
> - Which parts are general enough to reuse on other topics?
> - Which parts are specific enough to change what the agent does?

---

## 3. Baseline: no skills, no explicit context engineering

Start with the simplest possible baseline: ask a strong model to write a proposal directly.
This is useful because it gives us a concrete "before" picture.

```python
DEFAULT_TOPIC = (
    "Creating a secretary agent using OpenClaw: build an agent assistant that can help with inbox triage, "
    "calendar coordination, meeting preparation, and follow-up task tracking, then evaluate it against "
    "realistic office workflows and recent related systems."
)


def build_outline_request(topic: str) -> str:
    return textwrap.dedent(
        f"""
        Draft a project outline for the following student project topic:
        {topic}

        The outline must answer these questions:
        1. What is the problem you are solving?
        2. What data will you use, where will you get it, and how will you gather it?
        3. How will you solve the problem, which LLMs or methods will you use, and what is your multi-agent workflow?
        4. How will you measure success?

        Keep the answer concise and proposal-oriented.
        """
    ).strip()


baseline_model = ChatOpenAI(model=BASELINE_MODEL_NAME, temperature=0)
baseline_prompt = build_outline_request(DEFAULT_TOPIC)
baseline_response = baseline_model.invoke(baseline_prompt)
display(Markdown(baseline_response.content))
```

> **Typical baseline weaknesses:**
> - named benchmarks may be missing or unsupported,
> - the data collection plan stays vague,
> - the workflow is generic,
> - there is no explicit trace of how the agent reached its choices.

---

## 4. Filesystem-backed file handling with deep agents

Before introducing skills, it helps to show that a deep agent can use a shared filesystem-backed workspace.

In this section the agent will:
- read a seed file from `/workspace/scratch/`,
- write a derived note back into `/workspace/scratch/`,
- return a short explanation of what it did.

This is the simplest way to show that file handling is already part of the agent runtime, even before we add skill files.

### Why this matters

- a proposal workflow usually needs intermediate artifacts such as memos, outlines, and review notes,
- those artifacts are easier to inspect on disk than when hidden inside a long chat history,
- later agent steps can discover and reuse files instead of requiring the full context to be pasted again.

In practice, this means that context engineering is not only about prompts. It is also about **where state lives** and **how the agent can find it again**.

### Define custom tools and agent helpers

```python
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@tool
def tavily_search(query: str, max_results: int = 5, include_domains_csv: str = "") -> str:
    """Search the live web for current systems, benchmarks, datasets, APIs, and evaluation setups."""

    include_domains = [d.strip() for d in include_domains_csv.split(",") if d.strip()]
    raw = tavily_client.search(
        query=query,
        search_depth="advanced",
        topic="general",
        max_results=max_results,
        include_domains=include_domains or None,
        include_answer=False,
        include_raw_content="text",
    )

    rows = []
    for item in raw.get("results", [])[:max_results]:
        rows.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content_snippet": (item.get("content", "") or "")[:500],
            }
        )
    return json.dumps(rows, indent=2)


@tool
def fetch_url_excerpt(url: str, max_chars: int = 3000) -> str:
    """Fetch the visible text of a webpage so the agent can inspect details after search."""

    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; IE686-DeepAgents-Lab/1.0)"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.stripped_strings)
    return text[:max_chars]


proposal_tools = [tavily_search, fetch_url_excerpt]


def make_backend() -> FilesystemBackend:
    return FilesystemBackend(root_dir=LAB_ROOT, virtual_mode=True)


def reset_workspace() -> None:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    for child in WORKSPACE_ROOT.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    variant_root = LAB_ROOT / "skill_variants"
    if variant_root.exists():
        shutil.rmtree(variant_root)

    for relative in ["research", "outline", "review", "final", "scratch"]:
        (WORKSPACE_ROOT / relative).mkdir(parents=True, exist_ok=True)


def extract_text_response(result: dict[str, Any]) -> str:
    last_message = result["messages"][-1]
    content = last_message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or str(block))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    return str(content)


def invoke_with_trace(agent, prompt: str, run_name: str) -> tuple[dict[str, Any], str | None]:
    with collect_runs() as runs_cb:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"run_name": run_name},
        )

    run_url = None
    if runs_cb.traced_runs and os.getenv("LANGSMITH_API_KEY"):
        client = Client()
        run_url = client.get_run_url(run=runs_cb.traced_runs[0])
    return result, run_url


def build_lab_agent(skill_sources: list[str] | None):
    return create_deep_agent(
        model=MODEL_NAME,
        tools=proposal_tools,
        system_prompt=(
            "You help draft IE686 student project proposals. "
            "Use current evidence only."
        ),
        backend=make_backend(),
        skills=skill_sources,
        name="ie686_proposal_lab",
    )
```

### Built-in filesystem tools from `FilesystemMiddleware`

When you pass a `FilesystemBackend` into `create_deep_agent(...)`, Deep Agents already injects a file-oriented toolset through `FilesystemMiddleware`.
For this lab, that means we do **not** need to recreate custom file tools just to show discovery and editing.

The built-in tools that matter most here are:
- `ls`: discover files and directories
- `read_file`: inspect file contents
- `write_file`: create a new file
- `edit_file`: update an existing file

The middleware also provides `glob` and `grep` for broader discovery.

### Helper functions for workspace visualization

```python
def render_workspace_tree(relative_dir: str = "") -> str:
    clean = relative_dir.strip().strip("/")
    base = WORKSPACE_ROOT / Path(clean) if clean else WORKSPACE_ROOT
    if not base.exists():
        return f"/workspace/{clean} does not exist" if clean else "/workspace does not exist"

    lines = [f"$ ls -R /workspace/{clean}" if clean else "$ ls -R /workspace"]
    for path in sorted(base.rglob("*")):
        rel = path.relative_to(WORKSPACE_ROOT)
        kind = "dir" if path.is_dir() else "file"
        lines.append(f"{kind}: /workspace/{rel.as_posix()}")
    return "\n".join(lines)


def write_workspace_file(relative_path: str, content: str) -> Path:
    path = WORKSPACE_ROOT / Path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
```

### Filesystem demo: seed a file and let the agent use it

```python
reset_workspace()

seed_file = write_workspace_file(
    "scratch/topic_brief.md",
    textwrap.dedent(
        """
        Topic: Creating a secretary agent using OpenClaw

        Goal:
        Build an agent assistant for inbox triage, calendar coordination, meeting preparation,
        and follow-up task tracking.
        """
    ).strip()
    + "\n",
)

print("=== Notebook-side setup ===\n")
print(render_workspace_tree("scratch"))
print("\n=== Seed file ===\n")
print(seed_file.read_text(encoding="utf-8"))
```

### Run the filesystem demo agent

```python
def build_file_demo_agent():
    return create_deep_agent(
        model=MODEL_NAME,
        tools=[],
        backend=make_backend(),
        skills=None,
        system_prompt=(
            "You are demonstrating filesystem-backed context for an IE686 lab. "
            "Use the built-in filesystem tools from the backend. "
            "Start with `ls` before reading or modifying files, "
            "keep the edits concise, and explain which files you used."
        ),
        name="filesystem_demo_agent",
    )


reset_workspace()
write_workspace_file(
    "scratch/topic_brief.md",
    textwrap.dedent(
        """
        Topic: Creating a secretary agent using OpenClaw

        Goal:
        Build an agent assistant for inbox triage, calendar coordination, meeting preparation,
        and follow-up task tracking.

        Constraints:
        - The proposal should stay realistic for a student team.
        - The idea should reference recent related systems and evaluation setups.
        - Open questions should be separated from supported claims.
        """
    ).strip()
    + "\n",
)

file_demo_agent = build_file_demo_agent()
file_demo_prompt = textwrap.dedent(
    """
    First use `ls` to discover the workspace files.

    Then use `read_file` on /workspace/scratch/topic_brief.md.

    Then use `write_file` to create /workspace/scratch/proposal_scaffold.md with:
    1. a short prose summary of the project idea,
    2. three concrete follow-up questions the agent should research next.

    Finally, use `edit_file` once to add one more line at the end:
    "Open question: which office workflow should be the first evaluation scenario?"

    Return a short note describing which filesystem tools you used and in what order.
    """
).strip()

file_demo_result, file_demo_run_url = invoke_with_trace(
    file_demo_agent,
    file_demo_prompt,
    run_name="deepagents-filesystem-demo",
)

print(extract_text_response(file_demo_result))
print("\nFilesystem demo run:", file_demo_run_url or "LangSmith URL unavailable")
```

> **Try it out:**
> - Edit `deepagents_lab/workspace/scratch/topic_brief.md`.
> - Rerun only the filesystem demo cells.
> - Check the trace to see whether the agent starts with `ls`, then moves through `read_file`, `write_file`, and `edit_file`.

---

## 5. Skills as natural-language steering

Now we give a deep agent access to live web tools and the checked-in skill folders.
The user prompt stays nearly the same; the main change is **what natural-language instructions the agent can discover on disk**.

### Exact skill files used in section 5

The skill-enabled comparison in this section uses these checked-in files:

- [`project-outline-rubric`](deepagents_lab/skills/shared/project-outline-rubric/SKILL.md)
- [`proposal-writer`](deepagents_lab/skills/main/proposal-writer/SKILL.md)
- [`benchmark-research`](deepagents_lab/skills/research/benchmark-research/SKILL.md)

The critique skill is part of the same lab library and is introduced later:

- [`proposal-critic`](deepagents_lab/skills/review/proposal-critic/SKILL.md)

These skills are also what introduce the more polished writing behavior:
executive-summary prose, recent scientific grounding, and source-backed references.

### Compare: deep agent without skills vs. with checked-in skills

```python
reset_workspace()
no_skill_agent = build_lab_agent(skill_sources=None)
skilled_agent = build_lab_agent(skill_sources=[SHARED_SKILLS, MAIN_SKILLS, RESEARCH_SKILLS])

comparison_prompt = build_outline_request(DEFAULT_TOPIC)

no_skill_result, no_skill_run_url = invoke_with_trace(
    no_skill_agent,
    comparison_prompt,
    run_name="deepagents-no-skills",
)
skilled_result, skilled_run_url = invoke_with_trace(
    skilled_agent,
    comparison_prompt,
    run_name="deepagents-with-skills",
)

no_skill_text = extract_text_response(no_skill_result)
skilled_text = extract_text_response(skilled_result)

print("=== Deep agent without skills ===\n")
print(no_skill_text[:4000])
print("\n" + "=" * 100 + "\n")
print("=== Deep agent with checked-in skills ===\n")
print(skilled_text[:4000])
```

```python
print("No-skill LangSmith run:", no_skill_run_url or "LangSmith URL unavailable")
print("Skilled LangSmith run:", skilled_run_url or "LangSmith URL unavailable")
```

> **Try it out:**
> - Keep the prompt identical and edit one sentence in `deepagents_lab/skills/research/benchmark-research/SKILL.md`.
> - Rerun only the skilled agent cell.
> - Check whether the output becomes more concrete, more current, or more generic, and then inspect the trace in LangSmith.

---

## 6. Better skill design: weak trigger wording vs strong trigger wording

Skill quality matters twice:
1. the **description** controls whether the skill is loaded,
2. the **body** controls what the agent actually does after it loads the skill.

In this section we create a temporary weak variant of the research skill to show how vague trigger text hurts performance.

### Create a weak skill variant for comparison

```python
def make_weak_research_skill_variant() -> str:
    variant_root = LAB_ROOT / "skill_variants" / "weak_research"
    skill_dir = variant_root / "benchmark-research"
    if variant_root.exists():
        shutil.rmtree(variant_root)
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)

    weak_skill = textwrap.dedent(
        """
        ---
        name: benchmark-research
        description: Use when the user needs help with something related to projects or information.
        ---

        # Weak benchmark research skill

        Try to be helpful.
        """
    ).strip() + "\n"

    source_reference = (
        SKILLS_ROOT
        / "research"
        / "benchmark-research"
        / "references"
        / "research_checklist.md"
    )
    shutil.copy2(source_reference, skill_dir / "references" / "research_checklist.md")
    (skill_dir / "SKILL.md").write_text(weak_skill, encoding="utf-8")
    return "/skill_variants/weak_research"


weak_research_source = make_weak_research_skill_variant()

strong_skill_file = SKILLS_ROOT / "research" / "benchmark-research" / "SKILL.md"
weak_skill_file = LAB_ROOT / "skill_variants" / "weak_research" / "benchmark-research" / "SKILL.md"

print("=== Strong description ===")
print(re.search(r"^description:\s*(.+)$", strong_skill_file.read_text(encoding="utf-8"), flags=re.MULTILINE).group(1))
print("\n=== Weak description ===")
print(re.search(r"^description:\s*(.+)$", weak_skill_file.read_text(encoding="utf-8"), flags=re.MULTILINE).group(1))
```

### Run agent with weak skill variant

```python
weak_skill_agent = build_lab_agent(skill_sources=[SHARED_SKILLS, MAIN_SKILLS, weak_research_source])

weak_result, weak_run_url = invoke_with_trace(
    weak_skill_agent,
    comparison_prompt,
    run_name="deepagents-weak-skill",
)
weak_text = extract_text_response(weak_result)
```

```python
print("Weak-skill LangSmith run:", weak_run_url or "LangSmith URL unavailable")
print("\nWeak-skill output preview:\n")
print(weak_text[:3000])
```

In practice, the exact difference will vary by model and topic.
The main point is architectural: **skill descriptions are part of your retrieval layer for procedural knowledge**.

---

## 7. Build the individual pieces

We now implement the proposal pipeline as separate components:
1. research memo agent
2. outline agent
3. critique agent

These pieces correspond to the context-engineering techniques from the lecture:
- **write context**: agents save notes and drafts to `/workspace/...`
- **select context**: the research agent pulls only relevant web evidence
- **compress context**: the memo and outline are concise handoff artifacts
- **isolate context**: each component gets only the skill set it actually needs

### Define Pydantic schemas for structured outputs

```python
class ResearchMemo(BaseModel):
    comparable_systems: list[str] = Field(default_factory=list)
    candidate_datasets_or_apis: list[str] = Field(default_factory=list)
    benchmarks_and_metrics: list[str] = Field(default_factory=list)
    feasibility_notes: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class ProposalOutline(BaseModel):
    proposed_title: str
    problem: str
    data_plan: str
    planned_methods_and_models: str
    multi_agent_workflow: str
    evaluation_plan: str


class ReviewFeedback(BaseModel):
    missing_items: list[str] = Field(default_factory=list)
    weak_claims: list[str] = Field(default_factory=list)
    improvement_actions: list[str] = Field(default_factory=list)
    verdict: str


class ProposalDraft(BaseModel):
    proposed_title: str
    problem: str
    data_and_collection: str
    methods_and_models: str
    multi_agent_workflow: str
    evaluation_plan: str
    risks_and_open_questions: str
    source_urls: list[str] = Field(default_factory=list)
    final_markdown: str


def ensure_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content.strip() + "\n", encoding="utf-8")


def memo_to_markdown(memo: ResearchMemo) -> str:
    sections = {
        "Comparable systems": memo.comparable_systems,
        "Candidate datasets or APIs": memo.candidate_datasets_or_apis,
        "Benchmarks and metrics": memo.benchmarks_and_metrics,
        "Feasibility notes": memo.feasibility_notes,
        "Open questions": memo.open_questions,
        "Source URLs": memo.source_urls,
    }
    lines = ["# Research memo", ""]
    for heading, items in sections.items():
        lines.append(f"## {heading}")
        if items:
            lines.extend([f"- {item}" for item in items])
        else:
            lines.append("- None")
        lines.append("")
    return "\n".join(lines)


def outline_to_markdown(outline: ProposalOutline) -> str:
    sections = {
        "Problem": outline.problem,
        "Data and data collection": outline.data_plan,
        "Planned methods and models": outline.planned_methods_and_models,
        "Multi-agent workflow": outline.multi_agent_workflow,
        "Evaluation plan": outline.evaluation_plan,
    }
    lines = [f"# {outline.proposed_title}", ""]
    for heading, paragraph in sections.items():
        lines.append(f"## {heading}")
        lines.append(paragraph.strip())
        lines.append("")
    return "\n".join(lines)


def review_to_markdown(review: ReviewFeedback) -> str:
    sections = {
        "Missing items": review.missing_items,
        "Weak claims": review.weak_claims,
        "Improvement actions": review.improvement_actions,
    }
    lines = [f"# Review verdict: {review.verdict}", ""]
    for heading, items in sections.items():
        lines.append(f"## {heading}")
        if items:
            lines.extend([f"- {item}" for item in items])
        else:
            lines.append("- None")
        lines.append("")
    return "\n".join(lines)
```

### Build individual specialized agents

```python
def build_research_agent():
    return create_deep_agent(
        model=MODEL_NAME,
        tools=proposal_tools,
        backend=make_backend(),
        skills=[SHARED_SKILLS, RESEARCH_SKILLS],
        response_format=ResearchMemo,
        system_prompt=(
            "You are the research stage of an IE686 project-proposal pipeline. "
            "Always start by planning with write_todos for non-trivial tasks. "
            "Use live web research, capture exact benchmark and system names, and save a concise memo to "
            "/workspace/research/research_memo.md before returning. "
            "Look for recent scientific writing when it helps make the proposal more concrete."
        ),
        name="proposal_research_agent",
    )


def build_outline_agent():
    return create_deep_agent(
        model=MODEL_NAME,
        tools=[],
        backend=make_backend(),
        skills=[SHARED_SKILLS, MAIN_SKILLS],
        response_format=ProposalOutline,
        system_prompt=(
            "You convert research notes into a concrete project outline. "
            "Read /workspace/research/research_memo.md, apply the rubric skill, and save the draft to "
            "/workspace/outline/project_outline.md before returning. "
            "Write it as a compact executive summary in prose, roughly three pages in length. "
            "Include source information and a short references section based on the sources actually used."
        ),
        name="proposal_outline_agent",
    )


def build_critic_agent():
    return create_deep_agent(
        model=MODEL_NAME,
        tools=[],
        backend=make_backend(),
        skills=[SHARED_SKILLS, REVIEW_SKILLS],
        response_format=ReviewFeedback,
        system_prompt=(
            "You critique IE686 proposal drafts. "
            "Read /workspace/outline/project_outline.md, apply the rubric and critic skills, "
            "and save actionable feedback to /workspace/review/review_feedback.md before returning."
        ),
        name="proposal_critic_agent",
    )
```

### Run research agent

```python
reset_workspace()

research_agent = build_research_agent()
research_prompt = textwrap.dedent(
    f"""
    Prepare a research memo for this project topic:
    {DEFAULT_TOPIC}

    Find concrete comparable systems, datasets or APIs, benchmarks, and evaluation metrics.
    Look for recent scientific writing when it materially sharpens the proposal.
    Keep the memo focused on what would help a student team write a realistic project outline.
    """
).strip()

research_result, research_run_url = invoke_with_trace(
    research_agent,
    research_prompt,
    run_name="proposal-piece-research",
)

research_memo = research_result["structured_response"]
ensure_text_file(WORKSPACE_ROOT / "research" / "research_memo.md", memo_to_markdown(research_memo))

display(pd.Series(research_memo.model_dump(), name="value").to_frame())
print("Research run:", research_run_url or "LangSmith URL unavailable")
```

### Run outline agent

```python
outline_agent = build_outline_agent()
outline_prompt = textwrap.dedent(
    f"""
    Read /workspace/research/research_memo.md and draft the project outline for:
    {DEFAULT_TOPIC}

    Answer the four official outline questions.
    Write the result as a compact executive summary in prose, roughly three pages in length, not as a bullet list.
    Include source information and a short references section based on the sources actually used.
    """
).strip()

outline_result, outline_run_url = invoke_with_trace(
    outline_agent,
    outline_prompt,
    run_name="proposal-piece-outline",
)

outline = outline_result["structured_response"]
ensure_text_file(WORKSPACE_ROOT / "outline" / "project_outline.md", outline_to_markdown(outline))

display(Markdown(outline_to_markdown(outline)))
print("Outline run:", outline_run_url or "LangSmith URL unavailable")
```

### Run critique agent

```python
critic_agent = build_critic_agent()
critic_prompt = (
    "Review /workspace/outline/project_outline.md against the course outline rubric. "
    "Identify missing specificity, unsupported claims, weak evaluation, or places where the draft falls back into a bullet-list style."
)

critic_result, critic_run_url = invoke_with_trace(
    critic_agent,
    critic_prompt,
    run_name="proposal-piece-critic",
)

review = critic_result["structured_response"]
ensure_text_file(WORKSPACE_ROOT / "review" / "review_feedback.md", review_to_markdown(review))

display(Markdown(review_to_markdown(review)))
print("Critic run:", critic_run_url or "LangSmith URL unavailable")
```

> **Try it out:**
> - Change the topic and rerun only the research cell.
> - Inspect whether the outline and critique improve when the research memo becomes more specific.
> - Open the run traces and compare the tool usage of the research agent versus the outline and critic agents.

---

## 8. Final deep agent with isolated subagents

The final capstone combines everything:
- live web tools,
- checked-in skills,
- file-backed notes,
- isolated subagents,
- final structured output,
- LangSmith trace.

The important design point is that **subagents do not automatically inherit every skill**.
We pass each subagent its own `skills=[...]` configuration.

### Build the orchestrator with subagents

```python
def build_final_orchestrator():
    return create_deep_agent(
        model=MODEL_NAME,
        tools=proposal_tools,
        backend=make_backend(),
        skills=[SHARED_SKILLS, MAIN_SKILLS],
        response_format=ProposalDraft,
        system_prompt=(
            "You orchestrate an IE686 project-proposal writing pipeline. "
            "For any real proposal task you must first write todos, then delegate external research to the "
            "`researcher` subagent, delegate workflow and outline construction to `workflow_designer`, "
            "delegate draft review to `critic`, revise the draft, and save the final markdown to "
            "/workspace/final/final_proposal.md. "
            "Write the final draft as a compact executive summary in prose, roughly three pages in length. "
            "Only include datasets, benchmarks, systems, or recent scientific writing that are grounded in the gathered evidence. "
            "Include source information and a short references section based on the sources actually used."
        ),
        subagents=[
            {
                "name": "researcher",
                "description": "Researches comparable systems, datasets, APIs, benchmarks, and evaluation metrics for a proposed project topic.",
                "system_prompt": (
                    "You are the research specialist. Use the web tools, follow the benchmark-research skill, "
                    "look for recent scientific writing when it materially sharpens the proposal, "
                    "and write a concise memo to /workspace/research/research_memo.md. "
                    "Return a short summary with exact names and URLs."
                ),
                "tools": proposal_tools,
                "skills": [SHARED_SKILLS, RESEARCH_SKILLS],
            },
            {
                "name": "workflow_designer",
                "description": "Turns research notes into a concrete project outline with a plausible multi-agent workflow and evaluation plan.",
                "system_prompt": (
                    "You are the proposal drafting specialist. Read /workspace/research/research_memo.md, "
                    "apply the rubric and proposal-writer skills, and write the draft to "
                    "/workspace/outline/project_outline.md. Write the draft as a compact executive summary in prose, "
                    "not as a bullet list. Include source information and a short references section based on the "
                    "sources actually used. Return a concise summary of the structure."
                ),
                "tools": [],
                "skills": [SHARED_SKILLS, MAIN_SKILLS],
            },
            {
                "name": "critic",
                "description": "Critiques a proposal draft for missing rubric coverage, vague claims, and weak evaluation.",
                "system_prompt": (
                    "You are the proposal critic. Read /workspace/outline/project_outline.md or "
                    "/workspace/final/final_proposal.md, follow the proposal-critic skill, and write feedback to "
                    "/workspace/review/review_feedback.md. Return concise actionable feedback."
                ),
                "tools": [],
                "skills": [SHARED_SKILLS, REVIEW_SKILLS],
            },
        ],
        name="ie686_proposal_orchestrator",
    )


final_agent = build_final_orchestrator()
```

### Run the final orchestrator

```python
TOPIC = DEFAULT_TOPIC

final_prompt = textwrap.dedent(
    f"""
    Build a strong project proposal draft for this suggested topic:
    {TOPIC}

    Requirements:
    - answer the official four project-outline questions,
    - research comparable systems, candidate datasets or APIs, and suitable benchmarks or metrics,
    - use recent scientific writing when it helps make the proposal more concrete and current,
    - propose a concrete multi-agent workflow,
    - critique the draft and revise it once before finalizing,
    - include source information and a short references section based on the sources actually used,
    - write the final result as a compact executive summary in prose, roughly three pages in length,
    - save the final markdown draft to /workspace/final/final_proposal.md.
    """
).strip()

reset_workspace()
final_result, final_run_url = invoke_with_trace(
    final_agent,
    final_prompt,
    run_name="deepagents-final-proposal",
)

final_draft = final_result["structured_response"]
ensure_text_file(WORKSPACE_ROOT / "final" / "final_proposal.md", final_draft.final_markdown)

display(Markdown(final_draft.final_markdown))
print("Final LangSmith run:", final_run_url or "LangSmith URL unavailable")
```

### Inspect workspace artifacts

```python
artifact_rows = []
for path in sorted(WORKSPACE_ROOT.rglob("*")):
    if path.name == ".gitkeep":
        continue
    artifact_rows.append(
        {
            "path": str(path.relative_to(ROOT)),
            "is_dir": path.is_dir(),
            "bytes": path.stat().st_size if path.is_file() else 0,
        }
    )

pd.DataFrame(artifact_rows)
```

### Display generated artifacts

```python
for relative in [
    "research/research_memo.md",
    "outline/project_outline.md",
    "review/review_feedback.md",
    "final/final_proposal.md",
]:
    path = WORKSPACE_ROOT / relative
    if path.exists():
        print(f"\n=== {relative} ===\n")
        print(path.read_text(encoding="utf-8")[:4000])
```

> **What to inspect in LangSmith for the final run:**
> - Did the main agent call `task` and delegate to the correct subagent?
> - Which skills appear to have guided the research and critique phases?
> - Which files were written into `/workspace/`?
> - Where did the revision loop improve the draft?

---

## 9. Final try-it-out

Replace `TOPIC` with your own assigned student-project topic and rerun the final orchestration cells.

**Good variants:**
- `Text-to-SQL`
- `Online Shopping Assistant`
- `Text-to-BPMN`
- a more specific subtopic inside one of those areas

When you change the topic, keep the skills fixed first. Only after that should you edit the skills and inspect how the traces change.

---

## Summary: Key Takeaways

This notebook demonstrates several core concepts in context engineering with deep agents:

### 1. Skill Files as Control Surfaces
- `SKILL.md` files on disk act as natural-language instructions that steer agent behavior
- Editing skill files changes agent behavior **without changing Python code**
- Skill descriptions are part of the retrieval layer for procedural knowledge

### 2. Context Engineering Techniques
- **Write context**: agents save notes/drafts to `/workspace/...`
- **Select context**: research agents pull only relevant evidence
- **Compress context**: memos and outlines are concise handoff artifacts
- **Isolate context**: each subagent gets only the skills it actually needs

### 3. Filesystem-Backed Workspace
- `FilesystemBackend` with `virtual_mode=True` provides stable POSIX paths
- Intermediate artifacts are inspectable on disk
- Later agent steps can discover and reuse files

### 4. Subagent Isolation
- Subagents do not automatically inherit every skill
- Each subagent receives its own `skills=[...]` configuration
- This enables focused, specialized behavior

### 5. LangSmith Tracing
- All agent runs can be traced in LangSmith
- Traces show delegation, tool calls, and revision loops
- Useful for debugging and understanding agent behavior
