# IE686 Exercise: LLM Workflows with LangChain

This notebook focuses on practical LangChain workflow design.

- **Target course:** IE686 Large Language Models and Agents (FSS2026)
- **Focus in this notebook:** Prompting, structured outputs, routing, parallelization, and tool calling

## Documentation Quick Links

- [LangChain Python overview](https://docs.langchain.com/oss/python/langchain/overview)
- [Models](https://docs.langchain.com/oss/python/langchain/models)
- [Messages](https://docs.langchain.com/oss/python/langchain/messages)
- [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [Few-shot prompting](https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate)
- [Similarity-based example selection](https://reference.langchain.com/python/langchain-core/example_selectors/semantic_similarity/SemanticSimilarityExampleSelector)

---

## Why LangChain Instead of Direct Provider Calls?

Direct provider SDKs are great for simple single-call use cases. LangChain becomes useful when you need reusable workflows across models and providers.

**Use LangChain when you want:**

- A common interface for prompts, models, tools, and outputs
- Easier composition of multi-step workflows (routing, parallel steps, retries)
- Structured outputs with schema validation
- Cleaner migration between providers with less rewrite

**Rule of thumb:** Start direct for very small scripts; use LangChain once your workflow has multiple steps or needs to stay maintainable.

**Useful references:**

- [Models and invocation](https://docs.langchain.com/oss/python/langchain/models)
- [Runnable composition (LCEL)](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.base.Runnable)

---

## 1. Setup

Install the current LangChain packages and load environment variables.

**Required API key options:**

- `OPENAI_API_KEY` (recommended)
- `GROQ_API_KEY` (optional fallback)

**References:**

- [Install LangChain](https://docs.langchain.com/oss/python/langchain/install)
- [Model initialization and invoke/batch/stream](https://docs.langchain.com/oss/python/langchain/models)

### Install Required Packages

```python
%pip install -qU langchain==1.2.10 langchain-openai==1.1.10 langchain-groq python-dotenv pydantic
```

### Initialize the LLM

This section sets up the language model with support for both OpenAI and Groq backends. The code checks for available API keys and initializes the appropriate model.

```python
import os
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI

llm = None
if os.getenv("OPENAI_API_KEY"):
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        temperature=0.2,
    )
elif os.getenv("GROQ_API_KEY"):
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.2,
    )
else:
    raise ValueError("Set OPENAI_API_KEY or GROQ_API_KEY in your environment before running.")

print(f"Model backend ready: {llm.__class__.__name__}")
```

### Warm-up Test

Test the model with a simple invocation to ensure it's working correctly.

```python
warmup = llm.invoke("Respond with exactly 8 words: welcome students to workflow engineering.")
warmup
```

### Inspect the Response Object

Before building workflows, it's important to inspect what the model returns. The response is usually an `AIMessage`, not only plain text. It can include provider metadata, token usage, finish reason, and tool calls.

```python
# The return object is usually an AIMessage, not just a plain string.
print("Type:", type(warmup).__name__)
print("\ncontent:\n", warmup.content)
print("\nresponse_metadata keys:", list((warmup.response_metadata or {}).keys()))
print("usage_metadata:", warmup.usage_metadata)
print("tool_calls:", warmup.tool_calls)
```

**References:**

- [Messages](https://docs.langchain.com/oss/python/langchain/messages)
- [AIMessage reference](https://reference.langchain.com/python/langchain-core/messages/ai)

> **Try it:**
> - Change `temperature` to `0.8` and compare output variability.
> - Try another model name (for example, `gpt-5-nano` or a Groq model you have access to).

---

## 2. Prompt Workflow (Template → Model → Output Parser)

This is the core LangChain pattern for many LLM applications.

**Why this matters:**

- Keeps prompts reusable and parameterized
- Makes behavior more consistent across calls
- Separates prompt construction from model invocation and output handling

**References:**

- [Prompt classes API](https://reference.langchain.com/python/langchain_core/prompts/)
- [ChatPromptTemplate API](https://reference.langchain.com/python/langchain-core/prompts/chat/ChatPromptTemplate)

### Email Rewriting Example

This example demonstrates creating a prompt template, chaining it with a model, and parsing the output to rewrite emails in different tones.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

email_prompt = ChatPromptTemplate.from_template(
    """
You are an assistant for business communication.
Rewrite the draft email for a {tone} tone.
Keep it under {max_words} words.

Draft email:
{draft}
    """.strip()
)

email_chain = email_prompt | llm | StrOutputParser()

draft_email = "Hi team, we are late again. Please make sure this does not happen next sprint."
response = email_chain.invoke({"tone": "constructive", "max_words": 35, "draft": draft_email})
print(response)
```

> **Try it:**
> - Set `tone` to `firm`, `friendly`, and `executive`.
> - Lower `max_words` to 20.
> - Replace the draft with one from your own project communication.

---

## 3. Few-Shot Prompting (Start Simple)

Few-shot prompting means giving the model a few input/output demonstrations before the new input.

**Reference:**

- [Few-shot prompting guide](https://api.python.langchain.com/en/latest/core/prompts/langchain_core.prompts.few_shot.FewShotPromptTemplate.html)

### Define Examples

First, we define a set of examples that demonstrate the desired input-output pattern for stakeholder update emails.

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

style_examples = [
    {
        "situation": "Customer asks for a concise update about a project delay.",
        "tone": "transparent",
        "message": "We are two days behind due to a dependency issue; revised release is Thursday with daily status updates.",
    },
    {
        "situation": "Executive team requests a one-line status for a stable rollout.",
        "tone": "executive",
        "message": "Rollout remains on schedule with no critical risks this week.",
    },
    {
        "situation": "Stakeholders want clarification after a failed production deploy.",
        "tone": "accountable",
        "message": "Today's deploy failed due to a migration bug; rollback is complete and a patched release is planned for tomorrow.",
    },
]

style_examples
```

### Build and Run Few-Shot Prompt

The `FewShotPromptTemplate` automatically injects the examples into the prompt before the actual query.

```python
example_prompt = PromptTemplate.from_template(
    "Situation: {situation}\nTone: {tone}\nMessage: {message}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=style_examples,
    example_prompt=example_prompt,
    prefix="You write short stakeholder updates. Follow the style and structure of the examples.",
    suffix="Situation: {situation}\nTone: {tone}\nMessage:",
    input_variables=["situation", "tone"],
)

query_situation = "Leadership asks for a short note about a delayed analytics dashboard release."
query_tone = "transparent"

few_shot_chain = few_shot_prompt | llm | StrOutputParser()
few_shot_answer = few_shot_chain.invoke({"situation": query_situation, "tone": query_tone})
print(few_shot_answer)
```

### Compare Zero-Shot vs Few-Shot

Compare the output quality between zero-shot (no examples) and few-shot (with examples) approaches.

```python
zero_shot_prompt = PromptTemplate.from_template(
    "You write short stakeholder updates.\nSituation: {situation}\nTone: {tone}\nMessage:"
)
zero_shot_chain = zero_shot_prompt | llm | StrOutputParser()
zero_shot_answer = zero_shot_chain.invoke({"situation": query_situation, "tone": query_tone})

print("Zero-shot:\n", zero_shot_answer)
print("\nFew-shot (fixed examples):\n", few_shot_answer)
```

### Inspect the Rendered Prompt

View exactly what prompt is sent to the model with the injected examples.

```python
print("Rendered few-shot prompt:\n")
print(few_shot_prompt.format(situation=query_situation, tone=query_tone))
```

> **Try it:**
> - Add one more fixed example that matches your course domain.
> - Remove one example and observe how the output changes.
> - Change the example tone labels and see whether the model follows the new style.

---

## 4. Similarity-Based Example Selection

Fixed few-shot examples are a good start, but they do not adapt to different user inputs. Similarity-based selection keeps a larger pool of examples and picks the most relevant ones for each request.

**Why this matters:**

- Better example relevance for diverse inputs
- Less prompt clutter than pasting all examples
- Cleaner scaling from small demos to larger example sets

**Note:** For classroom stability, this notebook uses embedding similarity in pure Python (no FAISS native dependency).

**References:**

- [SemanticSimilarityExampleSelector](https://reference.langchain.com/python/langchain-core/example_selectors/semantic_similarity/SemanticSimilarityExampleSelector)
- [Example selectors](https://reference.langchain.com/python/langchain-core/example_selectors/length_based/LengthBasedExampleSelector)

### Compute Embeddings and Select Similar Examples

```python
import math
from langchain_openai import OpenAIEmbeddings

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is required for similarity-based example selection.")

embedding_model = OpenAIEmbeddings(
    model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
)

example_vectors = embedding_model.embed_documents([ex["situation"] for ex in style_examples])


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-12)


def select_similar_examples(query: str, k: int = 2):
    query_vec = embedding_model.embed_query(query)
    scored = [
        (cosine_similarity(query_vec, vec), ex)
        for vec, ex in zip(example_vectors, style_examples)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in scored[:k]]


selected_examples = select_similar_examples(
    "Need a stakeholder update about a delayed analytics release with clear accountability.",
    k=2,
)

for i, ex in enumerate(selected_examples, start=1):
    print(f"Selected {i}: {ex['situation']}")
```

### Use Similarity-Selected Examples in Prompt

```python
similarity_few_shot_prompt = FewShotPromptTemplate(
    examples=selected_examples,
    example_prompt=example_prompt,
    prefix="You write short stakeholder updates. Follow the style and structure of the examples.",
    suffix="Situation: {situation}\nTone: {tone}\nMessage:",
    input_variables=["situation", "tone"],
)

adaptive_answer = (similarity_few_shot_prompt | llm | StrOutputParser()).invoke(
    {
        "situation": "CIO asks for a concise status note on delayed analytics dashboard rollout.",
        "tone": "transparent",
    }
)

print(adaptive_answer)
```

### Compare Fixed vs Similarity-Selected Few-Shot

```python
# Compare fixed few-shot vs similarity-selected few-shot on the same input
comparison_input = {
    "situation": "Leadership requests an update about a delayed release with mitigation steps.",
    "tone": "transparent",
}

fixed_answer = (few_shot_prompt | llm | StrOutputParser()).invoke(comparison_input)

selected_for_comparison = select_similar_examples(comparison_input["situation"], k=2)
comparison_similarity_prompt = FewShotPromptTemplate(
    examples=selected_for_comparison,
    example_prompt=example_prompt,
    prefix="You write short stakeholder updates. Follow the style and structure of the examples.",
    suffix="Situation: {situation}\nTone: {tone}\nMessage:",
    input_variables=["situation", "tone"],
)
similarity_answer = (comparison_similarity_prompt | llm | StrOutputParser()).invoke(comparison_input)

print("Fixed few-shot:\n", fixed_answer)
print("\nSimilarity-selected few-shot:\n", similarity_answer)
```

> **Try it:**
> - Increase `k` from `2` to `3` and compare output quality.
> - Add 5 more examples to `style_examples` and inspect selected demonstrations.
> - Try a very different `situation` and observe how the selected examples change.

---

## 5. Structured Output Workflow

Use a schema to force machine-readable outputs that are easier to validate and reuse in downstream steps.

**Why this matters:**

- Reduces brittle string parsing
- Gives explicit field-level constraints (types, required keys, ranges)
- Makes workflow handoffs reliable (for storage, routing, evaluation, APIs)

In practice, this is one of the highest-leverage improvements for production-grade LLM workflows.

**References:**

- [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [Models and `with_structured_output`](https://docs.langchain.com/oss/python/langchain/models)

### Define a Pydantic Schema

```python
from pydantic import BaseModel, Field


class StudyPlan(BaseModel):
    topic: str = Field(description="Main topic")
    difficulty: str = Field(description="beginner, intermediate, advanced")
    key_concepts: list[str] = Field(description="3-6 key concepts")
    exercise_idea: str = Field(description="One practical exercise")


planner = llm.with_structured_output(StudyPlan)

plan = planner.invoke(
    "Create a short study plan about prompt engineering for master students in data science."
)
plan
```

### Access Structured Fields

```python
print("Topic:", plan.topic)
print("Difficulty:", plan.difficulty)
print("Key concepts:", ", ".join(plan.key_concepts))
print("Exercise idea:", plan.exercise_idea)
```

### Combine Chains with Structured Output

This example generates an article and then reviews it using structured feedback.

```python
class WritingFeedback(BaseModel):
    score: int = Field(ge=1, le=10, description="Overall quality score")
    strengths: list[str]
    improvements: list[str]


feedback_model = llm.with_structured_output(WritingFeedback)

article_prompt = ChatPromptTemplate.from_template(
    "Write a concise explanation of {topic} for {audience} in at most {max_words} words."
)
article_chain = article_prompt | llm | StrOutputParser()


def generate_and_review(topic: str, audience: str = "master students", max_words: int = 120):
    draft = article_chain.invoke(
        {"topic": topic, "audience": audience, "max_words": max_words}
    )
    feedback = feedback_model.invoke(
        f"Evaluate the following text for clarity and usefulness.\n\n{draft}"
    )
    return {"draft": draft, "feedback": feedback}


result = generate_and_review("how runnable workflows work in LangChain")
result
```

> **Try it:**
> - Change the `audience` (for example executives, developers, first-year students).
> - Add another field to `WritingFeedback` (for example `missing_points: list[str]`) and rerun.

---

## 6. Routing Workflow (Classify → Branch)

A router chooses which prompt/chain to run based on the user request.

**Why this matters:**

- Avoids one oversized prompt for all tasks
- Improves quality by selecting specialized branches
- Provides a clean control-flow pattern before moving to larger agent graphs

**References:**

- [RunnableLambda API](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.base.RunnableLambda)
- [Runnable composition](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.base.Runnable)

### Define Router and Route Chains

```python
from typing import Literal
from langchain_core.runnables import RunnableLambda


class RouteDecision(BaseModel):
    route: Literal["summarize", "rewrite", "brainstorm"]
    reason: str


router = llm.with_structured_output(RouteDecision)

summarize_chain = (
    ChatPromptTemplate.from_template("Summarize the following text in 3 bullet points:\n\n{request}")
    | llm
    | StrOutputParser()
)

rewrite_chain = (
    ChatPromptTemplate.from_template("Rewrite the following text to be clearer and more professional:\n\n{request}")
    | llm
    | StrOutputParser()
)

brainstorm_chain = (
    ChatPromptTemplate.from_template("Generate 6 creative ideas for this request:\n\n{request}")
    | llm
    | StrOutputParser()
)

ROUTES = {
    "summarize": summarize_chain,
    "rewrite": rewrite_chain,
    "brainstorm": brainstorm_chain,
}


def route_workflow(request: str):
    decision = router.invoke(
        f"Classify this request into summarize, rewrite, or brainstorm: {request}"
    )
    answer = ROUTES[decision.route].invoke({"request": request})
    return {
        "route": decision.route,
        "reason": decision.reason,
        "answer": answer,
    }


routing_chain = RunnableLambda(route_workflow)

routing_chain.invoke("We launched 4 campaigns this month and need an executive summary.")
```

> **Try it:**
> - Provide one input that should route to each branch (`summarize`, `rewrite`, `brainstorm`).
> - Add a fourth branch called `critique`.

---

## 7. Parallel Workflow (One Input → Multiple Outputs)

Use `RunnableParallel` to produce multiple artifacts from the same context in one call pattern.

**Why this matters:**

- Generates several useful views at once (summary, actions, risks)
- Lowers end-to-end latency for independent sub-tasks
- Encourages modular design where each branch has a clear responsibility

**References:**

- [RunnableParallel API](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.base.RunnableParallel)
- [Runnable interface and batching/streaming](https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.base.Runnable)

### Define Parallel Chains

```python
from langchain_core.runnables import RunnableParallel

meeting_notes = """
Sprint review notes:
- Checkout bug reduced conversion by 8% for two days.
- Team fixed bug and shipped patch.
- Mobile onboarding still has 42% drop-off at step 3.
- New A/B test for pricing page starts next Monday.
- Data team requests clearer event naming conventions.
""".strip()

summary_chain = (
    ChatPromptTemplate.from_template("Summarize these meeting notes in 4 concise bullets:\n\n{notes}")
    | llm
    | StrOutputParser()
)

actions_chain = (
    ChatPromptTemplate.from_template("Extract concrete action items with owners from these notes:\n\n{notes}")
    | llm
    | StrOutputParser()
)

risks_chain = (
    ChatPromptTemplate.from_template("List key risks and mitigation ideas from these notes:\n\n{notes}")
    | llm
    | StrOutputParser()
)

parallel_workflow = RunnableParallel(
    summary=summary_chain,
    action_items=actions_chain,
    risks=risks_chain,
)

parallel_result = parallel_workflow.invoke({"notes": meeting_notes})
parallel_result
```

> **Try it:**
> - Add a fourth parallel branch: `open_questions`.
> - Compare runtime and output quality to sequential execution.

---

## 8. Tool-Calling Workflow

The model decides when to call tools, then you execute tool functions and return tool results back to the model.

**Why this matters:**

- Connects LLM reasoning with deterministic code
- Enables actions and calculations outside the model weights
- Establishes the core loop behind many agent systems

In this section we do the full loop in small steps:

1. Define tools
2. Bind tools to the chat model
3. Let the model propose tool calls
4. Execute the tool calls in Python
5. Send tool results back to the model for the final answer

**References:**

- [Tool calling on models](https://docs.langchain.com/oss/python/langchain/models)
- [Tools](https://docs.langchain.com/oss/python/langchain/tools)

### Step 1: Imports and Tool Definitions

Each tool is a normal Python function with a typed signature and docstring. LangChain turns this into a schema the model can call.

```python
from datetime import datetime, date
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def days_until(target_date: str) -> int:
    """Return days from today until target_date in YYYY-MM-DD format."""
    t = date.fromisoformat(target_date)
    return (t - datetime.now().date()).days
```

### Step 2: Bind Tools to the Model

After binding, the model can decide to return `tool_calls` instead of (or in addition to) plain text.

```python
tools = [add_numbers, days_until]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

print("Tool names:", [t.name for t in tools])
print("add_numbers schema:", add_numbers.args)
print("days_until schema:", days_until.args)
```

### Step 3: Ask a Question That Requires Tools

This first model response often contains structured `tool_calls` instructions.

```python
query = "What is 17.5 + 26.25, and how many days until 2026-05-19?"
ai_msg = llm_with_tools.invoke(query)
ai_msg
```

### Inspect Tool Calls

```python
print("Model text content:", ai_msg.content)
print("\nTool calls requested by model:")
for call in ai_msg.tool_calls:
    print(call)
```

### Step 4: Execute Tool Calls in Python

You execute each requested tool call yourself, collect results, and return them as `ToolMessage` objects.

```python
tool_messages = []

if not ai_msg.tool_calls:
    print("No tool calls were requested by the model.")
else:
    for call in ai_msg.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        tool_result = tools_by_name[tool_name].invoke(tool_args)
        print(f"{tool_name}({tool_args}) -> {tool_result}")

        tool_messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=call["id"])
        )

tool_messages
```

### Step 5: Send Tool Results Back for Final Answer

Now the model has both the original question and the concrete tool outputs, so it can produce the final response.

```python
if tool_messages:
    final_msg = llm_with_tools.invoke([
        HumanMessage(content=query),
        ai_msg,
        *tool_messages,
    ])
    print(final_msg.content)
else:
    print(ai_msg.content)
```

### Optional: Wrap the Loop in a Reusable Function

Once the flow is clear, package it into a helper so you can reuse it in larger workflows.

```python
def run_tool_workflow(query: str):
    ai_msg = llm_with_tools.invoke(query)

    if not ai_msg.tool_calls:
        return {"tool_calls": [], "final_answer": ai_msg.content}

    tool_messages = []
    for call in ai_msg.tool_calls:
        tool_result = tools_by_name[call["name"]].invoke(call["args"])
        tool_messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=call["id"])
        )

    final_msg = llm_with_tools.invoke([
        HumanMessage(content=query),
        ai_msg,
        *tool_messages,
    ])

    return {
        "tool_calls": ai_msg.tool_calls,
        "final_answer": final_msg.content,
    }


run_tool_workflow("Add 2.5 and 9.75. Also how many days until 2026-04-14?")
```

> **Try it:**
> - Add one more tool (for example, `multiply_numbers` or `weekday_of_date`).
> - Ask a query that needs multiple tools in one response.
> - Add error handling for invalid date formats to make the workflow more robust.

---

## 9. Mini Challenge

Build a small end-to-end workflow:

1. Route a request (`summarize` / `rewrite` / `brainstorm`)
2. Generate a draft
3. Score quality with structured output
4. If score < 7, revise once automatically

**Design goals:**

- Keep each step inspectable
- Keep outputs machine-readable where possible
- Keep failure handling explicit (for example: low score → automatic revision)

### Starter Template

```python
# Starter template for the mini challenge

class QualityCheck(BaseModel):
    score: int = Field(ge=1, le=10)
    comment: str


quality_model = llm.with_structured_output(QualityCheck)


def mini_workflow(user_request: str):
    # 1) Route
    routed = route_workflow(user_request)

    # 2) Draft
    draft = routed["answer"]

    # 3) Evaluate
    quality = quality_model.invoke(
        f"Score this draft from 1-10 for usefulness and clarity, then comment briefly.\n\n{draft}"
    )

    # 4) Optional revise
    if quality.score < 7:
        revision_prompt = ChatPromptTemplate.from_template(
            "Improve this draft using the feedback below.\n\nDraft:\n{draft}\n\nFeedback:\n{feedback}"
        )
        revised = (revision_prompt | llm | StrOutputParser()).invoke(
            {"draft": draft, "feedback": quality.comment}
        )
    else:
        revised = draft

    return {
        "route": routed["route"],
        "initial_score": quality.score,
        "initial_comment": quality.comment,
        "final_output": revised,
    }


mini_workflow("We need a concise update for stakeholders about this sprint delay.")
```

---

## Wrap-up

You now implemented seven common workflow patterns in LangChain:

- Prompt chains
- Few-shot prompting (fixed examples)
- Structured outputs
- Similarity-based example selection
- Routing
- Parallel processing
- Tool calling

These patterns are reusable building blocks for larger agent systems and graph-based orchestration.

**Further reading:**

- [Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [LangGraph](https://docs.langchain.com/oss/python/langgraph)
