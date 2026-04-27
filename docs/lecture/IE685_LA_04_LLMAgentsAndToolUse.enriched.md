---
document_type: enriched-markdown
source_pdf: IE685_LA_04_LLMAgentsAndToolUse.pdf
page_count: 78
image_count: 18
extraction_tool: enrich_pdf_local.py
---

# LLM Agents and Tool Use

**IE685 Large Language Models and Agents**

---

# Outline

1. **Tool Usage for LLMs**
   - Why do LLMs need Tools?
   - Teaching Tool Use
   - Tool Calling Workflow
   - Model Context Protocol

2. **LLMs as Agents**
   - What is an Agent?
   - The ReAct Paradigm

3. **Agentic Harnesses for LLM Agents**

4. **Evaluating LLM Agents**

---

# Example Task: Question Answering

- Can a post-trained LLM do these tasks out-of-the-box?

---

# Supporting LLMs with Tools

- How did humanity develop over time to where we are today?
- An important factor: Usage of Tools
  - Spears, the plow, electricity, computers, …
  - Today we have many complex tools to help us solve problems, e.g. calculators, search engines, …

<!-- figure:page-004-img-002 p4 -->
![Human evolution illustration showing transition from hunter-gatherers to modern office workers](IE685_LA_04_LLMAgentsAndToolUse.assets/images/page-004-img-002-xref-37.jpeg)

> A satirical illustration of human evolution that critiques modern sedentary lifestyles by showing the transition from upright hunter-gatherers back to a hunched posture at a computer.

- Primate (knuckle-walking)
- Early hominid (stooped)
- Early human (upright with stone tool)
- Hunter (with spear)
- Agrarian (with rake/pitchfork)
- Industrial worker (with power tool)
- Modern office worker (seated at computer)
<!-- /figure:page-004-img-002 -->

Mialon, G., et al. 2023, Augmented Language Models: a Survey. *Transactions on Machine Learning Research*.

---

# Example: Code Generation for Computational Problems

- Leverages external tool (python interpreter) to decouple computation from reasoning
- LLM can make calls to the interpreter to run generated code

Chen, W. et al., 2023 Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks. *Transactions on Machine Learning Research*.

---

# Example: Retrieval-augmented Generation for Knowledge Problems

- Knowledge is constrained to pre-training cutoff date
- LLMs have limited context-windows
- Requires answering knowledge-intensive questions with:
  - Extra corpora (e.g. databases, the Web)
  - A retriever (e.g. BM25, DensePassageRetrieval, etc.)
- More about RAG next week!

---

# Teaching LLMs to use Tools

- Add special tokens to invoke tool calls for:
  - Search engines, calculators, etc.
  - Task-specific models (translation)
  - APIs
- Unnatural format requires task/tool-specific fine-tuning

Parisi, A., et al., 2022. Talm: Tool augmented language models. *arXiv preprint arXiv:2205.12255*.

Schick, T., et al., 2024. Toolformer: Language models can teach themselves to use tools. *Advances in Neural Information Processing Systems*, *36*.

---

# Tool Learning: Tutorial

- **Tutorial Learning**
  - Have model tuned for tool use read tool manuals (tutorials), so that it understands the functions of the tool and how to invoke them
  - Works with powerful LLMs

---

# Tool Learning Prompt

*(Visual content - see original PDF)*

---

# Tool Learning: Self-supervised

- **Self-supervised Tool Learning**
  - Pre-defined tool APIs
  - Encourage models to call and execute tool APIs
  - Design self-supervised loss to evaluate tool execution helpfulness

Schick, T et al., 2024. Toolformer: Language models can teach themselves to use tools. *Advances in Neural Information Processing Systems*, *36*.

---

# Tool Learning: RL

- **Reinforcement Learning**
  - Autonomous exploration and correction of errors based on environmental feedback through reinforcement learning
  - Action space defined by tools
  - Agent learns to select appropriate tool
  - Correct action maximize reward signal

---

# Early Example: WebGPT

- Supervised Learning performed at OpenAI:
  - Trying to copy human behavior to use search engines
  - Supervised fine-tuning + reinforcement learning
  - Only 6000 annotated data instances

Nakano, R., et al., 2021. WebGPT: Browser-assisted question-answering with human feedback. *arXiv preprint arXiv:2112.09332*.

---

# Early Example: WebGPT (continued)

- Excellent performance in long-form QA, even surpassing human experts sometimes

---

# Example: How to Define and Use Tools

- Tools are defined using JSON Schema with three key fields:
  - **name**: unique identifier (e.g., "get_current_weather")
  - **description**: what the tool does (used by LLM to decide when to call it)
  - **parameters**: JSON Schema object with input types and required fields

- Example (OpenAI function calling format):

```json
{
  "type": "function",
  "function": {
    "name": "get_current_weather",
    "description": "Get current weather in a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string" }
      },
      "required": ["location"]
    }
  }
}
```

- Alternative: Python functions with type hints + docstrings (auto-converted to schema)

---

# Tool Usage: General Process

*(Visual content - see original PDF)*

---

# Function Calling Steps

## 1. Model (ex. ChatGPT) is called using a query and a set of functions

Functions are formulated as dictionaries that have the key parameters: name, description and its parameters.

```json
{
  "name": "get_current_weather",
  "description": "Get the current weather in a given location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and state, e.g. San Francisco, CA"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"]
  }
}
```

Source: https://openai.com/blog/function-calling-and-other-api-updates

---

## 2. Model decides if it will call one of the functions passed to it

If it decides to call one, it will return as finish reason a function call and will give the parameters with which to call the function.

```json
{
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "function_call": {
        "name": "get_current_weather",
        "arguments": "{\"location\": \"Mannheim\"}"
      }
    },
    "finish_reason": "function_call"
  }]
}
```

Source: https://openai.com/blog/function-calling-and-other-api-updates

---

## 3. The function can be called by the client with the parameters returned from the model

```json
{
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "function_call": {
        "name": "get_current_weather",
        "arguments": "{\"location\": \"Mannheim\"}"
      }
    },
    "finish_reason": "function_call"
  }]
}
```

**get_current_weather(location:string, unit:"Celsius"|"Fahrenheit")** — Function that queries an external weather API

---

## 4. The function result together with the question are passed to the model again to summarize the result

User: "What is the weather currently in Mannheim?"

Function returns: `{ "temperature": 25, "unit": "celsius" }`

Model response: **"The temperature in Mannheim is 25 degrees Celsius."**

Source: https://openai.com/blog/function-calling-and-other-api-updates

---

# Example: Tool usage with OpenAI

**API Request - tools parameter in chat completions:**

```json
{
  "messages": [{"role": "user", "content": "What is the weather in Mannheim?"}],
  "tools": [{"type": "function", "function": {"name": "get_current_weather", ...}}]
}
```

**LLM Response - tool_calls in assistant message:**

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "function": {
      "name": "get_current_weather",
      "arguments": "{\"location\":\"Mannheim\"}"
    }
  }]
}
```

**Key points:**

- The LLM decides IF and WHICH tool to call based on the query and tool descriptions
- "tool_calls" signals the client to execute the function
- After execution, result is sent back as a "tool" role message for the final response

Source: https://openai.com/blog/function-calling-and-other-api-updates

---

# Evaluating Tool Usage

- **Berkeley Function-Calling Leaderboard (BFCL):**
  - Standard benchmark for evaluating LLM function calling
  - Tests: simple calls, multiple calls, parallel calls, relevance detection

- **Key evaluation metrics:**
  - **Invocation Accuracy**: Does the model correctly decide to call a function?
  - **Tool Selection Accuracy**: Does it pick the right function from a set?
  - **Parameter Name/Value F1**: Are the arguments correct and complete?
  - **AST Correctness**: Is the generated function call syntactically valid?

- Other benchmarks: API-Bank, ToolBench, Nexus Raven, Seal-Tools, Mannheim Function Calling Benchmark

---

# Model Context Protocol (MCP)

- **Problem:** N LLMs × M tools = N×M custom integrations
- **MCP:** A standard protocol connecting any LLM to any tool
  - Write tool integration once, use with any MCP-compatible LLM host

Anthropic, 2024. Introducing the Model Context Protocol. https://modelcontextprotocol.io

---

# Example: Model Context Protocol

**Benefits:**

- **Standardized:** build once, use with any MCP-compatible host
- **Growing ecosystem:** 1000+ community-built MCP servers
- **Security:** server-side authorization, controlled tool access

Source: Anthropic

---

# Public MCP Servers

- **Official MCP Registry:** https://registry.modelcontextprotocol.io/

- **Categories of public MCP servers and examples:**
  - File Systems: local files, Google Drive, Dropbox
  - Databases: PostgreSQL, SQLite, MongoDB
  - Web & Search: Brave Search, Google Search, web scraping
  - Developer Tools: GitHub, Git, Docker, Kubernetes
  - Productivity: Slack, Google Calendar, Notion, Linear
  - Data & Knowledge: persistent memory, knowledge graphs
  - AI & ML: Hugging Face models, image generation, embeddings

- Anyone can build and publish an MCP server
  - SDKs available in Python, TypeScript, Java, Kotlin, C#

---

# Excursion: Current Reasoning LLMs

- **Reasoning:** teaching models to explain step-by-step reasoning before answering
  - Chain of Thought at massive scale, enabled during post-training

- Tool use is a **key capability** to enable reasoning:
  - LLMs alone have limited reasoning, static knowledge, and cannot perform actions
  - Reasoning models can (learn to) decide when and how to use tools to overcome these limitations

---

# Reasoning LLMs

- **Example: DeepSeek R1 Training Recipe (5-step pipeline):**

  1. Pretrain base model
  2. Small-scale SFT with curated reasoning data (cold start)
  3. RL with reasoning-focused data (RL stage 1)
  4. Large-scale SFT mixing ~600k reasoning + ~200k general data
  5. Final RL with mixed rewards: accuracy + helpfulness + harmlessness (RL stage 2)

- **Test-time (inference) compute scaling (thinking tokens):**
  - Dynamic budget: allocate more compute to harder problems
  - Context awareness: model adapts reasoning depth to problem complexity
  - Budget forcing: control maximum reasoning tokens (cost vs. quality trade-off)

---

# Reasoning LLMs: A New RL Paradigm

- **Reinforcement Learning with Verifiable Rewards (RLVR)**
  - Core idea: Train reasoning without human-annotated reasoning chains
  - Use domains where correctness can be automatically verified
  - Usually placed between Instruction Tuning and RLHF
  - Gained momentum with DeepSeek R1 release (RL stage 1)

- **Two reward signals:**
  1. **Format reward**: verify the model uses `<think>…` template for reasoning
  2. **Accuracy reward**: verify final answer via e.g. code execution or math ground truth (automatically verifiable)

- **Advantages:** scales without expensive human annotation of reasoning chains, automatically trains tool use

---

# What is an Agent?

- LLM-powered Agents are artificial entities that enhance LLMs with essential capabilities enabling them to sense their environment, make decisions, and take actions.

---

# What is an Agent? (continued)

- An "intelligent" system that interacts with some "environment"
  - Physical environments: robot, autonomous car, …
  - Digital environments: DQN for Atari, Siri, AlphaGo
  - Humans as environment: Chatbots

---

# LLM Agents vs. LLM Workflows

- **LLM Workflow:** Fixed code execution flow
- **LLM Agent(s):** Autonomous decision making

*(Visual diagram - see original PDF)*

---

# LLM Agents

*(Visual content - see original PDF)*

---

# A Brief History of LLM Agents

Wang, L., et al., 2024. A survey on large language model based autonomous agents. *Frontiers of Computer Science*, *18* (6), p.186345.

---

# Reasoning OR Acting

*(Visual content - see original PDF)*

---

# The ReAct Paradigm

Yao, S., et al., 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In *The Eleventh International Conference on Learning Representations*.

---

# ReAct is Simple and Intuitive to Use

*(Visual content - see original PDF)*

---

# Zero-shot ReAct Prompt

*(Visual content - see original PDF)*

---

# Zero-shot ReAct Prompt (continued)

- **Synergy:**
  - Acting supports reasoning
  - Reasoning guides acting

---

# Converting Tasks to Text

- Many tasks can be turned into natural language for LLM agents
- "LLM grounding": Supplementing the LLM with use-case specific information, e.g. a data store that is part of a RAG system

Brohan, A., et al., 2023, March. Do as i can, not as i say: Grounding language in robotic affordances. In *Conference on robot learning* (pp. 287-318). PMLR.

Huang, W., et al., 2023, March. Inner Monologue: Embodied Reasoning through Planning with Language Models. In *Conference on Robot Learning* (pp. 1769-1782). PMLR.

---

# Acting without Reasoning

- Cannot explore systematically or incorporate feedback

---

# ReAct Enables Systematic Exploration

*(Visual content - see original PDF)*

---

# ReAct is General and Effective

Yao, S., et al., 2023, ReAct: Synergizing Reasoning and Acting in Language Models. In *The Eleventh International Conference on Learning Representations*.

---

# Example: Human/Agent/Environment Interaction

- High-Level Workflow of Coding Agent

Source: https://www.anthropic.com/engineering/building-effective-agents

---

# Example: Prompt of Web Browsing Agents (abbreviated)

**SYSTEM:** You are an agent trying to solve a web task based on the content of the page and user instructions. You can interact with the page and send messages to the user. Each time you submit an action it will be sent to the browser and you will receive a new page.

**USER:**

**## Goal:** Find the cheapest offer for an Iphone 13.

**## Observation of current step:**

## Currently open tabs: Tab 0 (active tab): Title: WebMall URL: http://localhost:8085/

### AXTree: …

### **Concrete Example**

Here is a concrete example of how to format your answer.

**Action space:** 15 different types of actions are available.

```xml
<action> click('a324') </action>
```

Agent reasoning:
```

From previous action I tried to set the value of year to "2022", using select_option, but it doesn't appear to be in the form. It may be a dynamic dropdown, I will try using click with the bid "a324" and look at the response from the page.


Available actions include: `noop(wait_ms: float)`, `scroll(delta_x: float, delta_y: float)`, `keyboard_press(key: str)`, `click(bid: str, button: Literal['left', 'middle', 'right'])`, `fill(bid: str, value: str)`, `hover(bid: str)`, `tab_focus(index: int)`, `new_tab()`, etc.

---

# Example: Trajectory of a Web Agent

- Web agent trajectories are extremely long and hard to debug:
  - Each step usually includes: full page observation (AXTree), reasoning (<think>), and action as well as action history
  - A single shopping task can produce 50+ steps

- **Example trajectory (abbreviated):**
  - Step 1: [OBS: AXTree ~500 lines][History] <think>I need to search… → click("search")
  - Step 2: [OBS: search results ~300 lines][History] <think>Found items… → click("item_3")
  - Step 3: [OBS: product page ~400 lines][History] <think>Checking price… → fill("qty","1")
  - … (dozens more steps with full page observations each and growing history) …
  - Step N: [OBS: confirmation] <think>Done → send_msg_to_user("Task complete")

- **Challenges:** massive token cost, error propagation across steps, hard to pinpoint failures

---

# Trajectory Tracking with LangSmith

- LangSmith by LangChain: observability platform for LLM applications
  - **Tracing:** visualize each step of agent execution (input, output, latency, tokens)
  - **Debugging:** inspect individual reasoning steps and tool calls
  - **Evaluation:** run test suites, compare model versions, track regressions

- **How it helps with agent trajectories:**
  - Collapsible tree of multi-step agent runs
  - Token usage and latency metrics per step
  - Filter and search within long trajectories

---

# Agentic Harnesses for LLM Agents

- **What is an Agentic Harness?**
  - The software infrastructure that wraps an LLM to enable agentic behavior

- **Components of an agentic harness:**
  - **Tool integration:** connecting the LLM to external APIs, databases, code execution
  - **Memory systems:** short-term (context window) and long-term (vector stores, databases)
  - **Control loops:** ReAct, retry logic, error handling
  - **Orchestration:** managing multi-step workflows and multi-agent collaboration

- The LLM is the **"brain"**, the harness provides the **"body"**

---

# Agentic Harnesses for LLM Agents (continued)

- The agentic harness allows the LLM to perceive its environment through observations

---

# Observation and Action

- The harness manages the observation-action loop and routes data between components

---

# The "Brain"

- The LLM serves as the "brain" within the agentic harness
  - The harness delegates reasoning and decision-making to the LLM
  - The harness provides context (observations, memory, tools)

---

# The "Brain" (continued)

- **Memory:** stores sequences of agent's past observations, thoughts and actions
  - Long-term and short-term memory
  - Long-term memory is often abstract
  - Used to retrieve relevant past memory
  - Is a **component of the harness**

- **Decision Making Process:**
  - **Planning:** Subgoal and decomposition – Break down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks
  - **Reasoning:** Self-criticism and self-reflection over past actions, learn from mistakes and refine for future steps

- Is **orchestrated by the harness**

---

# Collaboration

- **Multi-agent harnesses:** diverse agents interact to solve problems
- **Human-in-the-loop harnesses:** cooperative systems with human oversight

---

# How can Agents Communicate?

*(Visual content - see original PDF)*

---

# Standard: A2A Protocol (Agent2Agent)

- Agents advertise their capabilities using an "Agent Card" in JSON format
- Allows the client agent to identify the best agent that can perform a task and leverage A2A to communicate

Source: https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/

---

# Standard: A2A Protocol (continued)

*(Visual content - see original PDF)*

---

# Multi-Agent Orchestration

- Usually a "Manager" or "Commander" for orchestrating many agents
- Context may be shared or isolated
- Cooperative vs. competitive environments
- Centralized vs. decentralized communication
- Human intervention vs. full automation

---

# Example: Multi-Agent Coding

- **Commander** receives user questions and executes code
- **Writer** writes code
- **Safeguard** ensures no information leakage or malicious code

Wu, Q., et al., 2024. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. In *ICLR 2024 Workshop on Large Language Model (LLM) Agents*.

---

# Example: Magentic-One

- Multi-Agent system consisting of an orchestrator and various task specialists that have access to tools.
- Orchestrator keeps a task and progress ledger to track progress and adapt if things do not work as planned

Source: https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/

---

# Agentic Harness Architecture

- The harness integrates all components into a unified architecture:
  - **LLM Core:** reasoning, planning, language understanding
  - **Tool Layer:** API connectors, code execution, web browsing
  - **Memory Layer:** short-term context + long-term retrieval
  - **Control Loop:** ReAct cycle, error handling, termination conditions

---

# Summary: LLM Agents

- Current hot topic in research and application
- Combination of tool use and reasoning allows enhancement of LLM abilities while mitigating problematic behavior like hallucinations
- Reasoning Agents
- Orchestrating agents with different capabilities (specializations) allows to solve complex problems

**For more application examples, see the following surveys:**

Guo, T., et al., 2024. Large language model based multi-agents: A survey of progress and challenges. *arXiv preprint arXiv:2402.01680*.

Liu, J., et al., 2024. Large Language Model-Based Agents for Software Engineering: A Survey. *arXiv preprint arXiv:2409.02977*.

---

# Evaluating (Multi-)Agent Systems

- LLM-powered agents enable a rich set of capabilities but also amplify potential risks
  - How to evaluate agent performance and awareness of safety risks?
    - **Potential Risks:** leaking private data or causing financial loss
    - Identifying these risks is **labor-intensive** as testing becomes difficult with increased agent complexity

- Benchmarks for Agents need to cover a broad space including:
  - Tools
  - External resources
  - Correct **behavioral traces or labels**

---

# Example: Risks

*(Visual content - see original PDF)*

---

# Example: AgentBench

- Simulate interactive environments for LLMs to operate as autonomous agents
- 8 distinct environments of 3 types (Coding, Games, Web)
- Evaluation of agent core abilities like logical reasoning

Liu, X., et al., 2024. AgentBench: Evaluating LLMs as Agents. In *The Twelfth International Conference on Learning Representations*.

---

# Example: ToolEMU

- **Goal:** Identify risky behavior of agents
- **Emulates tool execution** and enables scalable testing of agents

Ruan, Y., et al., 2024. Identifying the Risks of LM Agents with an LM-Emulated Sandbox. In *The Twelfth International Conference on Learning Representations*.

---

# Example: WebShop

- Large-scale complex environment based on 1.16M Amazon products
- Challenges language and visual understanding and decision-making

Yao, S., et al., 2022. Webshop: Towards scalable real-world web interaction with grounded language agents. *Advances in Neural Information Processing Systems*, *35*, pp.20744-20757.

---

# Example: WebMall

- Benchmark for evaluating the ability of agents to find and compare offers from multiple shops
- Simulates an online shopping environment consisting of four heterogeneous online shops
- Defines basic and advanced tasks:
  - **Basic tasks:** comparing offers with concrete requirements, adding offers to the shopping cart
  - **Advanced tasks:** searches with vague requirements, searches for compatible and substitute products

- The shops offer different interfaces:
  - HTML pages, MCP APIs, NLWeb interface

**Sources:**
- https://wbsg-uni-mannheim.github.io/WebMall/
- https://webmall-1.informatik.uni-mannheim.de/ (Can be accessed with VPN turned on)

---

# Example: WebArena

- Simulate web environment with high similarity to real-world popular websites
- Embeds tools and knowledge resources as independent websites
- Benchmark for concrete web-based actions

Zhou, S., et al., 2024. WebArena: A Realistic Web Environment for Building Autonomous Agents. In *The Twelfth International Conference on Learning Representations.*

---

# More Benchmarks…

Mahmoud Mohammadi, Yipeng Li, Jane Lo, and Wendy Yip. 2025. Evaluation and Benchmarking of LLM Agents: A Survey. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '25).

---

# Example: Outcome-based LLM Agent Evaluation

- WebArena Benchmark

*(Visual content - see original PDF)*

---

# Example: Checkpoint-based LLM Agent Evaluation

- **Outcome-based (previous slide):** only checks if final goal is achieved
- **Checkpoint-based:** verifies intermediate states during task execution and penalizes execution mistakes

- **WorkArena benchmark:**
  - Web agent evaluation benchmark with real enterprise applications
  - Defines expected intermediate states (checkpoints) for each task
  - Evaluates whether the agent passes through required checkpoints

- **Key metrics:**
  - **Progress Rate:** fraction of expected checkpoints reached
  - **Step Success Rate:** fraction of plan steps that execute successfully

- **Advantage:** diagnoses WHERE agents fail, not just IF they fail

---

# How to Evaluate an Agent for Your Own Use Case?

- **Step 1: Define success criteria**
  - What does "good enough" look like? What metrics are best to evaluate that? (accuracy, latency, cost, safety)
  - Guide: platform.claude.com/docs/en/test-and-evaluate/define-success

- **Step 2: Choose evaluation method**
  - Code-based: automated checks based on outcome/checkpoints
  - LLM-as-Judge: use another LLM to evaluate outputs
  - Human evaluation: expert review for (subjective) quality

- **Step 3: Use evaluation tooling**
  - E.g. LangSmith: tracing + evaluation + dataset management

- **Step 4: Iterate**
  - Continuously evaluate and improve based on results
  - Practical guide: hamel.dev/blog/posts/evals/

---

# See You Next Week!

- Next week: Retrieval-Augmented Generation (RAG)
  - Architecture
  - Workflows
  - Evaluation

*(Visual content - see original PDF)*

---

# Credits

- This slide set is based on slides from:
  - Shunyu Yao
  - Yankai Lin
  - Yang Deng, An Zhang et al.
  - Afshine & Shervine Amidi

- Many thanks to all of you!