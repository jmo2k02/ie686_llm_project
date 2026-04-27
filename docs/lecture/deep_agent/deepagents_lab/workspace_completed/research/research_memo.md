# IE686 research memo: secretary agent using OpenClaw

## comparable_systems

### Office/workplace agent benchmarks and systems

- **WorkBench: a Benchmark Dataset for Agents in a Realistic Workplace Setting** — Olly Styles, Sam Miller, Patricio Cerda-Mardini, Tanaya Guha, Victor Sanchez, Bertie Vidgen / MindsDB, University of Glasgow, University of Warwick — **2024**  
  Relevance: closest public benchmark to “secretary agent” office workflows. Provides a sandbox workplace with **5 databases, 26 tools, and 690 tasks** covering email, calendar, CRM, analytics, project management, and multi-domain workflows. Uses **outcome-centric evaluation** based on final state changes rather than exact action traces. Strong evidence for evaluation design and task realism.  
  URL: https://arxiv.org/abs/2405.00823  
  Notes: especially relevant because tasks include reviewing a calendar before sending an email; reported best agent performance was still far from perfect.

- **OfficeBench: Benchmarking Language Agents across Multiple Applications for Office Automation** — Zilong Wang, Yuedong Cui, Li Zhong, Zimin Zhang, Da Yin, Bill Yuchen Lin, Jingbo Shang / UC San Diego, Amazon, University of Notre Dame (per paper author list) — **2024**  
  Relevance: highly aligned with office automation. The paper states it builds **300 tasks** simulating office automation with **documents, emails, and calendar events**, with customized evaluation including **exact match, fuzzy match, and execution-based evaluation**. Useful as a direct comparison point for multi-app office agents.  
  URL: https://arxiv.org/abs/2407.19056  
  Notes: fetch from PDF was weak/noisy, so details should be treated as coming from search snippets plus paper abstract metadata unless verified from the full text later.

- **OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments** — Tianbao Xie et al. / xlang-ai, Carnegie Mellon, Salesforce Research, Stanford, et al. — **2024**  
  Relevance: major benchmark for computer-use agents in real OS environments across Ubuntu, Windows, and macOS. Includes **369 real-world tasks** over web/desktop apps, file I/O, and cross-application workflows, with **execution-based evaluation** and human/model comparisons. Valuable for positioning the project as a computer-use agent rather than only a tool-calling chatbot.  
  URL: https://arxiv.org/abs/2404.07972  
  Notes: not office-specific, but strong for methodology and realism arguments.

- **WebArena: A Realistic Web Environment for Building Autonomous Agents** — Shuyan Zhou et al. / Carnegie Mellon University — **2023** (ICLR 2024 paper)  
  Relevance: foundational benchmark for realistic, reproducible web agents. Includes long-horizon web tasks and evaluates **functional correctness of task completion** rather than brittle action matching. Good precedent for sandboxed but realistic end-to-end evaluation.  
  URL: https://webarena.dev/  
  Paper URL: https://arxiv.org/abs/2307.13854  
  Notes: web-focused, not email/calendar-specific, but useful for environment design and success metrics.

- **OpenEnv / Calendar Gym** — Meta + Hugging Face framework; TuringEnterprises calendar environment — **2026 blog post; too recent for 2025 proposal evidence**  
  Relevance: describes evaluation against real environments with a **production-grade calendar management environment** using real constraints such as ACLs, partial visibility, stateful multi-step workflows, and failure recovery. Very conceptually relevant for calendar coordination evaluation.  
  URL: https://huggingface.co/blog/openenv-turing  
  Weak-evidence flag: this is a **2026 engineering blog**, not 2023–2025 scientific writing. Use only as optional inspiration, not core citation for the proposal.

### Recent scientific writing on assistants / scheduling / personal-agent setups

- **NATURAL PLAN: Benchmarking LLMs on Natural Language Planning** — Huaixiu Steven Zheng, Swaroop Mishra, Hugh Zhang, Xinyun Chen, Minmin Chen, Azade Nova, Le Hou, Heng-Tze Cheng, Quoc V. Le, Ed H. Chi, Denny Zhou / Google — **2024**  
  Relevance: directly includes **Meeting Planning** and **Calendar Scheduling** tasks. Useful if the project separates planning quality from tool-execution quality. The benchmark is text-based rather than API-executing, so it is a good planning sub-benchmark but not enough by itself for secretary-agent evaluation.  
  URL: https://arxiv.org/abs/2406.04520

- **Auto-SLURP: A Benchmark Dataset for Evaluating Multi-Agent Frameworks in Smart Personal Assistant** — Lei Shen, Xiaoyu Shen — **2025**  
  Relevance: personal-assistant framing is relevant. Extends SLURP with relabeling plus simulated servers/external services to support end-to-end evaluation of language understanding, execution, and response generation in a smart assistant setup.  
  URL: https://arxiv.org/abs/2504.18373  
  Notes: assistant-oriented, but not specifically office productivity.

- **ScheduleMe: Multi-Agent Calendar Assistant** — authors not fully verified from excerpt / PACLIC 2025 — **2025**  
  Relevance: directly about a multi-agent calendar assistant and references **SmartCal (Shen et al., 2024)** as improving tool-use reliability through a self-aware supervisory framework. Potentially useful for related-systems section.  
  URL: https://aclanthology.org/2025.paclic-1.27.pdf  
  Weak-evidence flag: only excerpt inspected; author list and exact claims should be rechecked from full paper before citing in final proposal.

### Tool-use / API-use benchmarks relevant for secretary-agent internals

- **API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs** — Yongchao Li et al. — **2023**  
  Relevance: broad benchmark for planning, retrieving, and invoking APIs. Paper reports **1,008 domains, 2,211 APIs, 2,202 dialogues, and 6,135 turns**. Useful for benchmarking tool-use competence beneath the full office workflow level.  
  URL: https://arxiv.org/abs/2304.08244  
  ACL URL: https://aclanthology.org/2023.emnlp-main.187/

- **Berkeley Function Calling Leaderboard (BFCL)** — Shishir G. Patil et al. / UC Berkeley — **2025**  
  Relevance: practical benchmark/leaderboard for function-calling accuracy with multi-turn and agentic evaluation variants. Good for measuring the model’s low-level tool-call reliability before full workflow evaluation.  
  URL: https://gorilla.cs.berkeley.edu/leaderboard.html

- **Agent-Diff: Benchmarking LLM Agents on Enterprise API Tasks via Code Execution with State-Diff-Based Evaluation** — Hubert M. Pysklo, Artem Zhuravel, Patrick D. Watson — **2026 preprint**  
  Relevance: extremely close methodologically: enterprise APIs (Slack, Box, Linear, Google Calendar), sandboxed execution, and **state-diff-based evaluation**. Good inspiration for a future extension.  
  URL: https://arxiv.org/abs/2602.11224  
  Weak-evidence flag: 2026, outside target period.

## candidate_data_and_apis

### Open/official datasets and corpora

- **Enron Email Dataset** — Carnegie Mellon University — **public release maintained through 2015**  
  Relevance: classic large email corpus (**about 500,000 emails from about 150 users**) for inbox triage, email threading, email classification, urgency/response-needed heuristics, and synthetic office-task generation.  
  URL: https://www.cs.cmu.edu/~enron/  
  Caution: old and domain-specific; not a direct inbox-triage benchmark with modern labels. Would likely need relabeling or synthetic task construction.

- **CEREC: entity resolution in email conversations** — Dan Moldovan / COLING 2020 via ACL Anthology listing — **2020**  
  Relevance: useful if meeting prep/follow-up requires grounding people, organizations, and references across email threads.  
  URL: https://aclanthology.org/volumes/2020.coling-main/  
  Weak-evidence flag: only anthology listing/snippet inspected, not full paper page.

- **AMI Meeting Corpus** — AMI project / University of Edinburgh and partners — **official corpus site; mid-2000s collection**  
  Relevance: official meeting corpus for meeting understanding, summarization, action-item extraction, and prep/follow-up tasks. Public download page confirms signals, transcripts, and some annotations are released publicly under CC BY 4.0.  
  URL: https://groups.inf.ed.ac.uk/ami/download/  
  Notes: staged design meetings; good for meeting prep/follow-up but not email/calendar execution.

- **ICSI Meeting Corpus** — International Computer Science Institute / hosted with AMI site — **official corpus site**  
  Relevance: real research meeting data (~70 hours) for summarization, meeting understanding, and follow-up extraction.  
  URL: https://groups.inf.ed.ac.uk/ami/icsi/  
  Notes: more naturalistic than AMI in some respects; still not tied to email/calendar APIs.

- **QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization** — Ming Zhong et al. / Yale LILY — **2021**  
  Relevance: important benchmark for meeting-prep and meeting-follow-up summarization. Repository says **1,808 query-summary pairs over 232 meetings in multiple domains**. Supports query-focused meeting assistance, which fits questions like “what decisions from prior meetings matter for tomorrow’s 1:1?”  
  URL: https://github.com/Yale-LILY/QMSum  
  Paper URL: https://aclanthology.org/2021.naacl-main.472/

- **MeetingBank: A Benchmark Dataset for Meeting Summarization** — Yebowen Hu et al. — **2023**  
  Relevance: newer meeting summarization benchmark with meeting minutes style outputs, potentially closer to business follow-up notes/action items than generic summaries.  
  URL: https://aclanthology.org/2023.acl-long.906/

### Real productivity APIs suitable for prototype + evaluation

- **Gmail API** — Google Workspace / Google Developers — **official**  
  Relevance: inbox triage, search, labels, threads, drafts, send, push notifications. Good candidate for inbox tasks in a prototype secretary agent.  
  URL: https://developers.google.com/workspace/gmail/api/guides

- **Google Calendar API** — Google Workspace / Google Developers — **official**  
  Relevance: create/update events, manage recurring events, invite attendees, synchronize resources, push notifications. Strong fit for scheduling and coordination experiments.  
  URL: https://developers.google.com/workspace/calendar/api/guides/overview

- **Google Tasks API** — Google Workspace / Google Developers — **official**  
  Relevance: explicit task-list backend for follow-up task tracking. Lightweight but practical.  
  URL: https://developers.google.com/workspace/tasks/reference/rest

- **Microsoft Graph Mail API** — Microsoft — **official**  
  Relevance: access Outlook mail, folders, drafts, reply/forward/send, categories, flags, etc. Important enterprise comparison point because many office environments use Microsoft 365.  
  URL: https://learn.microsoft.com/en-us/graph/api/resources/mail-api-overview

- **Microsoft Graph Calendar API** — Microsoft — **official**  
  Relevance: create/update/cancel meetings, accept/decline flows, meeting messages, attendee management, finding workable meeting times. Strong fit for realistic enterprise scheduling workflows.  
  URL: https://learn.microsoft.com/en-us/graph/api/resources/calendar-overview

- **Asana REST API** — Asana — **official**  
  Relevance: task/project tracking backend for post-meeting action items and follow-up workflows.  
  URL: https://developers.asana.com/reference/rest-api-reference

- **Jira Cloud REST API v3** — Atlassian — **official**  
  Relevance: practical enterprise baseline for issue/task tracking; useful if follow-up actions are project tickets rather than simple personal tasks.  
  URL: https://developer.atlassian.com/cloud/jira/platform/rest/v3/intro/

- **Linear GraphQL API** — Linear — **official**  
  Relevance: modern issue-tracking backend; particularly useful because Linear has explicit **Agent Interaction Guidelines**, making it plausible as a clean evaluation or integration target.  
  URL: https://developers.linear.app/docs/graphql/working-with-the-graphql-api

### Realistic data-collection approaches

- **Synthetic but stateful workplace sandbox** modeled after WorkBench / OfficeBench.  
  Rationale: easiest path to reproducible evaluation without exposing real email/calendar data. Could generate employees, meetings, tasks, inboxes, and event histories, then score by state diff/outcome.

- **Opt-in pilot on one user’s Google Workspace or Microsoft 365 sandbox account**.  
  Rationale: gives realistic tool/API behavior, permission issues, and failure modes. Best for formative evaluation, not first benchmark due to privacy and reproducibility issues.

- **Hybrid setup**: real APIs over synthetic accounts/data.  
  Rationale: strongest practical compromise. Use official APIs but populate test tenants with synthetic office workflows and seeded inbox/calendar/task states.

## benchmarks_and_metrics

### Strong benchmark candidates

- **Primary end-to-end benchmark candidate: WorkBench**  
  Why: closest current benchmark to secretary-agent tasks. It already covers sending emails, scheduling meetings, and multi-step business workflows.  
  Main metric: **task success via outcome-centric evaluation** (whether resulting database state matches gold outcome).  
  URL: https://arxiv.org/abs/2405.00823

- **Secondary office benchmark candidate: OfficeBench**  
  Why: more explicitly office-automation-oriented, including emails/calendar/documents and multi-application switching.  
  Main metrics mentioned: **exact match, fuzzy match, execution-based evaluation**.  
  URL: https://arxiv.org/abs/2407.19056

- **Computer-use robustness benchmark: OSWorld**  
  Why: if the project uses browser/desktop actions rather than only direct APIs, OSWorld gives external validity for multimodal computer-use competence.  
  Main metric: **execution-based task success rate**.  
  URL: https://arxiv.org/abs/2404.07972

- **Planning-only sub-benchmark: NATURAL PLAN**  
  Why: isolates planning quality for meeting planning / calendar scheduling without confounding API failures.  
  Main metric: benchmark solve rate/accuracy on planning tasks.  
  URL: https://arxiv.org/abs/2406.04520

- **Tool-call reliability benchmark: BFCL and/or API-Bank**  
  Why: if OpenClaw agent performance is poor, these can help localize whether failures come from tool invocation, planning, or state management.  
  URLs: https://gorilla.cs.berkeley.edu/leaderboard.html ; https://arxiv.org/abs/2304.08244

### Recommended evaluation metrics for realistic office workflows

- **Outcome success rate**: whether the final inbox/calendar/task state matches the intended result. Best core metric for office workflows.
- **Critical error rate**: wrong-recipient email, wrong time/day, wrong attendee set, duplicate meeting, dropped follow-up task. This matters more than average token-level quality.
- **Precision/recall on triage decisions**: e.g., urgent vs defer vs archive vs draft response needed.
- **Scheduling feasibility rate**: percent of proposed meetings satisfying all hard constraints (availability, duration, timezone, attendee requirements).
- **Action-item extraction quality**: precision/recall/F1 for follow-up tasks from meeting/email content.
- **Summary usefulness / factuality** for meeting prep notes: human rubric or benchmark-aligned quality scoring.
- **Latency / steps / tool calls**: operational efficiency matters for secretary-style assistants.
- **Recovery rate after failure**: whether the agent can handle API errors, missing permissions, or conflicting constraints.
- **User-oversight burden**: edits required before execution, approvals required, or human intervention count.

### Good evaluation design choices suggested by the evidence

- Prefer **execution-based or state-diff evaluation** over comparing exact action traces.
- Include **multi-domain tasks**: e.g., read email -> inspect calendar -> draft message -> create follow-up task.
- Measure both **effectiveness and safety**, because workplace mistakes are costly.
- Separate **planning competence** from **tool-execution competence** when possible.

## openclaw_notes

- **OpenClaw — official GitHub repository** — OpenClaw project — **active official source**  
  Relevance: the repo describes OpenClaw as a **personal AI assistant** run on the user’s own devices. It works across many messaging channels (e.g., WhatsApp, Telegram, Slack, Discord, Google Chat, Signal, iMessage, Microsoft Teams, Matrix, etc.). This supports the framing that OpenClaw is a local-first assistant platform rather than a single-task model.  
  URL: https://github.com/openclaw/openclaw

- **OpenClaw Docs: FAQ / Overview / Features** — OpenClaw Docs — **2026 docs snapshot**  
  Relevance: official docs describe OpenClaw as a **personal AI assistant** / “any OS gateway for AI agents,” with key capabilities including **multi-channel gateway**, **multi-agent routing with isolated sessions**, **media support**, **web control UI**, **mobile nodes**, and a persistent workspace-based operating context. These features are relevant if the secretary agent is implemented as a skill/agent inside OpenClaw.  
  URLs:  
  - https://docs.openclaw.ai/help/faq  
  - https://docs.openclaw.ai/  
  - https://docs.openclaw.ai/concepts/features

- **Personal Assistant Setup** — OpenClaw Docs — **official guide**  
  Relevance: explicitly positions OpenClaw as a gateway for a dedicated always-on personal assistant. The guide notes workspace memory files, proactive heartbeats, safety restrictions, and messaging-based operation. This is strong evidence that OpenClaw is intended as an assistant substrate, not just a coding shell.  
  URL: https://docs.openclaw.ai/start/openclaw

- **Introducing OpenClaw** — Peter Steinberger / OpenClaw Blog — **2026**  
  Relevance: official narrative on project purpose: “an open agent platform that runs on your machine and works from the chat apps you already use.” Useful for describing project motivation and local-first/privacy angle.  
  URL: https://openclaw.ai/blog/introducing-openclaw  
  Weak-evidence flag: blog post, not scientific writing.

- **PinchBench** — Kilo AI / PinchBench GitHub — **official benchmark site/repo**  
  Relevance: especially interesting because it states it measures how LLMs perform **as the brain of an OpenClaw agent**, with real tasks such as **scheduling meetings, triaging email, researching topics, and managing files**. This is one of the closest direct external signals that OpenClaw is already used/evaluated in secretary-like workflows.  
  URL: https://github.com/pinchbench/skill  
  Org URL: https://github.com/pinchbench  
  Weak-evidence flag: benchmark repo/site, not peer-reviewed scientific evidence; likely very useful as practical comparison infrastructure.

## recommended_sources

### Highest-value sources for the proposal body

1. **WorkBench: a Benchmark Dataset for Agents in a Realistic Workplace Setting** — MindsDB et al. — 2024  
   Why use: strongest benchmark precedent for office workflows and outcome-based evaluation.  
   URL: https://arxiv.org/abs/2405.00823

2. **OfficeBench: Benchmarking Language Agents across Multiple Applications for Office Automation** — Wang et al. — 2024  
   Why use: directly office automation with documents/email/calendar.  
   URL: https://arxiv.org/abs/2407.19056

3. **OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments** — Xie et al. — 2024  
   Why use: broader computer-use benchmark and execution-based evaluation precedent.  
   URL: https://arxiv.org/abs/2404.07972

4. **NATURAL PLAN: Benchmarking LLMs on Natural Language Planning** — Zheng et al. / Google — 2024  
   Why use: planning-focused meeting/calendar tasks.  
   URL: https://arxiv.org/abs/2406.04520

5. **API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs** — Li et al. — 2023  
   Why use: tool-use baseline/evaluation layer.  
   URL: https://arxiv.org/abs/2304.08244

6. **Berkeley Function Calling Leaderboard (BFCL)** — UC Berkeley — 2025  
   Why use: practical low-level tool-call comparison point.  
   URL: https://gorilla.cs.berkeley.edu/leaderboard.html

7. **OpenClaw official repo + docs** — OpenClaw — active  
   Why use: authoritative description of what OpenClaw is and how it supports assistant-like operation.  
   URLs: https://github.com/openclaw/openclaw ; https://docs.openclaw.ai/

8. **PinchBench** — Kilo AI / PinchBench — active  
   Why use: practical evidence that OpenClaw-like agents are being benchmarked on scheduling and email triage tasks.  
   URL: https://github.com/pinchbench/skill

9. **QMSum** — Zhong et al. / Yale LILY — 2021  
   Why use: meeting prep and query-focused summarization benchmark.  
   URLs: https://aclanthology.org/2021.naacl-main.472/ ; https://github.com/Yale-LILY/QMSum

10. **MeetingBank** — Hu et al. — 2023  
    Why use: meeting minutes / summarization benchmark for follow-up notes.  
    URL: https://aclanthology.org/2023.acl-long.906/

11. **Enron Email Dataset** — Carnegie Mellon University — official corpus  
    Why use: public email source for triage/thread-based experiments or synthetic task generation.  
    URL: https://www.cs.cmu.edu/~enron/

12. **Official Gmail / Google Calendar / Google Tasks / Microsoft Graph Mail / Microsoft Graph Calendar APIs**  
    Why use: realistic implementation targets for inbox, scheduling, and follow-up tracking.  
    URLs:  
    - https://developers.google.com/workspace/gmail/api/guides  
    - https://developers.google.com/workspace/calendar/api/guides/overview  
    - https://developers.google.com/workspace/tasks/reference/rest  
    - https://learn.microsoft.com/en-us/graph/api/resources/mail-api-overview  
    - https://learn.microsoft.com/en-us/graph/api/resources/calendar-overview

## open_questions

- **OpenClaw scientific evidence gap**: I found strong official docs/blog/repo evidence and practical benchmark evidence (PinchBench), but not a primary 2023–2025 peer-reviewed OpenClaw system paper. For the proposal, OpenClaw may need to be described from official project documentation unless a formal paper appears elsewhere.

- **OfficeBench verification**: the source is clearly relevant, but the PDF fetch was noisy. Before final proposal submission, verify author affiliations, exact task counts, and evaluation details directly from the paper PDF.

- **ScheduleMe / SmartCal verification**: excerpt suggests relevance to calendar assistants, but claims should be checked directly in the full PACLIC paper and any cited SmartCal source before inclusion.

- **Inbox triage datasets are weaker than meeting datasets**: there is no obvious modern, official, public dataset exactly matching “enterprise inbox triage + calendar coordination + task tracking” end-to-end. A likely best path is synthetic office workflow generation over public corpora + real APIs.

- **Benchmark fit decision**: if the project uses direct APIs, WorkBench/OfficeBench + custom synthetic tenant evaluation may fit best. If it uses browser/desktop interaction through OpenClaw computer-use tools, OSWorld-style evaluation is more appropriate.

- **Privacy and reproducibility tradeoff**: real user email/calendar data is realistic but difficult to share and score. Synthetic but stateful accounts in Google Workspace or Microsoft 365 may be the best IE686 project compromise.
