# OpenClaw Secretary Agent for Office Productivity

## Executive summary and context

This proposal asks whether an OpenClaw-based secretary agent can reliably support realistic office work across four linked tasks: inbox triage, calendar coordination, meeting preparation, and follow-up task tracking. The motivation is practical rather than purely conversational. In real office settings, these tasks are coupled: an assistant may need to read an email, infer whether it requires action, inspect the calendar, prepare context for an upcoming meeting, and then convert commitments into explicit tasks. Recent benchmarks indicate that multi-step office workflows remain challenging for current agents, especially when they require stateful tool use across applications. WorkBench is the closest benchmark precedent because it evaluates agents in a realistic workplace setting with outcome-based scoring, while OfficeBench and OSWorld further show that office automation and computer-use tasks remain difficult even in controlled environments. This project therefore focuses on a narrower but realistic question: can a constrained secretary agent improve workflow execution quality without causing unacceptable errors or increasing the user’s review burden?

The problem being solved is not just email classification or meeting summarization in isolation. It is the broader problem of coordinated office assistance under real constraints. A request such as “please set up time with the analytics team next week and remind me what we still need from them” requires classification of message intent, retrieval of thread context, schedule reasoning, event proposal, and follow-up tracking. That makes this a good fit for IE686 because it combines natural-language understanding, tool use, multi-agent coordination, and evaluation under measurable outcomes. Secretary work is also high stakes: incorrect recipients, wrong meeting times, or dropped action items are small interface errors but large workflow failures.

## Problem, data, methods, and workflow

The project will build a constrained multi-agent assistant in OpenClaw for a single-user or synthetic-team office setting. OpenClaw appears to be a plausible substrate because its repository and documentation present it as a personal-assistant framework with agent coordination and persistent context; however, the exact Gmail, Calendar, and Tasks integrations required for this project still need to be verified, and some wrappers may need to be implemented. The initial implementation will therefore be limited to Google Workspace test accounts, draft-only outbound email, user-approved calendar changes, and task creation in Google Tasks. The system will expose one user-facing secretary persona, but internally it will use a supervisor plus four specialist agents.

The supervisor agent will receive user requests or proactive triggers from new inbox items, decide whether the task is local or cross-application, route work to specialist agents, maintain shared state, and enforce approval rules. The inbox-triage agent will inspect incoming messages and assign one of a small set of labels such as informational, urgent, scheduling-related, or action-item-bearing. The calendar-coordination agent will query calendar state, evaluate hard constraints such as availability, time zones, duration, and required attendees, and then propose or create feasible events. The meeting-preparation agent will retrieve related email threads, recent calendar context, and prior meeting notes to generate a short prep brief centered on unresolved questions, dependencies, and likely decisions. The follow-up-task agent will extract commitments from meetings or resolved threads and convert them into explicit tasks with owners and tentative deadlines. Shared state for version 1 will include message IDs, event IDs, task IDs, confidence scores, evidence snippets, and pending approvals. Sending email and modifying existing meetings will always require user approval.

The data strategy is a hybrid one designed for both realism and reproducibility. The main execution environment will be one or more Google Workspace sandbox accounts populated with seeded office scenarios. These accounts will contain synthetic inboxes, recurring meetings, unfinished tasks, time-zone conflicts, and multistep communication threads so that the agent can interact with real APIs while the evaluation remains controlled. This design is inspired by recent office-agent benchmarks such as WorkBench and OfficeBench, which emphasize realistic but auditable workplace environments. The benchmark will include a fixed set of seeded scenarios with pre-specified expected end states, allowing exact scoring of inbox labels, calendar outcomes, created tasks, and approval counts.

Public datasets will be used primarily to seed realistic content rather than to serve as direct end-to-end benchmarks. The Enron Email Dataset will provide realistic email language and thread structure for triage and reply drafting, although it is old and will require relabeling or synthetic scenario construction. For meeting assistance, the AMI Meeting Corpus and QMSum are useful because they support meeting understanding and query-focused summarization, which map naturally to prep tasks such as identifying unresolved issues before a meeting. MeetingBank provides business-style meeting summaries that may help with follow-up-task generation. These datasets will therefore support content realism, while the API-backed sandbox will define the actual end-to-end evaluation tasks.

Methodologically, the project will test the hypothesis that explicit decomposition and role separation may improve reliability on cross-application tasks compared with simpler baselines. The system will combine structured function calling for Gmail, Calendar, and Tasks operations; retrieval over recent email threads and meeting artifacts; lightweight workspace memory; and guarded execution with confirmation before high-risk actions. A primary LLM with strong function-calling and long-context behavior will be used for the supervisor and generation-heavy subtasks, while smaller models may be considered for extraction subtasks if time permits. The proposal does not depend on one specific model vendor; instead, it focuses on an architecture that can be compared fairly against simpler alternatives using the same tool access.

## Evaluation plan

Success will be measured mainly through execution-based, outcome-centric evaluation. The primary metric will be end-to-end task success rate, defined as whether the final inbox, calendar, and task states match the intended result for a scenario. This follows the logic of WorkBench, where outcome correctness matters more than reproducing one exact action trace. The evaluation set will include both component tasks and integrated tasks. Component tasks will cover inbox triage, calendar scheduling, meeting-prep generation, and follow-up extraction separately. Integrated tasks will combine them, for example: read a scheduling email, inspect the calendar, propose a feasible meeting, draft the reply, and create a follow-up task. Expected end states will be labeled in advance so that success can be scored deterministically.

Several secondary metrics will capture quality, safety, and user burden. Inbox triage will be scored with precision and recall for urgency and intent labels on a small annotated set. Calendar coordination will be scored by scheduling feasibility rate, meaning the fraction of proposed meetings that satisfy all hard constraints. Follow-up extraction will be evaluated with precision, recall, and F1 against a gold set of action items derived from meeting or thread content. Meeting-preparation briefs will be scored by human raters on three fixed dimensions: factuality, completeness, and actionability. Operational metrics will include latency, tool-call count, and approval burden, measured as the number of user confirmations or edits required per completed task. Because secretary-style assistants can fail in costly ways, the project will also report a critical error rate covering wrong-recipient drafts, incorrect event creation, dropped action items, or duplicate meetings.

The project will compare the full multi-agent system against at least two fair baselines. The first baseline will be a single-agent system using the same model family, the same APIs, and the same task budget, but without role specialization or explicit shared state. The second will be a rules-plus-LLM baseline in which heuristics handle simple filtering and scheduling constraints while the model focuses on drafting and summarization. These baselines will help test whether any performance gain comes from decomposition and coordination rather than from extra model capacity or looser prompting. The project may also use NATURAL PLAN for planning-only diagnostics and API-Bank or BFCL for tool-calling diagnostics, but these will be treated as subproblem references rather than as substitutes for the end-to-end office benchmark.

## Risks and open questions

The main risks are privacy, scope, and benchmark construction. Real office data would improve realism but would reduce reproducibility and raise privacy concerns, so the main evaluation will remain centered on sandbox accounts and synthetic scenarios. There is also some uncertainty about how much OpenClaw integration work will be required, since the exact Gmail, Calendar, and Tasks connectors needed for this project have not been fully verified from the available documentation. Finally, no public dataset perfectly captures the full secretary workflow, so part of the project’s contribution will be creating a small but realistic seeded evaluation suite. These risks are manageable if the first version remains narrow: Google Workspace only, bounded task types, approval-gated external actions, and a clearly labeled benchmark with fixed end states.

## References and source information

Styles, O., Miller, S., Cerda-Mardini, P., Guha, T., Sanchez, V., and Vidgen, B. *WorkBench: a Benchmark Dataset for Agents in a Realistic Workplace Setting*. MindsDB and collaborators, 2024. https://arxiv.org/abs/2405.00823

Wang, Z., Cui, Y., Zhong, L., Zhang, Z., Yin, D., Lin, B. Y., and Shang, J. *OfficeBench: Benchmarking Language Agents across Multiple Applications for Office Automation*. 2024. https://arxiv.org/abs/2407.19056

Xie, T. et al. *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*. 2024. https://arxiv.org/abs/2404.07972

Zheng, H. S., Mishra, S., Zhang, H., Chen, X., Chen, M., Nova, A., Hou, L., Cheng, H.-T., Le, Q. V., Chi, E. H., and Zhou, D. *NATURAL PLAN: Benchmarking LLMs on Natural Language Planning*. Google, 2024. https://arxiv.org/abs/2406.04520

Li, Y. et al. *API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs*. 2023. https://aclanthology.org/2023.emnlp-main.187/ ; https://arxiv.org/abs/2304.08244

UC Berkeley Gorilla Project. *Berkeley Function Calling Leaderboard*. Accessed as a practical benchmark reference, 2025. https://gorilla.cs.berkeley.edu/leaderboard.html

Zhong, M. et al. *QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization*. NAACL 2021. https://aclanthology.org/2021.naacl-main.472/

Hu, Y. et al. *MeetingBank: A Benchmark Dataset for Meeting Summarization*. ACL 2023. https://aclanthology.org/2023.acl-long.906/

Carnegie Mellon University. *Enron Email Dataset*. https://www.cs.cmu.edu/~enron/

AMI Project. *AMI Meeting Corpus download page*. https://groups.inf.ed.ac.uk/ami/download/

Google Developers. *Gmail API Guides*. https://developers.google.com/workspace/gmail/api/guides

Google Developers. *Google Calendar API Overview*. https://developers.google.com/workspace/calendar/api/guides/overview

Google Developers. *Google Tasks API Reference*. https://developers.google.com/workspace/tasks/reference/rest

OpenClaw. *OpenClaw official repository*. https://github.com/openclaw/openclaw

OpenClaw. *OpenClaw documentation* and *personal assistant setup guide*. https://docs.openclaw.ai/ ; https://docs.openclaw.ai/start/openclaw
