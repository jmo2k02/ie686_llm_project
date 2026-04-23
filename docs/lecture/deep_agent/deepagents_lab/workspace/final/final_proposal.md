# OpenClaw Secretary — Executive Summary

Modern knowledge work is dominated by time-consuming coordination: triaging an overflowing inbox, negotiating meeting times, preparing concise meeting briefs, and tracking follow-ups. These tasks are high-frequency, low-strategy, and often interruptive—making them ideal targets for partial automation. OpenClaw Secretary is a modular, privacy-conscious multi-agent assistant that automates core secretary functions (inbox triage, calendar coordination, meeting preparation, and follow-up tracking) while keeping humans in control. The system will be implemented on OpenClaw as the orchestration/runtime layer, integrating OAuth-backed connectors, a retrieval-backed knowledge store, and narrowly scoped specialist agents. Evaluation will combine reproducible automated benchmarks and controlled user studies reflecting realistic office workflows.

1) Problem (what we are solving)

Knowledge workers regularly spend substantial weekly time on coordination tasks: deciding which messages require action, negotiating meeting times across calendars, preparing distilled briefs before meetings, and ensuring assigned follow-ups are completed. The project reduces cognitive load and time spent on these routine tasks by building an assistant that: (a) triages incoming messages into actionable categories; (b) coordinates and confirms meetings with minimal negotiation rounds; (c) generates concise, source-cited meeting briefs and agendas; and (d) detects and tracks follow-up tasks to closure. The assistant emphasizes conservative defaults, auditable decision logs, and explicit human approval for side-effecting actions.

2) Data (what data, where, how gathered)

Planned data sources and access model:
- Public corpora for reproducible benchmarking: Enron Email Dataset (https://www.cs.cmu.edu/~enron/), Avocado Research Email collection (LDC2015T03), QMSum (https://github.com/Yale-LILY/QMSum), MeetingBank (https://meetingbank.github.io/), AMI/ICSI meeting corpora.
- Production integration: OAuth connectors to Gmail API, Google Calendar API, and Microsoft Graph for reading/acting on mail and calendar items. Transcription via AssemblyAI or Google Speech-to-Text for meeting audio.
- Pilot collection: opt-in, anonymized enterprise logs and consenting meeting recordings for fine-tuning and user-study realism (legal/IRB review required). All production data is subject to tenant policies and may use tenant-hosted inference and tenant-managed key management.

Data handling and labeling:
- Triage label schema: {action_required, meeting_request, informational, read_later, spam}. Target held-out test set ≈2,000 messages with ≥200 examples per non-spam class for benchmarking.
- Meeting brief annotations: target 300–500 annotated meeting briefs mapping transcript passages to concise, cited summaries.
- Follow-up annotation: 300–500 examples labeling action items, owners, and completion timestamps for span-level F1 evaluation.
- When internal data is unavailable, use augmented public/synthetic data to match label distributions and document the domain shift risk.

3) Solution and multi-agent workflow (methods, models, orchestration)

Architecture and agent roles (OpenClaw as orchestrator):
- Inbox Triage Agent: a classifier (fine-tuned DistilBERT/BERT) and entity/temporal extractor that labels messages and extracts candidate actions (meeting proposals, tasks, deadlines).
- Scheduler Agent: availability matcher and negotiation manager that proposes slots (first proposal + up to one counterproposal = negotiation round), interfaces with calendars via API connectors, and surfaces final confirmations to users for approval.
- Meeting Prep Agent: retrieval-augmented generator (RAG) that constructs concise pre-meeting briefs with explicit citations to thread passages, documents, and calendar metadata. Use Longformer/LED or T5-family models for long-context summarization when needed.
- Follow-up Tracker Agent: extracts action items from emails/transcripts, assigns owners, issues reminders, and tracks closure state.
- Integrator/Orchestrator Agent (OpenClaw): routes data, enforces policy and RBAC, records provenance and decision logs, manages connectors and caching, and exposes a minimal UI/API for human approvals.

Model strategy:
- Early iterations prioritize prompt engineering + RAG to reduce annotation cost and speed iteration. Fine-tuning is reserved for persistent failure modes: triage classification, sensitivity detection, and stylistic constraints for generated replies/briefs.
- Retrieval is backed by a vector store (FAISS/Pinecone/Weaviate) over embeddings; retrieval contexts are fed to the generator with provenance metadata for citations.
- Safety: PII detectors and NER-based redaction precede any external transmission; tenant-hosted inference and tenant KMS are supported for sensitive deployments.

Baselines and reproducibility:
- Baselines: rule-based triage, DistilBERT classifier, lead-3/PEGASUS/T5 summarizers, deterministic first-available scheduler, and a zero-shot LLM prompting baseline for generation.
- Public datasets (Enron, QMSum, MeetingBank, AMI) will serve reproducible baselines; results on pilot internal data will be reported separately with anonymization procedures described.

4) Evaluation plan (how success is measured)

Automated metrics (component-level):
- Triage: per-class precision/recall/F1, confusion matrix, and calibration (Brier score). Target: F1 ≥ 0.80 on held-out test set (to be validated in pilot).
- Summarization: ROUGE/L, BERTScore, QA-based factuality (QAGS/QAFactEval), and citation-precision (percent of summary facts supported by retrieved passages). Human usefulness Likert ratings supplement automated scores.
- Scheduling: negotiation rounds to agreement, time-to-confirm, participant satisfaction (Likert), and % requiring manual reschedule. Target: ≥75% of meeting requests scheduled within ≤2 negotiation rounds.
- Follow-up tracking: span-level F1 for action-item extraction, owner-assignment accuracy, and time-to-closure distribution.

User study (system-level):
- Design: counterbalanced within-subjects 2-week deployment; target N=20–30 knowledge workers; pilot N≈6 for feasibility and power estimation. Primary outcome: time saved per week (instrumented). Secondary: task completion latency, SUS, override frequency, and qualitative interviews.
- Hypotheses: measurable time savings (target ≥30 minutes/week to be validated), reduced task latencies, and SUS improvement relative to baseline.
- Ethics: IRB submission, informed consent, and tenant/legal sign-off for any internal log usage.

Success criteria (example thresholds to validate in pilot):
- Triage F1 ≥ 0.80; scheduling success ≥75% within ≤2 rounds; meeting briefs rated "useful" by ≥70% of users; measurable average time saving per user (pilot-determined).

Timeline, resources, and risks

- Timeline (~6 months): months 0–1 connectors, data & policy framework; months 1–3 agent development and RAG prompt iterations; months 3–4 fine-tuning and synthetic benchmark creation; months 4–5 internal pilot; month 6 user study and analysis.
- Team: 4-person core (ML engineer, backend/infra, UX/designer, research assistant/annotator). Compute: intermittent 1–2 GPU-class instances for fine-tuning; production depends on tenant hosting choices.
- Annotation: ~1,000–3,000 triage labels initially, 300–500 meeting brief and follow-up annotations.
- Primary risks and mitigations: privacy/legal constraints (mitigate via opt-in, tenant-hosted inference, KMS, and IRB); hallucination (RAG, citation-precision checks, human approvals); adoption/trust (conservative defaults, editable drafts, provenance UI).

Sources and short references (selected)

- Toolformer — https://arxiv.org/abs/2302.04761 (self-supervised tool use)
- ReAct — https://arxiv.org/abs/2210.03629 (reasoning + acting pattern)
- Reflexion — https://arxiv.org/abs/2303.11366 (iterative agent improvement)
- RAG — https://arxiv.org/abs/2005.11401 (retrieval-augmented generation)
- QMSum — https://github.com/Yale-LILY/QMSum and https://aclanthology.org/2021.naacl-main.472.pdf (meeting summarization)
- MeetingBank — https://meetingbank.github.io/ and https://aclanthology.org/2023.acl-long.906.pdf (large meeting dataset)
- Enron Email Dataset — https://www.cs.cmu.edu/~enron/ (public email corpus)
- Avocado Research Email Collection (LDC2015T03) — https://catalog.ldc.upenn.edu/LDC2015T03
- AMI Meeting Corpus — https://www.openslr.org/16/
- Gmail API — https://developers.google.com/workspace/gmail/api
- Microsoft Graph — https://learn.microsoft.com/en-us/graph/overview
- QAGS (factuality metric) — https://aclanthology.org/2020.acl-main.450.pdf

Deliverables

- A reproducible prototype on OpenClaw with modular agents for triage, scheduling, meeting briefs, and follow-ups; connectors for Gmail/Calendar/Microsoft Graph; RAG-backed generation with citation provenance; and a human approval UI.
- A benchmark suite combining public datasets (Enron, QMSum, MeetingBank, AMI) and an anonymized internal pilot dataset (where permitted), with code and evaluation scripts for triage, summarization, scheduling, and follow-up extraction.
- A controlled within-subjects user study report with quantitative and qualitative analysis.

Next steps

- Finalize IRB/legal approvals for pilot data collection and tenant-hosting options.
- Implement connectors and a minimal triage + scheduler prototype on OpenClaw for a 6-week pilot.
- Run the pilot, collect annotated examples, and iterate on prompts and fine-tuning for persistent failure modes.

This proposal provides a concrete, evidence-grounded path to building and evaluating OpenClaw Secretary while prioritizing privacy, auditable decision-making, and realistic office workflows.
