critique

Major gaps in rubric coverage and weaknesses

- Data access and representativeness: The draft names Enron, QMSum, AMI, and MeetingBank but does not specify which labeled subsets will be used, how those corpora map to your target triage taxonomy, or whether you can legally access comparable internal enterprise logs. Enron is dated and non-representative of modern corporate email; relying on it without augmentation risks domain mismatch.
- Triage taxonomy and labeling plan: The proposal lacks a concrete label schema (exact classes), annotation protocol, inter-annotator agreement targets, or expected labeled dataset sizes per class. The triage success criterion (F1 ≥ 0.80) is ungrounded without a test-set size and label distribution.
- User study rigor: The within-subjects 2-week study is underspecified: no N, no recruitment criteria, no counterbalancing, no power analysis, and no plan to control for confounds (week-to-week variability in workload).
- Evaluation metrics are too thin or vague: e.g., "scheduling success rate within <=2 negotiation rounds" is not defined precisely (what constitutes a negotiation round, failed attempts due to availability vs. unwillingness). Summarization metrics rely on ROUGE/BERTScore/QAGS without acknowledging their known weaknesses; no plan to measure factuality or citation precision beyond QAGS.
- Baselines missing or weak: The proposal lists generic baselines (rule-based filters, off-the-shelf summarizers) but does not specify concrete models (e.g., DistilBERT classifier, LED/T5 summarizer, or a deterministic scheduler). This makes reproducibility and fair comparison difficult.
- Privacy and compliance specifics: High-level mitigations are listed (opt-in, RBAC, on-prem inference) but there is no concrete threat model, redaction method (PII detectors, named-entity rules), or legal/IRB review plan.
- Hallucination mitigation is underspecified: RAG + citations are suggested but no mechanism to evaluate citation accuracy or to prevent confident but unsupported assertions.

Vague claims and unsupported citations

- "Average time saved >=30 minutes/week" is an arbitrary target with no supporting pilot data or calculation.
- Claims about deployment costs and annotation effort (1,000–3,000 labels) lack granularity on per-class needs or expected annotation throughput.
- Suggesting differential-access and short-lived embeddings is fine, but there is no reference for the chosen cryptographic or key-management approach.

Weak design/evaluation choices

- Using Enron as a primary surrogate dataset for email triage is weak due to domain drift and dated conventions. Add modern enterprise or synthetic datasets.
- QAGS alone is insufficient for summarization faithfulness evaluation; QAGS has failure modes for long dialogs and conversational data.
- Scheduling success metric should include latency, user satisfaction, and fairness (avoiding busy-person bias), not only rounds.


edits (prioritized concrete text-level fixes)

1) Add an explicit triage label taxonomy and minimum labeled-test size. Example text: "Triage labels: {action_required, meeting_request, informational, read_later, spam}; target test set 2,000 messages with ≥200 examples per non-spam class." (Justify or mark as open if data access prevents this.)
2) Define baselines precisely. Add a short list: "Baselines: (a) rule-based subject/keyword filter; (b) BERT/DistilBERT classifier fine-tuned on triage labels; (c) LED/T5 baseline for meeting briefs; (d) deterministic scheduler that proposes first-available slots." Replace generic wording with these concrete names.
3) Specify user study design: target N (e.g., 20–30 participants), within-subjects counterbalancing, primary outcome (time saved/week), and brief power calculation or pilot plan. Add IRB/consent sentence.
4) Clarify scheduling metric and negotiation semantics: define "negotiation round" and secondary metrics (latency to confirm, participant satisfaction, fraction requiring manual follow-up).
5) Strengthen faithfulness evaluation: add citation-precision (percent of summary facts supported in retrieved docs), QAFactEval or human factuality checks, and per-statement grounding checks.
6) Nail down privacy controls: add mention of PII detectors for redaction, tenant-hosted model inference option, and encryption key management (KMS or tenant-provided keys) as a minimum.
7) Replace unsupported claim (30 minutes/week) with phrasing like "target: >=30 minutes/week; to be validated by pilot study" and add justification sentence or mark as hypothesis.


additions / alternative datasets, baselines, metrics

Datasets
- Replace/augment Enron with: (a) Avocado Research Email dataset (more recent enterprise-like messages), (b) synthetic corpora constructed from modern inbox patterns + crowdsourced templates, and (c) anonymized pilot data from opt-in employees (with IRB/legal signoff).
- For meeting summarization, keep QMSum and AMI but add AMI variants and MeetingBank; consider using the CIC datasets or internal recorded meetings if privacy allows.

Baselines
- Triage: rule-based (regex/keyword + sender heuristics), DistilBERT/BERT classifier, and a zero-shot instruction-following LLM baseline (e.g., prompting a general LLM without fine-tuning).
- Summarization: lead-3, PEGASUS/T5, and extractive + abstractive hybrid (LexRank + BART). Use a Longformer/LED baseline for long meeting contexts.
- Scheduling: human-only baseline and a deterministic slot-filling scheduler (first-available algorithm) with identical UI affordances.

Metrics
- Triage: per-class precision/recall/F1, confusion matrix, and calibration (Brier score); evaluate on held-out time-based splits.
- Summaries: ROUGE/BERTScore + QAFactEval or QAFactEval-style QA-based factuality, human Likert scales (usefulness, conciseness, factuality), and citation-precision (percent of asserted facts with supporting retrieved doc).
- Scheduling: rounds to agreement, time-to-confirm, participant satisfaction, and percentage of meetings needing manual reschedule.
- Follow-ups: span-level F1 for action items, owner-assignment accuracy, time-to-closure distribution.


judgment: pass/fail for rubric readiness

Fail (not yet rubric-ready).

Why: The draft answers the four IE686 questions at a high level and has good system thinking, but it lacks several concrete, testable details required by the rubric: a defined triage taxonomy and labeled-test sizes, explicit baselines and model choices, a fully-specified user-study plan (N, power), and concrete privacy/redaction procedures. These gaps prevent the proposal from being evaluable and reproducible as required by the course rubric.

What is required to pass: add the prioritized edits above (particularly triage schema + test-set size, concrete baselines, and a complete user-study plan with N and IRB/consent procedures). After those fixes the proposal will be rubric-ready.


revised_draft

Title: OpenClaw Secretary — A Multi-Agent Office Assistant for Inbox Triage, Calendar Coordination, Meeting Preparation, and Follow-up Tracking

Executive summary

Knowledge workers spend substantial hours on low-value coordination: managing inboxes, negotiating meeting times, preparing meeting briefs, and tracking follow-ups. OpenClaw Secretary proposes a modular multi-agent assistant hosted on OpenClaw that automates these tasks with explicit human-in-the-loop checkpoints and tenant-grade privacy controls. The system integrates OAuth-backed connectors for common mail and calendar APIs, a retrieval-backed knowledge store, and narrow specialist agents for inbox triage, scheduling negotiation, meeting-brief generation, and follow-up tracking. We prioritize retrieval-augmented generation (RAG) for faithfulness, prompt engineering for early iterations, and targeted fine-tuning for persistent failure modes. Evaluation combines automated metrics (triage per-class precision/recall/F1, ROUGE/BERTScore + QA-based factuality checks and citation-precision for summaries, scheduling rounds and time-to-confirm) with a within-subjects user study measuring time saved, task completion latency, SUS, and qualitative feedback.

Problem statement

We will reduce time and cognitive load on routine coordination tasks by building an assistant that automates inbox triage, negotiates and schedules meetings, prepares concise pre-meeting briefs, and tracks follow-ups while preserving user control and privacy.

Data

Primary data sources: opt-in internal email and calendar logs (with IRB/legal sign-off), meeting transcripts from consenting recordings, and shared documents. Early development will use public and synthetic corpora: Enron (for reproducibility but not as sole source), Avocado Research Email dataset, and synthetic modern-inbox corpora generated via crowdsourcing to match target label distributions. For meeting summarization use QMSum, AMI, MeetingBank, and any available anonymized tenant meetings. Connectors: Gmail API, Microsoft Graph, Zoom API; transcription: AssemblyAI or Google STT. Sensitive data will be redacted using PII/NER detectors, tenant-hosted inference option provided, and encryption with tenant-managed KMS when required.

Methods and multi-agent workflow

Five agents: (1) Inbox Triage — classifier + entity/time extractor to label messages and surface actionable items. Triage taxonomy (proposed): {action_required, meeting_request, informational, read_later, spam}. We plan a labeled test set of ~2,000 messages (≥200 examples per non-spam class) for benchmark reporting. (2) Scheduler — proposes and negotiates slots from calendar availability and confirms after human approval. A "negotiation round" is defined as one proposal + at-most-one counterproposal cycle from participants. (3) Meeting Prep — RAG-enabled brief generator that cites source passages and outputs action-item templates. (4) Follow-up Tracker — extracts action items and owners from transcripts/threads and tracks closure. (5) Integrator/Orchestrator — routes data, enforces policies, records provenance, and serves the UI.

Model strategy: use prompt engineering and RAG for generation; fine-tune compact transformer encoders (DistilBERT/BERT) for triage and NER; consider Longformer/LED or T5-family for long-context meeting briefs. Baselines: (a) rule-based filters, (b) DistilBERT classifier, (c) LED/T5 summarizer baseline, (d) deterministic first-available scheduler.

Evaluation plan

Automated benchmarks: triage per-class precision/recall/F1 and calibration (Brier score) on time-split held-out data; summarization evaluated by ROUGE/BERTScore and QA-based factuality checks (e.g., QAFactEval/QAGS) plus citation-precision (percent of claims grounded in retrieved documents) and human Likert judgments for usefulness and factuality. Scheduling metrics: rounds to agreement (defined above), time-to-confirm, participant satisfaction, and fraction requiring manual reschedule. Follow-ups: span-level F1 for extracted action items and owner-assignment accuracy.

User study: within-subjects 2-week deployment with N=20–30 knowledge-worker participants, counterbalanced order, measuring per-user time saved/week (primary), task completion latency, SUS, override rate, and semi-structured interviews. We will run a small pilot (N≈6) for feasibility and use that to justify final N/power calculation. All participants will consent and studies will be submitted for IRB review.

Success criteria

Triage: per-class F1 ≥0.80 on the held-out test set (to be reported per-class). Scheduling: ≥75% of meeting requests scheduled within ≤2 negotiation rounds and median time-to-confirm under 24 hours. Summaries: ≥70% of user-judged briefs labeled "useful" in study and citation-precision >80% on evaluated facts. User impact: average time saved target ≥30 minutes/week to be validated by the pilot (treated as a hypothesis until study confirms).

Risks and mitigations

Privacy: use opt-in and consent, tenant-hosted inference, PII/NER redaction, tenant KMS for encryption, and RBAC. Hallucination: require explicit source citations in the UI, evaluate citation-precision, conservative prompting, and require human approval for sensitive replies. Adoption: start with suggestions (no auto-send), incremental automation, clear provenance, and editable drafts.

References

- Enron Email Dataset — https://www.cs.cmu.edu/~enron/
- Avocado Research Email Dataset — https://www.cs.cmu.edu/ (cite dataset page)
- QMSum — https://github.com/Yale-LILY/QMSum and https://aclanthology.org/2021.naacl-main.472.pdf
- MeetingBank — https://meetingbank.github.io/ and https://aclanthology.org/2023.acl-long.906.pdf
- Toolformer — https://arxiv.org/abs/2302.04761
- ReAct — https://arxiv.org/abs/2210.03629
- RAG — https://arxiv.org/abs/2005.11401
- QAGS/QAFactEval — https://aclanthology.org/2020.acl-main.450.pdf


-- end of feedback
