{
  "title": "Research memo: Building a 'Secretary Agent' with OpenClaw",
  "date": "2026-03-18",
  "1_comparable_systems": [
    {
      "name": "Superhuman",
      "capabilities": "Email triage, fast inbox navigation, follow-up reminders, snippets; limited scheduling integrations",
      "stage": "production",
      "url": "https://superhuman.com"
    },
    {
      "name": "SaneBox",
      "capabilities": "Automated inbox triage (folders: SaneLater), snooze, follow-up reminders",
      "stage": "production",
      "url": "https://www.sanebox.com"
    },
    {
      "name": "Clara Labs (Clara)",
      "capabilities": "Human+AI scheduling assistant that manages meeting coordination and reminders",
      "stage": "production/service",
      "url": "https://claralabs.com/ (historical service; company site)")
    },
    {
      "name": "Calendly",
      "capabilities": "Automated scheduling and calendar coordination (focus: booking), integrates with calendars",
      "stage": "production",
      "url": "https://calendly.com"
    },
    {
      "name": "Reclaim.ai",
      "capabilities": "Calendar optimization: time-blocking, task-to-calendar automation, meeting batching",
      "stage": "production",
      "url": "https://reclaim.ai"
    },
    {
      "name": "Otter.ai",
      "capabilities": "Meeting transcription, highlights, searchable transcripts, summary snippets",
      "stage": "production",
      "url": "https://otter.ai"
    },
    {
      "name": "Fireflies.ai",
      "capabilities": "Meeting recording, automated notes and follow-ups, integrations with calendar/Slack",
      "stage": "production",
      "url": "https://www.fireflies.ai"
    },
    {
      "name": "Gong / Chorus (sales intelligence)",
      "capabilities": "Meeting capture, action-item extraction, follow-up tracking (enterprise)",
      "stage": "production",
      "url": "https://www.gong.io, https://www.chorus.ai"
    },
    {
      "name": "Boomerang / Respondable",
      "capabilities": "Email scheduling, follow-up nudges, response-probability estimation",
      "stage": "production",
      "url": "https://www.boomeranggmail.com"
    },
    {
      "name": "Google Workspace features (Priority Inbox, Smart Reply/Compose)",
      "capabilities": "Automated triage signals, suggested replies, calendar suggestions",
      "stage": "production",
      "url": "https://workspace.google.com"
    },
    {
      "name": "Reflexion (research framework)",
      "capabilities": "A research framework for language agents to self-reflect and improve—relevant to iterative assistant behavior (self-correction, tracking)",
      "stage": "research",
      "url": "https://arxiv.org/abs/2303.11366"
    },
    {
      "name": "Toolformer / ReAct style agents (research)",
      "capabilities": "Tool-use, reasoning+action orchestration for web/API interactions—enables assistants to call APIs reliably",
      "stage": "research",
      "url": "https://arxiv.org/abs/2302.04761, https://arxiv.org/abs/2210.03629"
    }
  ],

  "2_candidate_datasets_and_APIs": {
    "email_inbox": [
      {"name":"Enron Email Dataset (Cornell/CMU)", "access":"public, ~1.7GB tar.gz, widely used for email research","url":"https://www.cs.cmu.edu/~enron/"},
      {"name":"Avocado Research Email Collection (LDC2015T03)", "access":"LDC catalog (paid/subscription), ~279 accounts, structured metadata","url":"https://catalog.ldc.upenn.edu/LDC2015T03"}
    ],
    "meetings_and_calendar": [
      {"name":"AMI Meeting Corpus","access":"public download (100 hours), multi-modal transcripts/annotations","url":"https://www.openslr.org/16/","notes":"~100 hours, good for meeting structure tasks"},
      {"name":"ICSI Meeting Corpus","access":"public (register/download), ~70 hours, transcripts","url":"https://groups.inf.ed.ac.uk/ami/icsi/"},
      {"name":"QMSum","access":"public (GitHub + ACL paper), 232 meetings, query-based summaries","url":"https://github.com/Yale-LILY/QMSum and https://aclanthology.org/2021.naacl-main.472.pdf"},
      {"name":"MeetingBank","access":"public release (ACL 2023), 1,366 city-council meetings (~3,579 hours) —good for long, noisy meetings","url":"https://meetingbank.github.io/ and https://aclanthology.org/2023.acl-long.906.pdf"}
    ],
    "task_and_followup": [
      {"name":"Email+Calendar event corpora (task labels are sparse)", "access":"mix: public + proprietary; consider collecting instrumented user data or synthetic annotations","notes":"no large public dataset that pairs email->task-tracking+calendar at scale; recommended to collect internal opt-in logs or use crowd-labeling"}
    ],
    "integration_APIs_and_transcription": [
      {"name":"Gmail API","access":"requires OAuth and project registration; full mail access with scopes","url":"https://developers.google.com/workspace/gmail/api"},
      {"name":"Google Calendar API","access":"OAuth; easily accessible; event creation/reading","url":"https://developers.google.com/calendar"},
      {"name":"Microsoft Graph (Mail/Calendar/Teams)","access":"OAuth; enterprise scope; rich calendaring and mail operations","url":"https://learn.microsoft.com/en-us/graph/overview"},
      {"name":"Zoom API (recordings/transcripts)","access":"OAuth/JWT; paid features for transcripts","url":"https://marketplace.zoom.us/docs/api-reference/zoom-api"},
      {"name":"AssemblyAI / Rev.ai / Google Speech-to-Text / Amazon Transcribe","access":"paid APIs (auth keys), high-quality transcripts useful for meeting summarization","notes":"AssemblyAI has research-friendly docs; all are paid","urls":"https://www.assemblyai.com, https://www.rev.ai, https://cloud.google.com/speech-to-text, https://aws.amazon.com/transcribe/"}
    ]
  },

  "3_benchmarks_and_evaluation_metrics": [
    {"metric":"Task success / completion rate","notes":"User study metric: whether assistant completed requested tasks (booked meeting, sent follow-up)"},
    {"metric":"Precision/Recall/F1 for classification (triage labels, action-item detection)","notes":"Standard for detection/extraction tasks"},
    {"metric":"User time saved (minutes) and task duration reduction","notes":"Measured in user studies or instrumented deployment"},
    {"metric":"User satisfaction / usability (SUS), Net Promoter, Likert ratings","notes":"SUS common for system-level usability"},
    {"metric":"ROUGE / BLEU / ROUGE-L / BERTScore for summaries","notes":"Used in QMSum, MeetingBank baselines"},
    {"metric":"Factuality metrics: QAGS / FEQA / FactCC","notes":"Detects hallucinations in abstractive summaries (see Wang et al., 2020 QAGS)"},
    {"metric":"Human preference / A/B testing","notes":"Pairwise preference for summaries/outputs is common"}
  ],

  "4_recent_relevant_papers_(2020-2025)": [
    {"citation":"Lewis et al., 2020 - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)", "relevance":"Core method for grounding assistant responses with external documents/records.", "url":"https://arxiv.org/abs/2005.11401"},
    {"citation":"Schick et al., 2023 - Toolformer: Language Models Can Teach Themselves to Use Tools", "relevance":"Self-supervised method to teach LMs to call APIs reliably — directly relevant for a secretary agent using mail/calendar APIs.", "url":"https://arxiv.org/abs/2302.04761"},
    {"citation":"Yao et al., 2022 - ReAct: Synergizing Reasoning and Acting in Language Models", "relevance":"Interleaves reasoning traces and API actions — useful for planning and executing calendar/email operations.", "url":"https://arxiv.org/abs/2210.03629"},
    {"citation":"Shinn et al., 2023 - Reflexion: Language Agents with Verbal Reinforcement Learning (NeurIPS 2023)", "relevance":"Framework for self-reflection and iterative improvement for agents (tracking, error recovery, follow-ups).", "url":"https://arxiv.org/abs/2303.11366"},
    {"citation":"Zhong et al., 2021 - QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization", "relevance":"Benchmark and dataset for meeting summarization and QA-style queries over meetings.", "url":"https://aclanthology.org/2021.naacl-main.472.pdf"},
    {"citation":"MeetingBank paper, ACL 2023 - MeetingBank: A Benchmark Dataset for Meeting Summarization", "relevance":"Large, real-world dataset (city councils) for long noisy meeting summarization and extraction.", "url":"https://aclanthology.org/2023.acl-long.906.pdf"},
    {"citation":"Wang et al., 2020 - QAGS: Asking and Answering Questions to Evaluate Factuality", "relevance":"Factuality metric for abstractive summaries — important for reliable assistant-generated meeting notes and follow-ups.", "url":"https://aclanthology.org/2020.acl-main.450.pdf"},
    {"citation":"Wei et al., 2022 - Chain-of-Thought Prompting Elicits Reasoning in LLMs", "relevance":"Techniques for eliciting intermediate reasoning steps, helpful for explainable decision traces in assistants.", "url":"https://arxiv.org/abs/2201.11903"}
  ],

  "5_recommended_shortlist_for_proposal": {
    "systems_for_comparison_and_integration": ["Gmail API + Google Calendar API (production integrations)", "Otter.ai or AssemblyAI (transcription)", "Reclaim.ai or Calendly (scheduling UX patterns)"],
    "datasets_to_use": ["Enron (email) + Avocado (LDC) for offline email triage prototyping", "QMSum + MeetingBank + AMI/ICSI for meeting summarization and extraction"],
    "metrics_to_report": ["Task success (user study), Time saved (instrumented), Precision/Recall for action extraction, ROUGE / BERTScore for summaries, QAGS for factuality, SUS for usability"],
    "rationale":"This shortlist balances production-ready APIs (Gmail/Calendar) for integration with public datasets for reproducible evaluation (QMSum, MeetingBank, Enron). Metrics combine automated (ROUGE/BERTScore, QAGS) and user-centered measures (task success, time saved, SUS), and the research papers listed provide state-of-the-art methods to implement tool use (Toolformer), grounding (RAG), and iterative agent improvement (Reflexion/ReAct)."
  },

  "primary_source_URLs": [
    "https://arxiv.org/abs/2302.04761 (Toolformer)",
    "https://arxiv.org/abs/2210.03629 (ReAct)",
    "https://arxiv.org/abs/2303.11366 (Reflexion)",
    "https://arxiv.org/abs/2005.11401 (RAG)",
    "https://aclanthology.org/2021.naacl-main.472.pdf (QMSum)",
    "https://meetingbank.github.io/ and https://aclanthology.org/2023.acl-long.906.pdf (MeetingBank)",
    "https://www.cs.cmu.edu/~enron/ (Enron)",
    "https://catalog.ldc.upenn.edu/LDC2015T03 (Avocado LDC)",
    "https://www.openslr.org/16/ (AMI)",
    "https://groups.inf.ed.ac.uk/ami/icsi/ (ICSI)",
    "https://developers.google.com/workspace/gmail/api (Gmail API)",
    "https://learn.microsoft.com/en-us/graph/overview (Microsoft Graph)"
  ]
}
