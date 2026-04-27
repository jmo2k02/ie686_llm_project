---
name: benchmark-research
description: Use when the task requires web research about prior systems, recent scientific writing, datasets, benchmarks, APIs, or evaluation metrics for a project topic. Focus on exact names, URLs, recency, suitability, and comparison value.
---

# Benchmark Research

Use this skill when the agent must gather external evidence before drafting a proposal.

## Workflow

1. Start with 2-4 focused web searches that combine the topic with terms such as:
   - benchmark
   - dataset
   - evaluation
   - baseline
   - prior work
   - paper
   - recent
2. Prioritize sources that help answer:
   - Which comparable systems already exist?
   - Which datasets, APIs, or environments are realistic?
   - Which benchmarks or metrics would make evaluation convincing?
3. Explicitly look for recent scientific writing that can make the proposal more concrete and current.
4. Prefer recent papers, benchmark writeups, and system papers when they are relevant and credible.
5. After search, inspect the most relevant URLs closely enough to extract exact names and short evidence-backed notes.
6. Record source information for anything likely to be cited later:
   - title,
   - source or authoring organization,
   - year if available,
   - URL.
7. Save concise notes to `/workspace/research/research_memo.md` or `/workspace/research/benchmark_notes.md`.
8. Separate hard evidence from assumptions or open questions.

## Output standard

- Include exact benchmark, dataset, or system names.
- Include recent papers or scientific writing when they materially sharpen the proposal.
- Include source information precise enough for a references section later.
- Prefer a short set of strong candidates over a long noisy list.
- If a source is vague, do not overclaim what it proves.

For the target note structure, read `references/research_checklist.md`.
