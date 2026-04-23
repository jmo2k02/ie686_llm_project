---
name: proposal-writer
description: Use when turning research notes into a structured project outline or proposal draft. Writes a roughly three-page executive summary in prose, grounded in evidence, with a concrete multi-agent workflow.
---

# Proposal Writer

Use this skill after research notes exist or when a topic must be turned into a polished outline.

## Workflow

1. Read the project-outline rubric skill and the latest research notes in `/workspace/`.
2. Extract only evidence-backed datasets, APIs, benchmarks, prior systems, and recent scientific writing that are actually useful for the draft.
3. Track source information for anything you rely on in the writing:
   - source or authoring organization,
   - title,
   - year if available,
   - URL.
4. Draft a project title and a short opening paragraph that gives context and explains why the topic matters.
5. Write the proposal as an executive summary of roughly three pages:
   - start with context and the core problem,
   - explain what the team wants to build,
   - explain how the team plans to tackle the problem,
   - close with a concrete evaluation plan.
6. Add source information and a short references section that reflects the sources actually used for the writing.
7. Write the outline to `/workspace/outline/project_outline.md` during drafting and `/workspace/final/final_proposal.md` for the final revision.
8. If the evidence is thin, explicitly label assumptions and open questions.

## Writing constraints

- Prefer specific nouns over vague language.
- Do not promise implementation details that are unsupported by the research notes.
- The main body should be prose, not a bullet list.
- Use short section headings if useful, but keep each section paragraph-based.
- Make the workflow concrete enough to evaluate, but describe it in readable prose.
- Include a short references section at the end based only on the sources actually used.

For the preferred section order, read `references/output_template.md`.
