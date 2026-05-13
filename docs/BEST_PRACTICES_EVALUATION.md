# LLM-as-a-Judge Best Practices

**Sources**: Zheng et al. (MT-Bench, NeurIPS 2023), Li et al. (multi-LLM survey), Xie et al. (TravelPlanner 2024), IE685 Lecture §3 Prompt Engineering, IE685 Lecture §8 Security & Safety.

---

1. **Temperature = 0 for all judges.** Zheng et al.: deterministic output is the single most important reproducibility lever. Non-zero temperature adds noise without improving reliability.

2. **Rationale before verdict (chain-of-thought first).** IE685 §3: chain-of-thought reasoning before the score prevents the score from anchoring the reasoning. Every judge must output step-by-step reasoning for each constraint before committing to PASS / FAIL.

3. **Binary PASS / FAIL per constraint; no ordinal scale.** Constraint compliance is inherently binary. Ordinal scales increase inter-rater disagreement and position bias (Zheng et al.). N/A is allowed only when a hard-constraint category was absent from the original query.

4. **Exclude the plan-generation model family from the judge pool (no self-enhancement).** Zheng et al. Fig. 6–8: GPT-4 inflates scores for GPT-4 outputs. Li et al. CALM: heterogeneous judge families reduce inter-judge correlation. The travel planner uses OpenAI GPT — judges must come from Anthropic, Google, Mistral, and Meta families only.

5. **Pure independent judging — no cross-judge discussion.** Li et al.: panel-with-discussion inflates agreement without improving accuracy. Each judge receives only (plan, constraints, query, tavily_evidence) and produces its own verdict. Judges never see each other's output.

6. **N = 4 judges with majority vote; ties → FAIL.** Li et al.: four judges yield a clear majority (3–1) or a tie (2–2). Conservative tie-breaking toward FAIL prevents inflated pass rates.

7. **N/A constraints excluded from the denominator of all metrics.** Xie et al. Eq. 1: Micro Pass Rate = satisfied / applicable. N/A constraints neither pass nor fail; they are removed from the denominator. A hard constraint absent from the user query must be N/A, not PASS.

8. **Report both Micro Pass Rate and Macro Pass Rate.** Xie et al. Eq. 1–2: Micro = per-constraint fraction (informative during development); Macro = fraction of fully compliant plans (headline metric). Report both, plus Final Pass Rate (HC macro ∧ CC macro).

9. **Ground all factual claims via Tavily before judging.** IE685 §8: restrict factual assertions to externally verified knowledge. Use Tavily to retrieve real-world price ranges and geographic facts, and inject them into the judge prompt. This is the primary mechanism for detecting hallucinated prices and impossible city placements.

10. **MISSING INFO is a valid verdict — never guess.** IE685 §8: "I don't know" allowance. If the plan omits information needed to evaluate a constraint, record MISSING INFO (counted as FAIL for scoring). Never assume missing information is acceptable.

11. **Explicit anti-verbosity instruction in every judge prompt.** IE685 §3: verbose plans must not score higher than terse ones. The judge system prompt must state: "Do not reward plans for being detailed, well-written, or long."

12. **Immutable inputs — each judge sees an identical, frozen snapshot.** No shared mutable state between judge invocations. Pass (plan_text, hard_constraints, commonsense_constraints, query, tavily_evidence) as read-only inputs; never let one judge's output influence another's input.

13. **Validate judge output with Pydantic; retry on schema failure (max 2 retries).** Use `invoke_structured_model` from `utils/llm.py`. A judge that cannot conform to the output schema after 2 retries contributes FAIL for all its constraints (conservative default).

14. **Store raw judge responses as JSON Lines for audit.** Aggregated verdicts are the product; raw responses are the audit trail. Both must be persisted so bias patterns (verbosity, position) can be analysed post-hoc.

15. **Use Micro Pass Rate as the iteration signal, Macro as the acceptance gate.** Xie et al.: Macro Pass Rate is near-zero for early-stage plans and gives no gradient signal. Micro Pass Rate reveals which specific constraints are failing and guides iterative improvement.
