from travelplanner.schema.judge_artifact import ScorecardModel


def print_scorecard(scorecard: ScorecardModel, label: str) -> None:
    if isinstance(scorecard, dict):
        scorecard = ScorecardModel.model_validate(scorecard)
    print("=" * 64)
    print(f"SCORECARD — {label}")
    print("=" * 64)
    n_rv = len(scorecard.rationale_verifications)
    rv_summary = (
        f"{scorecard.rationale_pass_count} PASS, "
        f"{scorecard.rationale_fail_count} FAIL, "
        f"{scorecard.rationale_missing_count} MISSING_INFO"
        f"  (of {n_rv} slot{'s' if n_rv != 1 else ''})"
    ) if n_rv else "no slots verified"
    print(f"Rationale verification : {rv_summary}")
    print(f"HC  Micro Pass Rate    : {scorecard.hc_micro_pass_rate:.1%}")
    print(f"CC  Micro Pass Rate    : {scorecard.cc_micro_pass_rate:.1%}")
    print(f"HC  Macro Pass Rate    : {scorecard.hc_macro_pass_rate:.0%}")
    print(f"CC  Macro Pass Rate    : {scorecard.cc_macro_pass_rate:.0%}")
    print(f"Final Pass Rate        : {scorecard.final_pass_rate:.0%}")
    print()

    if scorecard.rationale_verifications:
        print("Rationale verification detail:")
        for rv in scorecard.rationale_verifications:
            tag = f"[{rv.verdict:12s}]"
            print(f"  {tag} Day {rv.day_index}/slot {rv.slot_position} — {rv.slot_name}  "
                  f"(source: {rv.source_type})")
            if rv.verdict != "PASS" and rv.reasoning:
                print(f"               reason: {rv.reasoning[:120]}")
        print()

    short_names = [m.split("/")[-1] for m in scorecard.judge_models]
    print("Per-constraint verdicts:")
    for c in scorecard.aggregated_constraints:
        votes = " | ".join(f"{name}={v}" for name, v in zip(short_names, c.judge_verdicts))
        print(f"  [{c.final_verdict:12s}] {c.id}: {c.constraint_text[:65]}")
        print(f"  {' ' * 15} {votes}")
    print()
    print(f"Judges    : {', '.join(scorecard.judge_models)}")
    print(f"Timestamp : {scorecard.timestamp}")
    print()