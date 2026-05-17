"""
Manual Evaluation — Correlation Analysis
Computes % agreement and Cohen's Kappa between manual (human) verdicts
and LLM-judge verdicts for all evaluated queries.

Outputs:
  results_per_query.csv   — per-query/agent breakdown
  results_summary.csv     — overall TP vs BL pooled across all queries
  correlation_plots.png   — bar charts
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT  = os.path.dirname(__file__)

QUERY_MAP = {
    1:  {"BL": "baseline/query_1_couple_citytrip_Adrian",
         "TP": "travel_agent/query_1_couple_citytrip_Adrian"},
    7:  {"BL": "baseline/query_7_business_solo_Adrian",
         "TP": "travel_agent/query_7_business_solo_Adrian"},
    16: {"BL": "baseline/query_16_already_at_destination_Adrian",
         "TP": "travel_agent/query_16_already_at_destination_Adrian"},
}

QUERY_LABELS = {1: "Q1 (couple citytrip)", 7: "Q5 (business solo)", 16: "Q16 (already at dest)"}


def load_judge(q_id, agent_key):
    path = os.path.join(BASE, "data", "evaluation",
                        QUERY_MAP[q_id][agent_key], "scorecard.json")
    with open(path, encoding="utf-8") as f:
        sc = json.load(f)
    return {c["id"]: c["final_verdict"] for c in sc["aggregated_constraints"]}


def encode(v):
    v = str(v).strip().upper().replace(" ", "_")
    if v == "PASS":         return 2
    if v == "MISSING_INFO": return 1
    if v == "FAIL":         return 0
    return None


def kappa_safe(y_m, y_j):
    try:
        if len(set(y_m + y_j)) < 2:
            return float("nan")
        return cohen_kappa_score(y_m, y_j, labels=[0, 1, 2])
    except Exception:
        return float("nan")


def agreement(y_m, y_j):
    if not y_m:
        return 0.0, 0
    pct = sum(a == b for a, b in zip(y_m, y_j)) / len(y_m) * 100
    return pct, len(y_m)


# ── Load CSVs ─────────────────────────────────────────────────────────────────

hc_cc = pd.read_csv(os.path.join(OUT, "manual_eval_hc_cc.csv"))
spots = pd.read_csv(os.path.join(OUT, "manual_eval_spotcheck.csv"))

# Normalise spotcheck verdicts
spots["manual_verdict"] = spots["manual_verdict"].astype(str).str.strip()
spots = spots[spots["manual_verdict"] != "-"]
spots = spots[spots["manual_verdict"].str.lower() != "nan"]


# ── Per-query results ─────────────────────────────────────────────────────────

rows = []

for q_id in [1, 7, 16]:
    q_label = QUERY_LABELS[q_id]

    for agent_key in ["BL", "TP"]:
        judge = load_judge(q_id, agent_key)
        manual_df = hc_cc[(hc_cc["query_id_used"] == q_id) &
                           (hc_cc["agent"] == agent_key)]

        # Constraints
        y_m, y_j = [], []
        for _, row in manual_df.iterrows():
            cid = row["constraint_id"]
            if cid not in judge:
                continue
            m = encode(row["manual_verdict"])
            j = encode(judge[cid])
            if m is None or j is None:
                continue
            y_m.append(m)
            y_j.append(j)

        c_pct, c_n = agreement(y_m, y_j)
        c_kappa     = kappa_safe(y_m, y_j)

        # Error direction: judge=PASS but manual=FAIL/MISSING (judge too lenient = "schöngeredet")
        #                  judge=FAIL/MISSING but manual=PASS   (judge too strict  = "schlechtgeredet")
        lenient = sum(j == 2 and m < 2 for m, j in zip(y_m, y_j))   # judge PASS, human not
        strict  = sum(j < 2 and m == 2 for m, j in zip(y_m, y_j))   # judge not PASS, human PASS

        # Slots
        slot_df = spots[(spots["query_id_used"] == q_id) &
                         (spots["agent"] == agent_key)]
        s_n = len(slot_df)
        diff_flags = slot_df["diff_from_llm_judge"].astype(str).str.strip().str.upper()
        s_agree = (diff_flags == "NO").sum()
        s_pct = s_agree / s_n * 100 if s_n else 0.0

        rows.append({
            "query_id":                q_id,
            "query_label":             q_label,
            "agent":                   "Baseline" if agent_key == "BL" else "TravelAgent",
            "constraint_n":            c_n,
            "constraint_agree_%":      round(c_pct, 1),
            "constraint_kappa":        round(c_kappa, 3) if not np.isnan(c_kappa) else "n/a",
            "judge_lenient (J=P,M!=P)": lenient,
            "judge_strict  (J!=P,M=P)": strict,
            "slot_n":                  s_n,
            "slot_agree_%":            round(s_pct, 1),
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "results_per_query.csv"), index=False)
print("Saved: results_per_query.csv")


# ── Overall pooled results ────────────────────────────────────────────────────

summary_rows = []

for agent_key, agent_label in [("BL", "Baseline"), ("TP", "TravelAgent")]:
    # Pool all constraints
    all_ym, all_yj = [], []
    for q_id in [1, 7, 16]:
        judge = load_judge(q_id, agent_key)
        manual_df = hc_cc[(hc_cc["query_id_used"] == q_id) &
                           (hc_cc["agent"] == agent_key)]
        for _, row in manual_df.iterrows():
            cid = row["constraint_id"]
            if cid not in judge:
                continue
            m = encode(row["manual_verdict"])
            j = encode(judge[cid])
            if m is None or j is None:
                continue
            all_ym.append(m)
            all_yj.append(j)

    c_pct, c_n = agreement(all_ym, all_yj)
    c_kappa     = kappa_safe(all_ym, all_yj)
    all_lenient = sum(j == 2 and m < 2 for m, j in zip(all_ym, all_yj))
    all_strict  = sum(j < 2 and m == 2 for m, j in zip(all_ym, all_yj))

    # Pool all slots
    slot_df = spots[spots["agent"] == agent_key]
    s_n = len(slot_df)
    diff_flags = slot_df["diff_from_llm_judge"].astype(str).str.strip().str.upper()
    s_agree = (diff_flags == "NO").sum()
    s_pct = s_agree / s_n * 100 if s_n else 0.0

    summary_rows.append({
        "agent":                   agent_label,
        "queries":                 "Q1 + Q5 + Q16",
        "constraint_n":            c_n,
        "constraint_agree_%":      round(c_pct, 1),
        "constraint_kappa":        round(c_kappa, 3) if not np.isnan(c_kappa) else "n/a",
        "judge_lenient (J=P,M!=P)": all_lenient,
        "judge_strict  (J!=P,M=P)": all_strict,
        "net_overrating":           all_lenient - all_strict,
        "slot_n":                  s_n,
        "slot_agree_%":            round(s_pct, 1),
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT, "results_summary.csv"), index=False)
print("Saved: results_summary.csv")


# ── Plots ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Manual vs. LLM-Judge Agreement — All Queries", fontsize=14, fontweight="bold")

COLORS = {"Baseline": "#1976d2", "TravelAgent": "#e65100"}
Q_LABELS_SHORT = {1: "Q1", 7: "Q5", 16: "Q16", "overall": "Overall"}

for ax, metric, title in [
    (axes[0], "constraint_agree_%", "Constraint Agreement (HC + CC)"),
    (axes[1], "slot_agree_%",       "Slot Spot-Check Agreement"),
]:
    x_labels, bl_vals, tp_vals = [], [], []

    for q_id in [1, 7, 16]:
        bl_row = df[(df["query_id"] == q_id) & (df["agent"] == "Baseline")]
        tp_row = df[(df["query_id"] == q_id) & (df["agent"] == "TravelAgent")]
        x_labels.append(Q_LABELS_SHORT[q_id])
        bl_vals.append(bl_row[metric].values[0])
        tp_vals.append(tp_row[metric].values[0])

    # Add overall
    x_labels.append("Overall")
    bl_vals.append(df_summary[df_summary["agent"] == "Baseline"][metric].values[0])
    tp_vals.append(df_summary[df_summary["agent"] == "TravelAgent"][metric].values[0])

    x = np.arange(len(x_labels))
    w = 0.35
    b1 = ax.bar(x - w/2, bl_vals, w, label="Baseline",     color=COLORS["Baseline"],     alpha=0.85)
    b2 = ax.bar(x + w/2, tp_vals, w, label="TravelAgent",  color=COLORS["TravelAgent"],  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.axhline(80, color="grey", linestyle="--", linewidth=0.8, label="80% threshold")
    ax.set_ylabel("Agreement (%)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{bar.get_height():.0f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

# ── Plot 3: Error direction (lenient vs strict) per query ─────────────────────

ax3 = axes[2]
lenient_col = "judge_lenient (J=P,M!=P)"
strict_col  = "judge_strict  (J!=P,M=P)"

x_labels, bl_lenient, tp_lenient, bl_strict, tp_strict = [], [], [], [], []
for q_id in [1, 7, 16]:
    bl_row = df[(df["query_id"] == q_id) & (df["agent"] == "Baseline")]
    tp_row = df[(df["query_id"] == q_id) & (df["agent"] == "TravelAgent")]
    x_labels.append(Q_LABELS_SHORT[q_id])
    bl_lenient.append(bl_row[lenient_col].values[0])
    tp_lenient.append(tp_row[lenient_col].values[0])
    bl_strict.append(bl_row[strict_col].values[0])
    tp_strict.append(tp_row[strict_col].values[0])

x_labels.append("Overall")
bl_lenient.append(int(df_summary[df_summary["agent"] == "Baseline"][lenient_col].values[0]))
tp_lenient.append(int(df_summary[df_summary["agent"] == "TravelAgent"][lenient_col].values[0]))
bl_strict.append(int(df_summary[df_summary["agent"] == "Baseline"][strict_col].values[0]))
tp_strict.append(int(df_summary[df_summary["agent"] == "TravelAgent"][strict_col].values[0]))

x = np.arange(len(x_labels))
w = 0.2

ax3.bar(x - 1.5*w, bl_lenient, w,
        label="Baseline — Judge too lenient\n(real problem was missed)",
        color="#ef9a9a", edgecolor="#c62828")
ax3.bar(x - 0.5*w, tp_lenient, w,
        label="TravelAgent — Judge too lenient\n(real problem was missed)",
        color="#ffcc80", edgecolor="#e65100")
ax3.bar(x + 0.5*w, bl_strict,  w,
        label="Baseline — Judge too strict\n(plan was unfairly penalised)",
        color="#90caf9", edgecolor="#1565c0")
ax3.bar(x + 1.5*w, tp_strict,  w,
        label="TravelAgent — Judge too strict\n(plan was unfairly penalised)",
        color="#a5d6a7", edgecolor="#2e7d32")

for i, vals in enumerate([bl_lenient, tp_lenient, bl_strict, tp_strict]):
    offset = [-1.5, -0.5, 0.5, 1.5][i]
    for xi, v in enumerate(vals):
        if v > 0:
            ax3.text(xi + offset * w, v + 0.1, str(v),
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

ax3.set_xticks(x)
ax3.set_xticklabels(x_labels, fontsize=10)
ax3.set_ylabel("Number of constraints")
ax3.set_title("Judge Bias Direction\nRed/orange = judge missed real problems  |  Blue/green = judge unfairly penalised",
              fontsize=10, fontweight="bold")
ax3.legend(fontsize=7, loc="upper right", framealpha=0.9)

plt.tight_layout()
plot_path = os.path.join(OUT, "correlation_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print("Saved: correlation_plots.png")


# ── Print summary to console ──────────────────────────────────────────────────

print("\n=== Per-Query Results ===")
print(df.to_string(index=False))
print("\n=== Overall (pooled across all queries) ===")
print(df_summary.to_string(index=False))
print()
print("  judge_lenient: judge=PASS but you said FAIL/MISSING_INFO -> judge missed a real problem -> system looks better than it is")
print("  judge_strict:  judge=FAIL/MISSING but you said PASS      -> judge too harsh -> system unfairly penalised")
print("  net_overrating = lenient - strict  (positive = system overrated by judges)")
