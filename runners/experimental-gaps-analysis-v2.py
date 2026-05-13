#!/usr/bin/env python3
"""
experimental-gaps-analysis-v2.py — analysis for the v2 cross-corpus matrix
with the corrected runners (entity-keyed in-loop on action-alignment + 4-
condition recall with continuity-perq-frontloaded ablation).

Differs from experimental-gaps-analysis.py (v1, May 5):
  - Reads from benchmarks/reports/id-rag-parity-v2/ (not id-rag-parity/)
  - Recognizes 4 recall conditions: baseline, continuity-blanket,
    continuity-perq-frontloaded, continuity-in-loop
  - Recognizes 3 alignment conditions: baseline, continuity,
    continuity-in-loop (now genuinely different — entity-keyed in-loop)
  - Computes the three new pairwise contrasts:
      baseline → continuity-blanket          (any retrieval)
      continuity-blanket → perq-frontloaded  (keying, timing held)
      perq-frontloaded → continuity-in-loop  (timing, retrieval-data held)
  - Tolerant of missing cells (data-pipeline/gpt-4o/run-2/recall-over-time
    failed with persistent OpenAI 502; reports n_cells per condition)

Outputs:
  benchmarks/reports/id-rag-parity-v2/EXPERIMENTAL_GAPS_ANALYSIS_V2.md
  benchmarks/reports/id-rag-parity-v2/experimental-gaps-analysis-v2.json
"""
import json, math
from pathlib import Path
from collections import defaultdict

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent / "reports" / "id-rag-parity-v2"

FIXTURES = ["data-pipeline", "mobile-app"]
MODELS = ["gpt-4o", "claude-sonnet-4-6"]
RUNS = [1, 2, 3]

# Condition lists per runner
RECALL_CONDITIONS = ["baseline", "continuity-blanket", "continuity-perq-frontloaded", "continuity-in-loop"]
ALIGN_CONDITIONS = ["baseline", "continuity", "continuity-in-loop"]

# ── Load all action-alignment + recall-over-time JSONs (tolerant) ───────────
aa_records = []
rt_per_session = []
rt_drift_slopes = []
missing_cells = []

for fx in FIXTURES:
    for m in MODELS:
        for r in RUNS:
            cell = f"{fx}/{m}/run-{r}"
            aa_path = ROOT / cell / "action-alignment.json"
            rt_path = ROOT / cell / "recall-over-time.json"
            if aa_path.exists():
                aa = json.loads(aa_path.read_text())
                for rec in aa.get("results", []):
                    aa_records.append({
                        "fixture": fx, "model": m, "run": r,
                        "condition": rec["condition"],
                        "actionId": rec["actionId"],
                        "sonnet_score": rec["judgeScore"],
                        "judge": aa.get("judgeModel", "claude-sonnet-4-6"),
                    })
            else:
                missing_cells.append(f"{cell}/action-alignment")
            if rt_path.exists():
                rt = json.loads(rt_path.read_text())
                for cond_block in rt.get("conditions", []):
                    cond = cond_block["condition"]
                    rt_drift_slopes.append({
                        "fixture": fx, "model": m, "run": r,
                        "condition": cond,
                        "meanAcrossSessions": cond_block["meanAcrossSessions"],
                        "driftSlope": cond_block["driftSlope"],
                    })
                    for s in cond_block.get("sessionSummaries", []):
                        rt_per_session.append({
                            "fixture": fx, "model": m, "run": r,
                            "condition": cond,
                            "sessionIdx": s["sessionIdx"],
                            "mean": s["summary"]["mean"],
                            "fractionAbove070": s["summary"]["fractionAbove070"],
                        })
            else:
                missing_cells.append(f"{cell}/recall-over-time")

aa_cells = len({(r['fixture'],r['model'],r['run']) for r in aa_records})
rt_cells = len({(r['fixture'],r['model'],r['run']) for r in rt_drift_slopes})

print(f"Loaded {len(aa_records)} action-alignment records ({aa_cells} cells)")
print(f"Loaded {len(rt_per_session)} recall-over-time session records ({rt_cells} cells)")
if missing_cells:
    print(f"Missing cells: {missing_cells}")

# ════════════════════════════════════════════════════════════════════════════
# Statistical helpers (pure Python, no scipy)
# ════════════════════════════════════════════════════════════════════════════

def mean(xs): return sum(xs)/len(xs) if xs else 0

def cell_means(records, score_field):
    bucket = defaultdict(list)
    for r in records:
        key = (r["fixture"], r["model"], r["run"], r["condition"])
        bucket[key].append(r[score_field])
    return {k: sum(v)/len(v) for k, v in bucket.items()}

def paired_diffs(means_dict, cond_a, cond_b):
    diffs = []
    cells = sorted({(f, m, r) for (f, m, r, c) in means_dict.keys()})
    for f, m, r in cells:
        a = means_dict.get((f, m, r, cond_a))
        b = means_dict.get((f, m, r, cond_b))
        if a is not None and b is not None:
            diffs.append(a - b)
    return diffs

def wilcoxon_signed_rank(diffs):
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"W": 0, "n": 0, "p_two_sided_normal_approx": float("nan"),
                "z": 0, "note": "all zero diffs"}
    abs_diffs = sorted(((abs(d), d) for d in nonzero), key=lambda t: t[0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j+1 < n and abs_diffs[j+1][0] == abs_diffs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j+1):
            ranks[k] = avg
        i = j + 1
    W_pos = sum(ranks[k] for k in range(n) if abs_diffs[k][1] > 0)
    W_neg = sum(ranks[k] for k in range(n) if abs_diffs[k][1] < 0)
    W = min(W_pos, W_neg)
    mu = n * (n+1) / 4
    sigma = math.sqrt(n * (n+1) * (2*n+1) / 24)
    z = (W - mu) / sigma if sigma > 0 else 0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"W": W, "W_pos": W_pos, "W_neg": W_neg, "n": n,
            "z": z, "p_two_sided_normal_approx": p}

def cohen_d_paired(diffs):
    if len(diffs) < 2: return float("nan")
    m = mean(diffs)
    var = sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)
    sd = math.sqrt(var)
    return m / sd if sd > 0 else float("nan")

# ════════════════════════════════════════════════════════════════════════════
# Build cell-level mean dicts for each runner
# ════════════════════════════════════════════════════════════════════════════
aa_cell_means = cell_means(aa_records, "sonnet_score")

# For recall, "score" per cell × condition is `meanAcrossSessions`.
rt_cell_means = {}
for r in rt_drift_slopes:
    key = (r["fixture"], r["model"], r["run"], r["condition"])
    rt_cell_means[key] = r["meanAcrossSessions"]

# ════════════════════════════════════════════════════════════════════════════
# A. ACTION-ALIGNMENT: did the corrected in-loop produce different scores
#    than continuity? (v1 had byte-identical code so this was forced 0.)
# ════════════════════════════════════════════════════════════════════════════
align_pairs = [
    ("continuity", "baseline"),
    ("continuity-in-loop", "baseline"),
    ("continuity-in-loop", "continuity"),
]
align_results = {}
for (a, b) in align_pairs:
    diffs = paired_diffs(aa_cell_means, a, b)
    if not diffs: continue
    w = wilcoxon_signed_rank(diffs)
    d = cohen_d_paired(diffs)
    align_results[f"{a}_vs_{b}"] = {
        "n_pairs": len(diffs),
        "mean_diff": mean(diffs),
        "diffs": diffs,
        "wilcoxon": w,
        "cohen_d_paired": d,
    }

# ════════════════════════════════════════════════════════════════════════════
# B. RECALL-OVER-TIME: 6 pairwise contrasts, with the M2 ablation as
#    contrast #5 (perq-frontloaded vs in-loop holding retrieval-data
#    constant — does fresh re-retrieval add anything?)
# ════════════════════════════════════════════════════════════════════════════
recall_pairs = [
    # Effect of any retrieval (baseline → blanket)
    ("continuity-blanket", "baseline"),
    # Effect of better keying (blanket → perq-frontloaded), timing held
    ("continuity-perq-frontloaded", "continuity-blanket"),
    # Effect of fresh re-retrieval (perq-frontloaded → in-loop) — THE M2 ABLATION
    ("continuity-in-loop", "continuity-perq-frontloaded"),
    # Reference contrasts vs baseline
    ("continuity-perq-frontloaded", "baseline"),
    ("continuity-in-loop", "baseline"),
    # Legacy contrast (for comparison with v1 paper claim)
    ("continuity-in-loop", "continuity-blanket"),
]
recall_results = {}
for (a, b) in recall_pairs:
    diffs = paired_diffs(rt_cell_means, a, b)
    if not diffs: continue
    w = wilcoxon_signed_rank(diffs)
    d = cohen_d_paired(diffs)
    recall_results[f"{a}_vs_{b}"] = {
        "n_pairs": len(diffs),
        "mean_diff": mean(diffs),
        "diffs": diffs,
        "wilcoxon": w,
        "cohen_d_paired": d,
    }

# ════════════════════════════════════════════════════════════════════════════
# C. CEILING CHECK on action alignment under v2 in-loop (reviewer R1)
# ════════════════════════════════════════════════════════════════════════════
ceiling = {}
for cond in ALIGN_CONDITIONS:
    scores = [r["sonnet_score"] for r in aa_records if r["condition"] == cond]
    n = len(scores)
    if n == 0: continue
    bins = {i: 0 for i in range(1, 11)}
    for s in scores:
        if 1 <= s <= 10: bins[s] += 1
    ceiling[cond] = {
        "n": n,
        "mean": sum(scores) / n,
        "fraction_at_10": bins[10] / n,
        "fraction_at_9_or_10": (bins[9] + bins[10]) / n,
        "fraction_at_or_above_8": sum(bins[i] for i in range(8, 11)) / n,
        "fraction_below_5": sum(bins[i] for i in range(1, 5)) / n,
        "histogram": {str(k): v for k, v in bins.items()},
    }

# ════════════════════════════════════════════════════════════════════════════
# D. DRIFT: per-session means by condition (reviewer M3)
# ════════════════════════════════════════════════════════════════════════════
def session_mean(records, sessionIdx, condition):
    xs = [r["mean"] for r in records
          if r["sessionIdx"] == sessionIdx and r["condition"] == condition]
    return sum(xs)/len(xs) if xs else float("nan")

drift = {"session_means": {}, "drift_slopes": {}}
for cond in RECALL_CONDITIONS:
    s_means = [session_mean(rt_per_session, s, cond) for s in range(7)]
    if not any(s_means) or all(math.isnan(x) for x in s_means): continue
    drift["session_means"][cond] = {
        "session_1": s_means[0],
        "session_7": s_means[6],
        "all_sessions": s_means,
        "session1_minus_session7": s_means[0] - s_means[6] if not math.isnan(s_means[0]) and not math.isnan(s_means[6]) else float("nan"),
    }
    slopes = [r["driftSlope"] for r in rt_drift_slopes if r["condition"] == cond]
    drift["drift_slopes"][cond] = {
        "mean_slope_per_session": sum(slopes)/len(slopes) if slopes else float("nan"),
        "n_cells": len(slopes),
    }

# ════════════════════════════════════════════════════════════════════════════
# E. SELF-JUDGING SPLIT (reviewer M1) — on action-alignment scores
# ════════════════════════════════════════════════════════════════════════════
m1_split = {}
for cond in ALIGN_CONDITIONS:
    sonnet_self = [r["sonnet_score"] for r in aa_records
                   if r["condition"] == cond and r["model"] == "claude-sonnet-4-6"]
    sonnet_xfer = [r["sonnet_score"] for r in aa_records
                   if r["condition"] == cond and r["model"] == "gpt-4o"]
    if not sonnet_self or not sonnet_xfer: continue
    m1_split[cond] = {
        "sonnet_judges_sonnet": mean(sonnet_self),
        "sonnet_judges_gpt4o": mean(sonnet_xfer),
        "sonnet_self_minus_xfer": mean(sonnet_self) - mean(sonnet_xfer),
        "n_sonnet_self": len(sonnet_self),
        "n_sonnet_xfer": len(sonnet_xfer),
    }

# ════════════════════════════════════════════════════════════════════════════
# Save JSON + write markdown
# ════════════════════════════════════════════════════════════════════════════
out = {
    "generated_at": "2026-05-07",
    "scope": "v2 cross-corpus matrix; corrected runners (entity-keyed in-loop on action-alignment + 4-condition recall with M2 ablation)",
    "missing_cells": missing_cells,
    "alignment_results": align_results,
    "recall_results": recall_results,
    "ceiling": ceiling,
    "drift": drift,
    "M1_self_judging": m1_split,
}
(ROOT / "experimental-gaps-analysis-v2.json").write_text(json.dumps(out, indent=2))
print(f"\nJSON saved to {ROOT / 'experimental-gaps-analysis-v2.json'}")

# ── Markdown ────────────────────────────────────────────────────────────────
def fmt_p(p):
    if math.isnan(p): return "p = nan"
    if p < 0.001: return "p < 0.001"
    if p < 0.01: return f"p = {p:.3f}"
    return f"p = {p:.3f}"

md = []
md.append("# v2 Cross-Corpus Matrix — Analysis (Corrected Runners + M2 Ablation)")
md.append("")
md.append("**Generated:** 2026-05-07. Source: `benchmarks/reports/id-rag-parity-v2/`.")
md.append("")
md.append(f"**Coverage:** {aa_cells}/12 action-alignment cells, {rt_cells}/12 recall-over-time cells.")
if missing_cells:
    md.append(f"**Missing:** `{', '.join(missing_cells)}` — persistent OpenAI 502 on this single cell across both initial run + resume retry.")
md.append("")
md.append("This is the v2 analysis that follows from the runner fixes in commit `ee3b4e9e`. The two methodological flaws surfaced by the May 2026 stress-test review have been resolved at the runner level:")
md.append("")
md.append("1. **action-alignment.ts** — `continuity-in-loop` now does entity-keyed retrieval (mirrors production middleware's file-path extraction), distinct from `continuity`'s full-prompt retrieval. v1 had these as byte-identical code.")
md.append("2. **recall-over-time.ts** — adds a fourth condition `continuity-perq-frontloaded` that holds retrieval data constant while varying injection timing, isolating M2 (injection-timing ablation) cleanly.")
md.append("")
md.append("---")
md.append("")

# Action-alignment results
md.append("## A. Action-alignment — does the corrected in-loop differ from continuity?")
md.append("")
md.append("**v1's finding:** continuity ≈ continuity-in-loop (mean Δ = 0.00, p = 0.68). This was an artifact: the two conditions executed identical code.")
md.append("**v2's question:** with entity-keyed retrieval as the in-loop variant, do scores diverge?")
md.append("")
md.append("| Condition pair | n | Mean Δ | W | z | p | Cohen's d |")
md.append("|---|---|---|---|---|---|---|")
for key, r in align_results.items():
    label = key.replace("_vs_", " vs ")
    w = r["wilcoxon"]
    md.append(f"| {label} | {r['n_pairs']} | **{r['mean_diff']:+.2f}** | {w['W']:.1f} | {w['z']:.2f} | **{fmt_p(w['p_two_sided_normal_approx'])}** | {r['cohen_d_paired']:.2f} |")
md.append("")

# Determine the in-loop vs continuity verdict
if "continuity-in-loop_vs_continuity" in align_results:
    inloop_vs_cont = align_results["continuity-in-loop_vs_continuity"]
    p = inloop_vs_cont["wilcoxon"]["p_two_sided_normal_approx"]
    delta = inloop_vs_cont["mean_diff"]
    if p < 0.05 and delta > 0:
        md.append(f"**Verdict:** in-loop (entity-keyed) > continuity (full-prompt) by {delta:+.2f} mean cell points, {fmt_p(p)}. Entity extraction surfaces decisions the full-prompt query missed.")
    elif p < 0.05 and delta < 0:
        md.append(f"**Verdict:** in-loop (entity-keyed) < continuity (full-prompt) by {delta:+.2f} mean cell points, {fmt_p(p)}. Full-prompt query retrieves more relevant context than narrow entity extraction.")
    else:
        md.append(f"**Verdict:** no significant difference between in-loop and continuity (mean Δ = {delta:+.2f}, {fmt_p(p)}). Likely ceiling effect — see section C.")
md.append("")

# Recall results
md.append("## B. Recall-over-time — three pairwise contrasts (the M2 ablation lives in contrast #3)")
md.append("")
md.append("| Contrast | What it isolates | n | Mean Δ | W | p | Cohen's d |")
md.append("|---|---|---|---|---|---|---|")
labels_for = {
    "continuity-blanket_vs_baseline": "Effect of any retrieval",
    "continuity-perq-frontloaded_vs_continuity-blanket": "Better keying (timing held)",
    "continuity-in-loop_vs_continuity-perq-frontloaded": "**M2 ablation: timing only**",
    "continuity-perq-frontloaded_vs_baseline": "(reference) perq vs baseline",
    "continuity-in-loop_vs_baseline": "(reference) in-loop vs baseline",
    "continuity-in-loop_vs_continuity-blanket": "(legacy v1 paper contrast)",
}
for key in labels_for:
    if key not in recall_results: continue
    r = recall_results[key]
    w = r["wilcoxon"]
    md.append(f"| {key.replace('_vs_', ' vs ')} | {labels_for[key]} | {r['n_pairs']} | **{r['mean_diff']:+.3f}** | {w['W']:.1f} | **{fmt_p(w['p_two_sided_normal_approx'])}** | {r['cohen_d_paired']:.2f} |")
md.append("")

# M2 verdict
if "continuity-in-loop_vs_continuity-perq-frontloaded" in recall_results:
    m2 = recall_results["continuity-in-loop_vs_continuity-perq-frontloaded"]
    p = m2["wilcoxon"]["p_two_sided_normal_approx"]
    delta = m2["mean_diff"]
    md.append(f"### M2 verdict: does fresh re-retrieval per session add anything beyond the same retrieval frozen at session 1?")
    md.append("")
    if p < 0.05 and delta > 0.005:
        md.append(f"**Yes — timing matters.** Fresh per-session re-retrieval beats frozen-at-session-1 by **{delta:+.3f} cosine units, {fmt_p(p)}** (Cohen's d = {m2['cohen_d_paired']:.2f}). The 'in-loop pattern is the contribution' claim is supported: re-firing under noise produces real gain.")
    elif p < 0.05 and delta < -0.005:
        md.append(f"**Inverted — frozen-at-session-1 beats in-loop by {-delta:+.3f}, {fmt_p(p)}.** Counterintuitive result; possibly retrieval drift from semantic noise. Worth investigating.")
    elif p >= 0.05 and abs(delta) < 0.01:
        md.append(f"**No — timing does not matter.** Fresh re-retrieval and frozen-at-session-1 score the same (mean Δ = {delta:+.3f}, {fmt_p(p)}). The contribution is the *retrieval keying*, not the *injection timing*. The in-loop pattern's defense reduces to 'Continuity has good retrieval' — same as passive RAG with the same index.")
    else:
        md.append(f"**Inconclusive** — Δ = {delta:+.3f}, {fmt_p(p)}, Cohen's d = {m2['cohen_d_paired']:.2f}. Either small effect or low statistical power.")
md.append("")

# Ceiling check
md.append("## C. Ceiling effect on action alignment (R1)")
md.append("")
md.append("v1 saw ~80% of continuity scores at 9-10 (Sonnet) / ~99% (Gemini). v2's entity-keyed in-loop:")
md.append("")
md.append("| Condition | n | Mean | % at 10 | % at 9 or 10 | % ≥ 8 | % below 5 |")
md.append("|---|---|---|---|---|---|---|")
for cond in ALIGN_CONDITIONS:
    if cond not in ceiling: continue
    r = ceiling[cond]
    md.append(f"| {cond} | {r['n']} | {r['mean']:.2f} | {100*r['fraction_at_10']:.1f}% | {100*r['fraction_at_9_or_10']:.1f}% | {100*r['fraction_at_or_above_8']:.1f}% | {100*r['fraction_below_5']:.1f}% |")
md.append("")

# Drift
md.append("## D. Drift across the 7-session window (M3)")
md.append("")
md.append("| Condition | Session 1 | Session 7 | Δ (S1 − S7) | Mean drift slope |")
md.append("|---|---|---|---|---|")
for cond in RECALL_CONDITIONS:
    if cond not in drift["session_means"]: continue
    sm = drift["session_means"][cond]
    ds = drift["drift_slopes"][cond]
    md.append(f"| {cond} | {sm['session_1']:.3f} | {sm['session_7']:.3f} | {sm['session1_minus_session7']:+.3f} | {ds['mean_slope_per_session']:+.4f} |")
md.append("")

# M1
md.append("## E. Self-judging split (M1)")
md.append("")
md.append("| Condition | Sonnet judges Sonnet-agent | Sonnet judges GPT-4o-agent | Δ (self − xfer) | n per split |")
md.append("|---|---|---|---|---|")
for cond in ALIGN_CONDITIONS:
    if cond not in m1_split: continue
    r = m1_split[cond]
    md.append(f"| {cond} | {r['sonnet_judges_sonnet']:.2f} | {r['sonnet_judges_gpt4o']:.2f} | {r['sonnet_self_minus_xfer']:+.2f} | {r['n_sonnet_self']} |")
md.append("")

md.append("---")
md.append("")
md.append("## Outstanding work")
md.append("")
md.append("- **Inter-judge replication on v2 outputs.** The v2 action-alignment.json files have not yet been re-judged with Gemini. Existing inter-judge data (`inter-judge-cross-corpus.json` in id-rag-parity/, n=1080) is on v1 outputs and is no longer directly applicable to v2 numbers. Re-running re-judge-cross-corpus.py against v2 paths is the next step (~2h, ~$2).")
if missing_cells:
    md.append(f"- **Persistent 502 cell.** {missing_cells[0]} failed twice with OpenAI 502. The other 23 cells provide robust coverage; the missing recall datapoint affects 1 of 12 paired observations in the keying/timing contrasts.")
md.append("")

(ROOT / "EXPERIMENTAL_GAPS_ANALYSIS_V2.md").write_text("\n".join(md))
print(f"Markdown saved to {ROOT / 'EXPERIMENTAL_GAPS_ANALYSIS_V2.md'}")
