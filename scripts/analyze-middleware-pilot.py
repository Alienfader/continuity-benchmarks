#!/usr/bin/env python3
"""
Analyze the output of `scripts/run-middleware-pilot.sh`.

Loads the five condition results from the pilot run, joins them on
`questionId`, and emits:

  1. Per-condition mean cosine + standard deviation
  2. Lift over baseline (Δ + Cohen's d) for each non-baseline condition
  3. Paired Wilcoxon signed-rank test for the headline contrast:
     auto-middleware vs continuity-in-loop
     (does the production delivery shape preserve the simulator's lift?)

Usage:
  python3 scripts/analyze-middleware-pilot.py reports/middleware-pilot/<timestamp>/

Output: prints a summary table to stdout + writes
        <out_base>/analysis.json with raw paired diffs.

Why no scipy: the rest of the bench harness uses pure-Python statistics
(see runners/experimental-gaps-analysis-v2.py) so the pilot stays
runnable on a fresh Python 3.10+ install.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def usage_and_exit() -> None:
    print("usage: analyze-middleware-pilot.py <pilot_output_dir>", file=sys.stderr)
    sys.exit(2)


def load_recall_over_time(pilot_dir: Path) -> dict[str, dict[str, float]]:
    """
    Load recall-over-time output. Returns {condition_name: {qid: score}}.
    The runner writes its `--output` arg + ".json"; we wrote
    "<base>/recall-over-time/result", so the file is at result.json.
    """
    candidate = pilot_dir / "recall-over-time" / "result.json"
    if not candidate.exists():
        raise FileNotFoundError(f"missing {candidate}")
    raw = json.loads(candidate.read_text())
    out: dict[str, dict[str, float]] = {}
    for condition in raw["conditions"]:
        scores: dict[str, float] = {}
        for entry in condition["perQuestion"]:
            qid = entry["questionId"]
            score = float(entry["score"])
            scores[qid] = score
        out[condition["condition"]] = scores
    return out


def load_middleware_replay(pilot_dir: Path, mode: str) -> dict[str, float]:
    """
    Load a middleware-replay output for one mode.
    Returns {qid: cosineVsGroundTruth}.
    """
    candidate = pilot_dir / f"middleware-replay-{mode}" / "middleware-replay.json"
    if not candidate.exists():
        raise FileNotFoundError(f"missing {candidate}")
    raw = json.loads(candidate.read_text())
    return {q["questionId"]: float(q["cosineVsGroundTruth"]) for q in raw["questions"]}


def mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    s2 = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(s2)


def cohens_d_paired(diffs: list[float]) -> float:
    """Cohen's d for paired samples = mean(diffs) / sd(diffs)."""
    if len(diffs) < 2:
        return float("nan")
    sd = stdev(diffs)
    if sd == 0:
        return float("nan")
    return mean(diffs) / sd


def wilcoxon_signed_rank_paired(diffs: list[float]) -> tuple[float, float]:
    """
    Paired Wilcoxon signed-rank test (normal-approximation p-value).
    Returns (W_statistic, p_value_two_sided).

    Drops zero diffs (Pratt-style? — actually just drops; matches v2 analysis).
    """
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return (0.0, 1.0)
    ranks = sorted(range(n), key=lambda i: abs(nonzero[i]))
    # Average-rank ties on |diff|.
    abs_diffs = [abs(nonzero[i]) for i in ranks]
    rank_values = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1] == abs_diffs[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-indexed
        for k in range(i, j + 1):
            rank_values[k] = avg_rank
        i = j + 1
    w_plus = sum(
        rank_values[k] for k, idx in enumerate(ranks) if nonzero[idx] > 0
    )
    w_minus = sum(
        rank_values[k] for k, idx in enumerate(ranks) if nonzero[idx] < 0
    )
    w = min(w_plus, w_minus)
    # Normal approximation
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return (w, 1.0)
    z = (w - mu) / sigma
    # Two-sided p via standard normal CDF (no scipy)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return (w, p)


def fmt(x: float, n: int = 4) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.{n}f}"


def main() -> None:
    if len(sys.argv) != 2:
        usage_and_exit()
    pilot_dir = Path(sys.argv[1]).resolve()
    if not pilot_dir.is_dir():
        print(f"not a directory: {pilot_dir}", file=sys.stderr)
        sys.exit(1)

    # Load all 5 conditions, each as {qid: score}.
    rot = load_recall_over_time(pilot_dir)
    by_condition: dict[str, dict[str, float]] = {
        "baseline": rot["baseline"],
        "continuity-in-loop (sim)": rot["continuity-in-loop"],
    }
    for mode in ("mcp-search", "agent-loop", "auto-middleware"):
        by_condition[f"mcp-{mode}"] = load_middleware_replay(pilot_dir, mode)

    # Pair on questionId.
    qids = sorted(set.intersection(*[set(d.keys()) for d in by_condition.values()]))
    if not qids:
        print("no overlapping question IDs across conditions", file=sys.stderr)
        sys.exit(1)
    n = len(qids)

    # Per-condition summary.
    summary: dict[str, dict[str, float]] = {}
    for cname, scores in by_condition.items():
        xs = [scores[q] for q in qids]
        summary[cname] = {
            "mean": mean(xs),
            "stdev": stdev(xs),
            "n": float(n),
        }

    # Lift over baseline + paired d.
    baseline_scores = [by_condition["baseline"][q] for q in qids]
    lifts: dict[str, dict[str, float]] = {}
    for cname, scores in by_condition.items():
        if cname == "baseline":
            continue
        xs = [scores[q] for q in qids]
        diffs = [xs[i] - baseline_scores[i] for i in range(n)]
        lifts[cname] = {
            "mean_delta": mean(diffs),
            "cohens_d": cohens_d_paired(diffs),
        }

    # Headline contrast: auto-middleware vs continuity-in-loop (sim).
    sim_scores = [by_condition["continuity-in-loop (sim)"][q] for q in qids]
    amw_scores = [by_condition["mcp-auto-middleware"][q] for q in qids]
    amw_vs_sim_diffs = [amw_scores[i] - sim_scores[i] for i in range(n)]
    w, p = wilcoxon_signed_rank_paired(amw_vs_sim_diffs)
    amw_vs_sim = {
        "mean_delta": mean(amw_vs_sim_diffs),
        "cohens_d": cohens_d_paired(amw_vs_sim_diffs),
        "wilcoxon_w": w,
        "wilcoxon_p_two_sided": p,
        "n_pairs": n,
    }

    # Print.
    print()
    print("=" * 70)
    print(f"Middleware-replay pilot — analysis ({n} questions paired)")
    print("=" * 70)
    print()
    print("Per-condition mean cosine vs ground truth:")
    print(f"  {'condition':<28} {'mean':>8} {'sd':>8}")
    for cname, s in summary.items():
        print(f"  {cname:<28} {fmt(s['mean']):>8} {fmt(s['stdev']):>8}")
    print()
    print("Lift over baseline:")
    print(f"  {'condition':<28} {'Δ mean':>8} {'Cohen d':>8}")
    for cname, l in lifts.items():
        print(f"  {cname:<28} {fmt(l['mean_delta']):>8} {fmt(l['cohens_d']):>8}")
    print()
    print(
        "Headline contrast — auto-middleware vs continuity-in-loop (sim):"
    )
    print(f"  Δ mean (real − sim): {fmt(amw_vs_sim['mean_delta'])}")
    print(f"  Cohen's d (paired):  {fmt(amw_vs_sim['cohens_d'])}")
    print(
        f"  Wilcoxon W = {amw_vs_sim['wilcoxon_w']:.1f}, "
        f"p (two-sided, normal approx) = {fmt(amw_vs_sim['wilcoxon_p_two_sided'])}, "
        f"n = {n}"
    )
    print()
    if abs(amw_vs_sim["mean_delta"]) < 0.02 and amw_vs_sim["wilcoxon_p_two_sided"] > 0.1:
        print(
            "  → Reading: production delivery shape (real middleware) "
            "is consistent with the §4.7 in-loop simulator within pilot noise."
        )
    elif amw_vs_sim["mean_delta"] > 0.02:
        print(
            "  → Reading: production delivery shape DELIVERS a measurable "
            "lift over the simulator. The simulator under-counts the production benefit."
        )
    else:
        print(
            "  → Reading: production delivery shape DELIVERS LESS than "
            "the simulator. The §4.7 numbers may overstate the production lift."
        )
    print()

    # Write JSON.
    out_path = pilot_dir / "analysis.json"
    out_path.write_text(
        json.dumps(
            {
                "n_questions": n,
                "questionIds": qids,
                "by_condition": summary,
                "lift_over_baseline": lifts,
                "auto_middleware_vs_in_loop_sim": amw_vs_sim,
                "raw_paired_diffs_amw_vs_sim": amw_vs_sim_diffs,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  analysis.json written → {out_path}")
    print()


if __name__ == "__main__":
    main()
