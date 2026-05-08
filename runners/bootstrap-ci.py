#!/usr/bin/env python3
"""
bootstrap-ci.py — BCa 95% bootstrap confidence intervals on Cohen's d
for every paired contrast in the v2 cross-corpus matrix.

Reads:
  benchmarks/reports/id-rag-parity-v2/experimental-gaps-analysis-v2.json

Writes:
  benchmarks/reports/id-rag-parity-v2/bootstrap-ci.json

Pure Python, no scipy — matches the inline-stat-helpers pattern in
experimental-gaps-analysis-v2.py. 10,000 resamples, BCa method.

Method (Efron & Tibshirani 1993):
  1. Compute d on the original n=12 paired diffs.
  2. Resample diffs with replacement B=10,000 times; compute d* per resample.
  3. Bias-correction z0 = Phi^-1( fraction of d* < d ).
  4. Acceleration a from jackknife on d (leave-one-out).
  5. Adjusted percentiles a1, a2 from BCa formula; CI = (d*_(a1), d*_(a2)).
"""
import json, math, random
from pathlib import Path

# ── Path resolution (mirrors experimental-gaps-analysis-v2.py) ─────────────
def _resolve(candidates):
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

_HERE = Path(__file__).resolve().parent
ROOT = _resolve([
    _HERE.parents[3] / "benchmarks" / "reports" / "id-rag-parity-v2",
    _HERE.parent / "reports" / "id-rag-parity-v2",
])

INPUT = ROOT / "experimental-gaps-analysis-v2.json"
OUTPUT = ROOT / "bootstrap-ci.json"

# ── Stat helpers (pure Python, match v2 analysis script) ───────────────────
def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def cohen_d_paired(diffs):
    if len(diffs) < 2:
        return float("nan")
    m = mean(diffs)
    var = sum((d - m) ** 2 for d in diffs) / (len(diffs) - 1)
    sd = math.sqrt(var)
    return m / sd if sd > 0 else float("nan")

def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# Inverse normal CDF (Beasley-Springer-Moro approximation, accurate to ~1e-9)
def normal_ppf(p):
    if p <= 0.0 or p >= 1.0:
        if p == 0.0:
            return -float("inf")
        if p == 1.0:
            return float("inf")
        raise ValueError("normal_ppf: p out of (0,1)")
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d_ = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d_[0]*q + d_[1])*q + d_[2])*q + d_[3])*q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    q = math.sqrt(-2.0 * math.log(1 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d_[0]*q + d_[1])*q + d_[2])*q + d_[3])*q + 1.0)

# ── BCa bootstrap on Cohen's d ─────────────────────────────────────────────
def bca_ci(diffs, n_resamples=10000, alpha=0.05, seed=20260507):
    """
    Returns (d_hat, ci_lo, ci_hi). BCa method per Efron & Tibshirani (1993),
    chapter 14. Pure Python; uses random.choices for resampling.
    """
    n = len(diffs)
    d_hat = cohen_d_paired(diffs)
    if not math.isfinite(d_hat) or n < 2:
        return d_hat, float("nan"), float("nan")

    rng = random.Random(seed)
    # 1) bootstrap distribution of d*
    boot = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        d_star = cohen_d_paired(sample)
        if math.isfinite(d_star):
            boot.append(d_star)
    boot.sort()
    if len(boot) < 100:
        return d_hat, float("nan"), float("nan")

    # 2) bias correction z0 from fraction of bootstrap d* < observed d
    less = sum(1 for x in boot if x < d_hat)
    p0 = less / len(boot)
    p0 = min(max(p0, 1e-9), 1 - 1e-9)
    z0 = normal_ppf(p0)

    # 3) acceleration a from jackknife
    jk = []
    for i in range(n):
        leave_one = diffs[:i] + diffs[i+1:]
        jk.append(cohen_d_paired(leave_one))
    jk_mean = mean(jk)
    num = sum((jk_mean - x) ** 3 for x in jk)
    den = 6.0 * (sum((jk_mean - x) ** 2 for x in jk) ** 1.5)
    a = num / den if den > 0 else 0.0

    # 4) adjusted percentiles
    z_lo = normal_ppf(alpha / 2)
    z_hi = normal_ppf(1 - alpha / 2)
    def adj(z_alpha):
        denom = 1 - a * (z0 + z_alpha)
        if denom == 0:
            denom = 1e-12
        return normal_cdf(z0 + (z0 + z_alpha) / denom)
    a1 = adj(z_lo)
    a2 = adj(z_hi)
    a1 = min(max(a1, 0.0), 1.0)
    a2 = min(max(a2, 0.0), 1.0)

    # 5) percentile lookup on the bootstrap distribution
    def pct(p):
        idx = int(round(p * (len(boot) - 1)))
        idx = min(max(idx, 0), len(boot) - 1)
        return boot[idx]
    return d_hat, pct(a1), pct(a2)

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    if not INPUT.exists():
        raise SystemExit(f"missing input: {INPUT}")
    raw = json.loads(INPUT.read_text())

    contrasts_out = {}
    # Section prefix prevents collision (continuity-in-loop_vs_baseline
    # appears in both recall_results and alignment_results).
    section_prefix = {"recall_results": "recall::", "alignment_results": "alignment::"}
    for section in ("recall_results", "alignment_results"):
        for name, payload in raw.get(section, {}).items():
            diffs = payload.get("diffs") or []
            if not diffs:
                continue
            d_hat, lo, hi = bca_ci(diffs, n_resamples=10000)
            contrasts_out[section_prefix[section] + name] = {
                "section": section,
                "n_pairs": len(diffs),
                "d": d_hat,
                "d_ci_lo": lo,
                "d_ci_hi": hi,
                "diffs": diffs,
            }

    out = {
        "n_resamples": 10000,
        "method": "BCa",
        "alpha": 0.05,
        "seed": 20260507,
        "contrasts": contrasts_out,
    }
    OUTPUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUTPUT}")

    # ── stdout summary table ──────────────────────────────────────────────
    print()
    print(f"{'contrast':<55} {'n':>3}  {'d':>8}  {'95% CI':>22}")
    print("-" * 92)
    for name, r in contrasts_out.items():
        ci = f"[{r['d_ci_lo']:+.2f}, {r['d_ci_hi']:+.2f}]"
        print(f"{name:<55} {r['n_pairs']:>3}  {r['d']:>+8.3f}  {ci:>22}")

if __name__ == "__main__":
    main()
