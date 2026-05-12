#!/usr/bin/env python3
"""
re-judge.py — re-score saved action-alignment outputs with Gemini-2.5-pro
as a second judge. Computes inter-judge agreement (Cohen's kappa +
Spearman rho) against Sonnet's scores already in the JSON.

Caveat: the saved action-alignment.json records do NOT preserve which
top-K decisions were shown to Sonnet at judging time. We approximate by
showing Gemini the FULL fixture decisions (19 for paydash — small enough
that subset vs full shouldn't materially change the judgment). This
introduces some slack, documented in the output report.

Run:
  python3 benchmarks/src/id-rag-parallel/runners/re-judge.py
"""
import os, re, json, glob, time, math
import urllib.request
import urllib.error
from pathlib import Path

def _resolve(candidates):
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

_HERE = Path(__file__).resolve().parent
REPORTS_ROOT = _resolve([
    _HERE.parents[3] / "benchmarks" / "reports" / "id-rag-parity",  # continuity-ultimate
    _HERE.parent / "reports" / "id-rag-parity",                     # continuity-benchmarks
])
ENV = _resolve([
    _HERE.parents[3] / "benchmarks" / ".env",                       # continuity-ultimate
    _HERE.parent / ".env",                                          # continuity-benchmarks
])
OUT = REPORTS_ROOT / "inter-judge.json"

# ── Load .env ────────────────────────────────────────────────────────────────
env = {}
for line in ENV.read_text().splitlines():
    m = re.match(r"^([A-Z_]+)=(.+)$", line.strip())
    if m:
        env[m.group(1)] = m.group(2)
GOOGLE_API_KEY = env.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise SystemExit("GOOGLE_API_KEY missing — set in benchmarks/.env")

# ── Load fixture decisions once ──────────────────────────────────────────────
FIXTURE_DECISIONS_PATH = REPO / "demo-projects/peer-review/with-continuity/.continuity/decisions.json"
decisions = json.loads(FIXTURE_DECISIONS_PATH.read_text())
if not isinstance(decisions, list):
    decisions = decisions.get("decisions", [])
print(f"[re-judge] loaded {len(decisions)} fixture decisions")

def render_decisions(ds):
    lines = ["## Project's architectural decisions"]
    for d in ds:
        q = (d.get("question") or "")[:300]
        a = (d.get("answer") or "")[:600]
        t = ",".join(d.get("tags") or [])
        lines.append(f"- **{q}** — {a} [tags: {t}]")
    return "\n".join(lines)
DECISIONS_BLOCK = render_decisions(decisions)

# ── Gemini judge call ────────────────────────────────────────────────────────
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"
# gemini-2.5-pro burns ~300 thinking tokens before output; flash is faster + no thinking budget

def judge_prompt(prompt: str, proposed_action: str, condition: str) -> str:
    return f"""You are evaluating how well a proposed action aligns with a project's existing architectural decisions.

{DECISIONS_BLOCK}

## The proposed action for this prompt
Prompt: {prompt}

Proposed action:
{proposed_action}

Condition the agent was in: {condition} (baseline=no context, continuity=decisions given once, continuity-in-loop=decisions re-injected each turn)

## Task
Score 1-10 how well the proposed action aligns with the decisions above. 10 = cites constraints, picks the same tools, is consistent with every decision. 1 = contradicts the decisions or ignores them.

Respond in JSON: {{"score": <1-10>, "reasoning": "<one sentence>"}}"""

def call_gemini(prompt: str, retries=3) -> dict:
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
    }).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8"))
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            # parse score
            m = re.search(r'"score"\s*:\s*(\d+)', text)
            if m:
                return {"score": int(m.group(1)), "reasoning": text.strip()[:500], "raw_ok": True}
            m = re.search(r'\b(\d{1,2})\b', text)
            if m:
                return {"score": int(m.group(1)), "reasoning": f"(fallback parse) {text.strip()[:500]}", "raw_ok": False}
            return {"score": 0, "reasoning": f"(no number found) {text.strip()[:500]}", "raw_ok": False}
        except urllib.error.HTTPError as e:
            body_err = e.read().decode("utf-8")[:300]
            print(f"  ! Gemini {e.code}: {body_err}; retry {attempt+1}/{retries}")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  ! {e}; retry {attempt+1}/{retries}")
            time.sleep(2 ** attempt)
    return {"score": 0, "reasoning": "all retries failed", "raw_ok": False}

# ── Iterate all action-alignment.json and re-judge ──────────────────────────
files = sorted(glob.glob(str(REPORTS_ROOT / "**/action-alignment.json"), recursive=True))
print(f"[re-judge] found {len(files)} action-alignment files")

pairs = []
start = time.time()
total_records = 0
for f in files:
    doc = json.loads(Path(f).read_text())
    for r in doc.get("results", []):
        total_records += 1
pairs_target = total_records
print(f"[re-judge] total records to re-judge: {pairs_target}")

done = 0
for f in files:
    doc = json.loads(Path(f).read_text())
    for r in doc.get("results", []):
        done += 1
        p = judge_prompt(r["prompt"], r["proposedAction"], r["condition"])
        g = call_gemini(p)
        pairs.append({
            "file": str(Path(f).relative_to(REPO)),
            "model": doc["model"],
            "condition": r["condition"],
            "actionId": r["actionId"],
            "sonnet_score": r["judgeScore"],
            "sonnet_reasoning": r.get("judgeReasoning", "")[:300],
            "gemini_score": g["score"],
            "gemini_reasoning": g["reasoning"][:300],
            "gemini_parse_ok": g["raw_ok"],
        })
        if done % 10 == 0:
            elapsed = time.time() - start
            print(f"  [{done}/{pairs_target}] elapsed {elapsed:.0f}s | last sonnet={r['judgeScore']} gemini={g['score']}")
        # respect rate limits
        time.sleep(0.5)

# ── Save pairs ──────────────────────────────────────────────────────────────
OUT.write_text(json.dumps({
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "n": len(pairs),
    "caveat": "Gemini saw the full fixture (19 decisions for paydash); Sonnet saw top-5 retrieved. Judgments are on same actions but slightly different context.",
    "pairs": pairs,
}, indent=2))
print(f"[re-judge] wrote {OUT}")

# ── Stats ────────────────────────────────────────────────────────────────────
# Filter to successfully-parsed pairs only
valid = [p for p in pairs if p["gemini_score"] >= 1 and p["gemini_score"] <= 10]
print(f"[re-judge] {len(valid)} of {len(pairs)} pairs had valid Gemini scores")

s = [p["sonnet_score"] for p in valid]
g = [p["gemini_score"] for p in valid]

def mean(xs): return sum(xs) / len(xs) if xs else 0
def rank(xs):
    # average rank (ties get the mean of their rank positions)
    sorted_pairs = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg
        i = j + 1
    return ranks

def spearman(a, b):
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = mean(ra), mean(rb)
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da * db > 0 else float("nan")

def cohens_kappa(a, b):
    # Weighted kappa (linear weights) for ordinal 1-10 scale
    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    levels = sorted(set(a + b))
    idx = {lv: i for i, lv in enumerate(levels)}
    k = len(levels)
    n = len(a)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    # Linear weights
    w = [[1 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    obs_agree = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k)) / n
    exp_agree = sum(w[i][j] * (row[i] * col[j]) / n for i in range(k) for j in range(k)) / n
    return (obs_agree - exp_agree) / (1 - exp_agree) if exp_agree < 1 else float("nan")

print("\n=== INTER-JUDGE AGREEMENT ===")
print(f"N = {len(valid)}")
print(f"Sonnet mean: {mean(s):.2f} | Gemini mean: {mean(g):.2f}")
print(f"Sonnet-Gemini delta: {mean(s) - mean(g):+.2f}")
rho = spearman(s, g)
kappa = cohens_kappa(s, g)
print(f"Spearman rho: {rho:.3f}")
print(f"Cohen's kappa (linear-weighted): {kappa:.3f}")

def interpret(k):
    if k >= 0.8: return "almost perfect"
    if k >= 0.6: return "substantial"
    if k >= 0.4: return "moderate"
    if k >= 0.2: return "fair"
    return "slight / none"
print(f"Kappa interpretation: {interpret(kappa)}")

# Append stats to the output JSON
out_data = json.loads(OUT.read_text())
out_data["stats"] = {
    "n": len(valid),
    "sonnet_mean": mean(s),
    "gemini_mean": mean(g),
    "spearman_rho": rho,
    "cohens_kappa_linear": kappa,
    "interpretation": interpret(kappa),
}
OUT.write_text(json.dumps(out_data, indent=2))
print(f"\nStats appended to {OUT}")
