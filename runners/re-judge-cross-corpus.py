#!/usr/bin/env python3
"""
re-judge-cross-corpus.py — second-judge sample for the May 2026 §4.7 matrix.

Sibling to re-judge.py (which only covers paydash-api / gpt-4o-mini, n=540,
written 2026-04-24). This script extends inter-judge coverage to the
24-invocation cross-corpus matrix:

  data-pipeline/{gpt-4o, claude-sonnet-4-6}/run-{1,2,3}/action-alignment.json
  mobile-app   /{gpt-4o, claude-sonnet-4-6}/run-{1,2,3}/action-alignment.json

Re-scores each record with Gemini 2.5 Flash (matching §4.4's vendor) and
saves paired Sonnet vs Gemini scores to inter-judge-cross-corpus.json.
Existing inter-judge.json is preserved.

Run:
  # Default — re-judges v1 cross-corpus action-alignment outputs:
  python3 runners/re-judge-cross-corpus.py

  # Target a different reports root (e.g. v2 cross-corpus):
  python3 runners/re-judge-cross-corpus.py --reports-root reports/id-rag-parity-v2
  REPORTS_ROOT=reports/id-rag-parity-v2 python3 runners/re-judge-cross-corpus.py
"""
import os, re, sys, json, glob, time, math, subprocess, tempfile, argparse
from pathlib import Path

def _resolve(candidates):
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

_HERE = Path(__file__).resolve().parent

# CLI override — `--reports-root <path>` lets us target the v2 outputs
# without editing the resolution candidates. ENV var fallback for
# scripted invocations. Either form, when set, completely replaces the
# default v1-id-rag-parity resolution path.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--reports-root", dest="reports_root", default=None)
_args, _unknown = _parser.parse_known_args()
sys.argv = [sys.argv[0], *_unknown]  # so downstream argv inspection sees a clean list

_REPORTS_ROOT_OVERRIDE = _args.reports_root or os.environ.get("REPORTS_ROOT")
if _REPORTS_ROOT_OVERRIDE:
    REPORTS_ROOT = Path(_REPORTS_ROOT_OVERRIDE).resolve()
    if not REPORTS_ROOT.exists():
        raise SystemExit(f"--reports-root {_REPORTS_ROOT_OVERRIDE} does not exist")
else:
    REPORTS_ROOT = _resolve([
        _HERE.parents[3] / "benchmarks" / "reports" / "id-rag-parity",  # continuity-ultimate
        _HERE.parent / "reports" / "id-rag-parity",                     # continuity-benchmarks
    ])
ENV = _resolve([
    _HERE.parents[3] / "benchmarks" / ".env",                       # continuity-ultimate (in-tree)
    _HERE.parent / ".env",                                          # continuity-benchmarks
    _HERE.parents[1] / "continuity-ultimate" / "benchmarks" / ".env",  # sibling continuity-ultimate
])
OUT = REPORTS_ROOT / "inter-judge-cross-corpus.json"
FIXTURES_ROOT = _resolve([
    _HERE.parents[3] / "verification" / "shared" / "id-rag-parallel" / "fixtures",  # continuity-ultimate
    _HERE.parent / "fixtures",                                                       # continuity-benchmarks
])

# Limit which action-alignment.json files we re-judge — only the
# §4.7 cross-corpus matrix (data-pipeline + mobile-app), not paydash-api.
SCOPE_FIXTURES = {"data-pipeline", "mobile-app"}

# ── Load .env ────────────────────────────────────────────────────────────────
env = {}
for line in ENV.read_text().splitlines():
    m = re.match(r"^([A-Z_]+)=(.+)$", line.strip())
    if m:
        env[m.group(1)] = m.group(2)
GOOGLE_API_KEY = env.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise SystemExit("GOOGLE_API_KEY missing — set in benchmarks/.env")

# ── Load decisions for every in-scope fixture ────────────────────────────────
def load_decisions(fixture_name):
    p = FIXTURES_ROOT / fixture_name / ".continuity" / "decisions.json"
    raw = json.loads(p.read_text())
    if isinstance(raw, list):
        return raw
    return raw.get("decisions", [])

DECISIONS_BY_FIXTURE = {f: load_decisions(f) for f in SCOPE_FIXTURES}
for f, ds in DECISIONS_BY_FIXTURE.items():
    print(f"[re-judge-cross-corpus] {f}: {len(ds)} decisions")

def render_decisions(ds):
    lines = ["## Project's architectural decisions"]
    for d in ds:
        q = (d.get("question") or "")[:300]
        a = (d.get("answer") or "")[:600]
        t = ",".join(d.get("tags") or [])
        lines.append(f"- **{q}** — {a} [tags: {t}]")
    return "\n".join(lines)

DECISIONS_BLOCK_BY_FIXTURE = {f: render_decisions(ds) for f, ds in DECISIONS_BY_FIXTURE.items()}

# ── Gemini judge call ────────────────────────────────────────────────────────
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GOOGLE_API_KEY}"

def judge_prompt(fixture: str, prompt: str, proposed_action: str, condition: str) -> str:
    block = DECISIONS_BLOCK_BY_FIXTURE[fixture]
    return f"""You are evaluating how well a proposed action aligns with a project's existing architectural decisions.

{block}

## The proposed action for this prompt
Prompt: {prompt}

Proposed action:
{proposed_action}

Condition the agent was in: {condition} (baseline=no context, continuity=decisions given once, continuity-in-loop=decisions re-injected each turn)

## Task
Score 1-10 how well the proposed action aligns with the decisions above. 10 = cites constraints, picks the same tools, is consistent with every decision. 1 = contradicts the decisions or ignores them.

Respond in JSON: {{"score": <1-10>, "reasoning": "<one sentence>"}}"""

def call_gemini(prompt: str, retries=3) -> dict:
    """
    Call Gemini via curl --max-time so a stuck TLS read can't hang forever.
    The original urllib implementation hit a known urllib bug where timeout=60
    is ignored on certain SSL read paths — we observed a 1h28m hang at record
    476 in the first attempt at this matrix. curl's --max-time is a hard
    wall-clock cap enforced by the process itself.
    """
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
    })
    for attempt in range(retries):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as bf:
            bf.write(body)
            body_path = bf.name
        try:
            proc = subprocess.run(
                ["curl", "-s", "--max-time", "45", "--connect-timeout", "10",
                 "-X", "POST", URL,
                 "-H", "Content-Type: application/json",
                 "--data-binary", f"@{body_path}"],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                print(f"  ! curl exit={proc.returncode} stderr={proc.stderr[:200]!r}; retry {attempt+1}/{retries}", flush=True)
                time.sleep(2 ** attempt)
                continue
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                print(f"  ! non-JSON response head: {proc.stdout[:200]!r}; retry {attempt+1}/{retries}", flush=True)
                time.sleep(2 ** attempt)
                continue
            if "error" in data:
                print(f"  ! Gemini error: {data['error'].get('message','')[:200]}; retry {attempt+1}/{retries}", flush=True)
                time.sleep(2 ** attempt)
                continue
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            m = re.search(r'"score"\s*:\s*(\d+)', text)
            if m:
                return {"score": int(m.group(1)), "reasoning": text.strip()[:500], "raw_ok": True}
            m = re.search(r'\b(\d{1,2})\b', text)
            if m:
                return {"score": int(m.group(1)), "reasoning": f"(fallback parse) {text.strip()[:500]}", "raw_ok": False}
            return {"score": 0, "reasoning": f"(no number found) {text.strip()[:500]}", "raw_ok": False}
        except subprocess.TimeoutExpired:
            print(f"  ! curl wall timeout (60s); retry {attempt+1}/{retries}", flush=True)
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  ! {e}; retry {attempt+1}/{retries}", flush=True)
            time.sleep(2 ** attempt)
        finally:
            try: os.unlink(body_path)
            except OSError: pass
    return {"score": 0, "reasoning": "all retries failed", "raw_ok": False}

# ── Iterate cross-corpus action-alignment files only ─────────────────────────
all_files = sorted(glob.glob(str(REPORTS_ROOT / "**/action-alignment.json"), recursive=True))
files = [f for f in all_files if any(f"/{fx}/" in f for fx in SCOPE_FIXTURES)]
print(f"[re-judge-cross-corpus] found {len(files)} in-scope action-alignment files (of {len(all_files)} total)")

total_records = 0
for f in files:
    doc = json.loads(Path(f).read_text())
    total_records += len(doc.get("results", []))
print(f"[re-judge-cross-corpus] total records to re-judge: {total_records}")

pairs = []
done = 0
start = time.time()
for f in files:
    doc = json.loads(Path(f).read_text())
    fixture = doc.get("fixture")
    if fixture not in DECISIONS_BY_FIXTURE:
        print(f"  ! unknown fixture {fixture} in {f}; skipping")
        continue
    for r in doc.get("results", []):
        done += 1
        p = judge_prompt(fixture, r["prompt"], r["proposedAction"], r["condition"])
        g = call_gemini(p)
        pairs.append({
            "file": str(Path(f).relative_to(REPO)),
            "fixture": fixture,
            "model": doc["model"],
            "condition": r["condition"],
            "actionId": r["actionId"],
            "sonnet_score": r["judgeScore"],
            "sonnet_reasoning": r.get("judgeReasoning", "")[:300],
            "gemini_score": g["score"],
            "gemini_reasoning": g["reasoning"][:300],
            "gemini_parse_ok": g["raw_ok"],
        })
        # Log every record for the first 5, then every 25 thereafter.
        # First-5 verbose helps debug initial-call hangs; the steady-state
        # cadence of 25 keeps the log compact.
        if done <= 5 or done % 25 == 0:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_records - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total_records}] elapsed {elapsed:.0f}s, eta {eta:.0f}s | last sonnet={r['judgeScore']} gemini={g['score']} parse_ok={g['raw_ok']}", flush=True)
        time.sleep(0.5)  # rate-limit cushion

# ── Save pairs ──────────────────────────────────────────────────────────────
OUT.write_text(json.dumps({
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "scope": "id-rag-parity §4.7 cross-corpus matrix (data-pipeline + mobile-app × gpt-4o + claude-sonnet-4-6 × 3 runs)",
    "n": len(pairs),
    "judge_a": "claude-sonnet-4-6 (original judge in action-alignment.json)",
    "judge_b": "gemini-2.5-flash (this re-judge)",
    "caveat": "Gemini saw the full fixture decisions; Sonnet saw top-K retrieved. Same actions, slightly different judging context — same caveat as inter-judge.json.",
    "pairs": pairs,
}, indent=2))
print(f"[re-judge-cross-corpus] wrote {OUT}")

# ── Stats ────────────────────────────────────────────────────────────────────
valid = [p for p in pairs if 1 <= p["gemini_score"] <= 10]
print(f"[re-judge-cross-corpus] {len(valid)} of {len(pairs)} pairs had valid Gemini scores")

s = [p["sonnet_score"] for p in valid]
g = [p["gemini_score"] for p in valid]

def mean(xs): return sum(xs) / len(xs) if xs else 0
def rank(xs):
    sorted_pairs = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg
        i = j + 1
    return ranks

def spearman(a, b):
    if len(a) != len(b) or len(a) < 2: return float("nan")
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = mean(ra), mean(rb)
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da * db > 0 else float("nan")

def cohens_kappa(a, b):
    if len(a) != len(b) or len(a) == 0: return float("nan")
    levels = sorted(set(a + b))
    idx = {lv: i for i, lv in enumerate(levels)}
    k = len(levels)
    n = len(a)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b): obs[idx[x]][idx[y]] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[1 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    obs_agree = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k)) / n
    exp_agree = sum(w[i][j] * (row[i] * col[j]) / n for i in range(k) for j in range(k)) / n
    return (obs_agree - exp_agree) / (1 - exp_agree) if exp_agree < 1 else float("nan")

def interpret(k):
    if k >= 0.8: return "almost perfect"
    if k >= 0.6: return "substantial"
    if k >= 0.4: return "moderate"
    if k >= 0.2: return "fair"
    return "slight / none"

print("\n=== INTER-JUDGE AGREEMENT (cross-corpus §4.7 matrix) ===")
print(f"N = {len(valid)}")
print(f"Sonnet mean: {mean(s):.2f} | Gemini mean: {mean(g):.2f}")
print(f"Sonnet-Gemini delta: {mean(s) - mean(g):+.2f}")
rho = spearman(s, g)
kappa = cohens_kappa(s, g)
print(f"Spearman rho: {rho:.3f}")
print(f"Cohen's kappa (linear-weighted): {kappa:.3f}")
print(f"Kappa interpretation: {interpret(kappa)}")

# ── Per-fixture / per-condition breakdown ───────────────────────────────────
print("\n=== Per-fixture / per-condition breakdown ===")
groups = {}
for p in valid:
    key = (p["fixture"], p["model"], p["condition"])
    groups.setdefault(key, []).append(p)
for key in sorted(groups.keys()):
    ps = groups[key]
    sk = [x["sonnet_score"] for x in ps]
    gk = [x["gemini_score"] for x in ps]
    delta = mean(sk) - mean(gk)
    print(f"  {key[0]:14} {key[1]:24} {key[2]:20} n={len(ps):3}  sonnet={mean(sk):.2f}  gemini={mean(gk):.2f}  delta={delta:+.2f}")

# ── Append summary stats to JSON ────────────────────────────────────────────
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
