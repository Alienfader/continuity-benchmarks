#!/usr/bin/env python3
"""
longmemeval-rejudge-gpt4o.py — re-judge LongMemEval Gemini-Flash agent
responses with GPT-4o, for leaderboard-comparable accuracy numbers.

Reads:
  reports/longmemeval/run-N/results.json
    (output of runners/longmemeval.ts; contains agent_response per record)

Writes:
  reports/longmemeval/run-N/inter-judge-gpt4o.json
    (paired Gemini-judge vs GPT-4o-judge label per record + stats)

Idempotent — checkpoints every 25 records; resumes on re-invoke.

Cost: ~$1.50-2 for 300 records on GPT-4o ($2.50/M input, $10/M output).
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent


def load_env_key(name, env_path):
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            m = re.match(r"^([A-Z_]+)=(.+)$", line.strip())
            if m and m.group(1) == name:
                return m.group(2)
    return os.environ.get(name)


def judge_prompt(question, expected, response):
    """LongMemEval-style autoeval. Approximates the official evaluate_qa.py
    rubric (yes/no on factual consistency)."""
    return f"""You are evaluating whether a model's response correctly answers a question, given the ground-truth reference answer.

Question: {question}

Reference answer: {expected}

Model response: {response}

Is the model response consistent with the reference answer? Match on key facts; the response need not be a verbatim copy. "I don't know" only counts as correct if the reference also indicates no clear answer (abstention).

Respond in strict JSON: {{"label": 0 or 1, "reasoning": "<one sentence>"}}"""


def call_gpt4o(api_key, prompt, retries=3, timeout=60):
    """Call GPT-4o via curl --max-time so a stuck TLS read can't hang forever.
    Same pattern as re-judge-cross-corpus.py."""
    url = "https://api.openai.com/v1/chat/completions"
    body = {
        "model": "gpt-4o",
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    last_err = None
    for attempt in range(retries):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            json.dump(body, f)
            body_path = f.name
        try:
            proc = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "--max-time", str(timeout),
                    "-H", f"Authorization: Bearer {api_key}",
                    "-H", "Content-Type: application/json",
                    "-X", "POST",
                    "-d", f"@{body_path}",
                    url,
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                last_err = f"curl rc={proc.returncode}: {proc.stderr[:200]}"
                time.sleep(2 * (attempt + 1))
                continue
            resp = json.loads(proc.stdout)
            if "error" in resp:
                last_err = f"openai error: {resp['error'].get('message', '')[:200]}"
                if "rate" in last_err.lower() or "limit" in last_err.lower():
                    time.sleep(5 * (attempt + 1))
                    continue
                return {"label": 0, "reasoning": last_err, "raw_ok": False}
            text = resp["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(text)
                lbl = 1 if parsed.get("label") == 1 else 0
                return {"label": lbl, "reasoning": parsed.get("reasoning", ""), "raw_ok": True}
            except json.JSONDecodeError:
                return {"label": 0, "reasoning": f"json parse failed: {text[:120]}", "raw_ok": False}
        finally:
            Path(body_path).unlink(missing_ok=True)
    return {"label": 0, "reasoning": f"retries exhausted: {last_err}", "raw_ok": False}


def cohens_kappa(a, b):
    """Linear-weighted Cohen's kappa on 0/1 labels."""
    if len(a) == 0:
        return 0.0
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n
    pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="path to reports/longmemeval/run-N")
    p.add_argument("--max-records", type=int, default=None, help="cap records (cost control)")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"results.json not found: {results_path}")

    out_path = run_dir / "inter-judge-gpt4o.json"
    checkpoint_path = run_dir / "inter-judge-gpt4o.checkpoint.json"

    api_key = load_env_key("OPENAI_API_KEY", REPO / ".env")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY missing — set in .env or environment")

    data = json.loads(results_path.read_text())
    records = data["results"]
    if args.max_records:
        records = records[: args.max_records]
    print(f"[rejudge] {len(records)} records to re-judge")

    pairs = []
    done_keys = set()
    if checkpoint_path.exists():
        ck = json.loads(checkpoint_path.read_text())
        pairs = ck.get("pairs", [])
        done_keys = {(p["question_id"], p["condition"]) for p in pairs}
        print(f"[rejudge] resuming from checkpoint: {len(pairs)} done")

    start = time.time()
    done = 0
    for r in records:
        done += 1
        key = (r["question_id"], r["condition"])
        if key in done_keys:
            continue

        prompt = judge_prompt(
            question=_lookup_question(r["question_id"], data),
            expected=r["expected_answer"],
            response=r["agent_response"],
        )
        result = call_gpt4o(api_key, prompt)
        pairs.append({
            "question_id": r["question_id"],
            "question_type": r["question_type"],
            "condition": r["condition"],
            "gemini_label": r["autoeval_label"],
            "gpt4o_label": result["label"],
            "gpt4o_reasoning": result["reasoning"][:300],
            "gpt4o_parse_ok": result["raw_ok"],
        })

        if len(pairs) % 25 == 0:
            checkpoint_path.write_text(json.dumps({"pairs": pairs}, indent=2))
        if done <= 5 or done % 25 == 0:
            elapsed = time.time() - start
            rate = done / max(0.1, elapsed)
            eta = (len(records) - done) / max(0.001, rate)
            print(f"  [{done}/{len(records)}] elapsed {elapsed:.0f}s eta {eta:.0f}s | gemini={r['autoeval_label']} gpt4o={result['label']} parse={result['raw_ok']}")
        time.sleep(0.5)

    # ── Stats ─────────────────────────────────────────────────────────────
    stats = _compute_stats(pairs)

    out_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir.relative_to(REPO)),
        "judge_a": "gemini-2.5-flash (original)",
        "judge_b": "gpt-4o (re-judge)",
        "n": len(pairs),
        "pairs": pairs,
        "stats": stats,
    }, indent=2))
    checkpoint_path.unlink(missing_ok=True)
    print(f"\n[rejudge] wrote {out_path}")
    print(_render_stats_md(stats))


def _lookup_question(qid, data):
    """results.json doesn't carry the question text directly; pull from
    the sample fixture if available, else fall back to expected_answer
    context."""
    sample_path = REPO / data["sample"]
    if sample_path.exists():
        sample = json.loads(sample_path.read_text())
        for rec in sample:
            if rec["question_id"] == qid:
                return rec["question"]
    return "(question text not available)"


def _compute_stats(pairs):
    by_condition = defaultdict(lambda: {"n": 0, "gemini": [], "gpt4o": []})
    by_type = defaultdict(lambda: {"n": 0, "gpt4o": []})
    overall_g = []
    overall_o = []
    for p in pairs:
        by_condition[p["condition"]]["n"] += 1
        by_condition[p["condition"]]["gemini"].append(p["gemini_label"])
        by_condition[p["condition"]]["gpt4o"].append(p["gpt4o_label"])
        by_type[(p["condition"], p["question_type"])]["n"] += 1
        by_type[(p["condition"], p["question_type"])]["gpt4o"].append(p["gpt4o_label"])
        overall_g.append(p["gemini_label"])
        overall_o.append(p["gpt4o_label"])

    overall = {
        "n": len(pairs),
        "gemini_accuracy": sum(overall_g) / max(1, len(overall_g)),
        "gpt4o_accuracy": sum(overall_o) / max(1, len(overall_o)),
        "agreement_rate": sum(1 for g, o in zip(overall_g, overall_o) if g == o) / max(1, len(overall_g)),
        "cohens_kappa": cohens_kappa(overall_g, overall_o),
    }

    per_condition = {}
    for cond, v in by_condition.items():
        per_condition[cond] = {
            "n": v["n"],
            "gemini_accuracy": sum(v["gemini"]) / max(1, v["n"]),
            "gpt4o_accuracy": sum(v["gpt4o"]) / max(1, v["n"]),
            "agreement_rate": sum(1 for g, o in zip(v["gemini"], v["gpt4o"]) if g == o) / max(1, v["n"]),
            "cohens_kappa": cohens_kappa(v["gemini"], v["gpt4o"]),
        }

    per_type = {}
    for (cond, qt), v in by_type.items():
        per_type[f"{cond}::{qt}"] = {
            "n": v["n"],
            "gpt4o_accuracy": sum(v["gpt4o"]) / max(1, v["n"]),
        }

    return {"overall": overall, "per_condition": per_condition, "per_condition_per_type": per_type}


def _render_stats_md(stats):
    o = stats["overall"]
    lines = [
        "",
        "=== Overall ===",
        f"  n = {o['n']}",
        f"  Gemini-judge accuracy: {o['gemini_accuracy']:.1%}",
        f"  GPT-4o-judge accuracy: {o['gpt4o_accuracy']:.1%}",
        f"  agreement: {o['agreement_rate']:.1%}",
        f"  Cohen's kappa: {o['cohens_kappa']:.3f}",
        "",
        "=== Per condition (GPT-4o-judge) ===",
    ]
    for c, v in stats["per_condition"].items():
        lines.append(f"  {c:30s} gpt4o={v['gpt4o_accuracy']:.1%}  gemini={v['gemini_accuracy']:.1%}  agreement={v['agreement_rate']:.1%}  k={v['cohens_kappa']:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
