"""
SQLab — Baseline inference script for evaluating LLM agents.

Uses OpenAI API to play all 17 SQLab tasks (PostgreSQL incident response)
and report per-task scores. Baseline results from 6 models validate the
difficulty curve: easy tasks (0.7-1.0), medium tasks (0.4-0.9), hard compound
tasks (0.3-0.7). This confirms SQLab is hard enough to challenge frontier
models while remaining solvable enough to provide useful RL training signal.

Requires OPENAI_API_KEY environment variable.

Usage:
    python -m sqlab.baseline [--base-url URL] [--tasks TASK_IDS]
"""

import argparse
import json
import sys
import time

import openai
import requests

from sqlab.models import DBSreAction

# System prompt is deliberately minimal: establishes the SRE role and gives
# 6 rules without task-specific hints. This tests the model's ability to
# diagnose from the alert and metrics alone — the actual skill we want to train.
SYSTEM_PROMPT = """You are an expert PostgreSQL DBA and Site Reliability Engineer.
You are responding to a database incident. Your goal is to diagnose the root cause
and fix it using SQL commands.

IMPORTANT RULES:
1. Respond with ONLY a single SQL command — no explanations, no markdown.
2. Start by diagnosing (EXPLAIN, pg_stat_activity, pg_locks, pg_indexes, etc.)
3. Then fix the issue (CREATE INDEX, VACUUM, ANALYZE, pg_terminate_backend, etc.)
4. Do NOT drop data tables or truncate data.
5. For connection issues, also set a timeout to prevent recurrence.
6. For compound problems, fix ALL issues — not just one."""


def build_prompt(obs: dict) -> str:
    """Build the user prompt from an observation.

    Includes alert, last output, error, metrics, and step count.
    The step counter provides urgency context, mirroring real incident
    time pressure where SREs must resolve issues within SLA windows.
    """
    parts = [f"ALERT: {obs.get('alert', 'No alert')}"]

    if obs.get("command_output"):
        parts.append(f"\nLast command output:\n{obs['command_output']}")
    if obs.get("error"):
        parts.append(f"\nError: {obs['error']}")

    metrics = obs.get("metrics", {})
    if metrics:
        parts.append(f"\nCurrent metrics: {json.dumps(metrics, indent=2, default=str)}")

    step = obs.get("step_number", 0)
    max_steps = obs.get("max_steps", 15)
    parts.append(f"\nStep {step}/{max_steps}")
    parts.append("\nRespond with a single SQL command:")

    return "\n".join(parts)


def extract_sql(text: str) -> str:
    """Extract SQL from model response, stripping markdown code blocks.

    Robust extraction handles bare SQL, ```sql blocks, and quoted strings.
    This prevents format-related failures from contaminating baseline scores.
    """
    text = text.strip()
    if "```" in text:
        blocks = text.split("```")
        if len(blocks) >= 2:
            code = blocks[1].strip()
            if code.lower().startswith("sql"):
                code = code[3:].strip()
            return code
    # Remove any leading/trailing quotes
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text


def run_episode(base_url: str, task_id: str, client: openai.OpenAI, model: str = "gpt-4o") -> dict:
    """Run one episode against the environment server."""
    # Reset
    resp = requests.post(f"{base_url}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    history = []
    step = 0

    while not obs.get("done", False):
        prompt = build_prompt(obs)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.0,
        )

        sql = extract_sql(response.choices[0].message.content)
        history.append(sql)

        # Step
        resp = requests.post(f"{base_url}/step", json={"command": sql})
        resp.raise_for_status()
        obs = resp.json()
        step += 1

    # Get grader score
    try:
        grader_resp = requests.get(f"{base_url}/grader")
        grader_data = grader_resp.json()
    except Exception:
        grader_data = {}

    return {
        "task_id": task_id,
        "steps": step,
        "score": grader_data.get("score", 0.0),
        "breakdown": grader_data.get("breakdown", {}),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="SQLab baseline inference")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Environment server URL")
    parser.add_argument("--tasks", nargs="*",
                        default=["task_1", "task_2", "task_3", "task_4", "task_5",
                                 "task_6", "task_7", "task_8", "task_9", "task_10",
                                 "task_11", "task_12", "task_13", "task_14",
                                 "task_15", "task_16", "task_17"],
                        help="Task IDs to run")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    args = parser.parse_args()

    # OpenAI client reads OPENAI_API_KEY from environment.
    # Temperature=0.0 ensures deterministic, reproducible baseline scores.
    client = openai.OpenAI()

    results = []
    for task_id in args.tasks:
        print(f"\n{'='*60}")
        print(f"Running {task_id}...")
        print(f"{'='*60}")

        t0 = time.time()
        result = run_episode(args.base_url, task_id, client, args.model)
        elapsed = time.time() - t0

        result["time_s"] = round(elapsed, 1)
        results.append(result)

        print(f"  Score: {result['score']:.2f}")
        print(f"  Steps: {result['steps']}")
        print(f"  Time:  {result['time_s']}s")
        print(f"  Breakdown: {json.dumps(result['breakdown'], indent=4)}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_score = 0
    for r in results:
        print(f"  {r['task_id']:>8}: score={r['score']:.2f}  steps={r['steps']}  time={r['time_s']}s")
        total_score += r["score"]
    avg_score = total_score / len(results) if results else 0
    print(f"\n  Average score: {avg_score:.3f}")
    print(f"  Total tasks:   {len(results)}")

    # Write results to file
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
