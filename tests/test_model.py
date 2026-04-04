#!/usr/bin/env python3
"""
SQLab — Unified test runner for any Ollama model.

Usage:
    python test_model.py <model_name>
    python test_model.py qwen2.5-coder:14b
    python test_model.py deepseek-coder-v2:16b
    python test_model.py phi4:14b

Uses:
  - SQLab container API at http://localhost:8000
  - Ollama OpenAI-compatible API at http://localhost:11434/v1
"""

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime

import requests

# ── Config ──────────────────────────────────────────────────────────
ENV_URL = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434/v1"
MAX_STEPS = 15
OLLAMA_TIMEOUT = 120  # seconds per LLM call
HTTP_TIMEOUT = 60     # seconds per env step API call
RESET_TIMEOUT = 300   # seconds for reset (fault injection can be slow)

TASK_IDS = [f"task_{i}" for i in range(1, 18)]

SYSTEM_PROMPT = """You are an expert PostgreSQL Database SRE (Site Reliability Engineer).
You are given an alert about a database issue. Your job is to diagnose the problem
and fix it by issuing SQL commands.

IMPORTANT RULES:
1. You may think and reason about the problem, but you MUST wrap your final SQL command in <sql> tags.
2. Issue EXACTLY ONE SQL command per turn. Example: <sql>SELECT 1</sql>
3. Start by diagnosing the issue using PostgreSQL system views and EXPLAIN ANALYZE.
4. Then fix the root cause. For compound problems, fix ALL issues — not just one.
5. Do NOT drop data tables or truncate data.
6. You have at most 15 steps. Be efficient.
7. The database is 'demo' with schema 'bookings'. Tables use bookings.table_name format.

REMEMBER: Always wrap your SQL in <sql>YOUR SQL HERE</sql> tags.
"""


def llm_call(model: str, messages: list[dict], temperature: float = 0.2) -> str:
    """Call model via Ollama's OpenAI-compatible API."""
    resp = requests.post(
        f"{OLLAMA_URL}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def env_reset(task_id: str) -> dict:
    """Reset environment for a specific task via HTTP."""
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id},
        timeout=RESET_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(command: str) -> dict:
    """Execute one SQL command via HTTP."""
    slow_ops = ["vacuum", "create index", "reindex", "analyze", "explain analyze"]
    timeout = RESET_TIMEOUT if any(op in command.lower() for op in slow_ops) else HTTP_TIMEOUT
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": {"command": command}},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def get_grader() -> dict:
    """Fetch grader result for the last completed episode."""
    resp = requests.get(f"{ENV_URL}/grader", timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def extract_sql(response: str) -> str:
    """Extract SQL from LLM response.

    Priority order:
    1. <sql>...</sql> tags (preferred — model was instructed to use these)
    2. ```sql...``` markdown fences (fallback)
    3. Raw text with non-SQL lines stripped (last resort)
    """
    text = response.strip()

    # 1. Try <sql> tags first
    match = re.search(r'<sql>(.*?)</sql>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. Try markdown code fences anywhere in the response
    fence_match = re.search(r'```(?:sql)?\s*\n?(.*?)```', text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # 3. Fallback: strip non-SQL lines
    lines = [l.strip() for l in text.split("\n") if l.strip() and not l.strip().startswith("--")]
    if not lines:
        return text

    return "\n".join(lines)


def run_task(model: str, task_id: str, task_info: dict) -> dict:
    """Run a single task and return results."""
    print(f"\n{'='*70}")
    print(f"TASK: {task_id} — {task_info['name']} [{task_info['difficulty']}]")
    print(f"{'='*70}")

    start_time = time.time()

    # Reset environment
    reset_resp = env_reset(task_id)
    obs = reset_resp.get("observation", reset_resp)

    alert = obs.get("alert", "No alert")
    schema_hint = obs.get("command_output", "")

    print(f"Alert: {alert[:120]}...")

    # Build initial messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"ALERT:\n{alert}\n\n"
            f"DATABASE INFO:\n{schema_hint}\n\n"
            f"Issue a diagnostic SQL command to begin investigating."
        )},
    ]

    steps = []
    done = False
    final_score = None
    is_resolved = False

    for step_num in range(1, MAX_STEPS + 1):
        if done:
            break

        # Get LLM response
        try:
            raw_response = llm_call(model, messages)
        except Exception as e:
            print(f"  Step {step_num}: LLM ERROR: {e}")
            steps.append({"step": step_num, "error": f"LLM: {e}"})
            break

        sql_command = extract_sql(raw_response)
        print(f"  Step {step_num}: {sql_command[:100]}{'...' if len(sql_command) > 100 else ''}")

        # Execute in environment
        try:
            step_resp = env_step(sql_command)
        except Exception as e:
            print(f"  Step {step_num}: ENV ERROR: {e}")
            steps.append({"step": step_num, "command": sql_command, "error": f"ENV: {e}"})
            break

        step_obs = step_resp.get("observation", step_resp)
        output = step_obs.get("command_output", "")
        error = step_obs.get("error", None)
        reward = step_resp.get("reward", step_obs.get("reward", 0))
        done = step_resp.get("done", step_obs.get("done", False))
        metadata = step_obs.get("metadata", {})

        is_resolved = metadata.get("is_resolved", False)
        final_score = metadata.get("grader_score", None)

        print(f"         → reward={reward}, done={done}, resolved={is_resolved}")
        if error:
            print(f"         → error: {error[:150]}")

        steps.append({
            "step": step_num,
            "command": sql_command,
            "output": output[:500] if output else None,
            "error": error,
            "reward": reward,
            "done": done,
            "resolved": is_resolved,
        })

        if done:
            break

        # Build feedback for LLM
        feedback_parts = []
        if output:
            feedback_parts.append(f"QUERY RESULT:\n{output[:3000]}")
        if error:
            feedback_parts.append(f"ERROR:\n{error[:1000]}")
        feedback_parts.append(
            f"Step {step_num}/{MAX_STEPS}. Resolved: {is_resolved}. "
            f"Issue the next SQL command."
        )

        messages.append({"role": "assistant", "content": raw_response})
        messages.append({"role": "user", "content": "\n\n".join(feedback_parts)})

    elapsed = time.time() - start_time

    # Get grader result
    grader = None
    try:
        grader = get_grader()
    except Exception:
        pass

    if grader and grader.get("task_id") == task_id:
        final_score = grader.get("score", final_score)
        print(f"\n  GRADER: score={final_score}, breakdown={grader.get('breakdown', {})}")

    print(f"  RESULT: resolved={is_resolved}, score={final_score}, "
          f"steps={len(steps)}, time={elapsed:.1f}s")

    return {
        "task_id": task_id,
        "task_name": task_info["name"],
        "difficulty": task_info["difficulty"],
        "fault_type": task_info["fault_type"],
        "is_resolved": is_resolved,
        "grader_score": final_score,
        "steps_used": len(steps),
        "elapsed_s": round(elapsed, 1),
        "grader_breakdown": grader.get("breakdown") if grader else None,
        "steps": steps,
    }


def model_to_filename(model: str) -> str:
    """Convert model name to safe filename. e.g. 'qwen2.5-coder:14b' -> 'qwen2.5-coder-14b'."""
    return re.sub(r'[^a-zA-Z0-9._-]', '-', model).strip('-')


def main():
    parser = argparse.ArgumentParser(description="SQLab — Test a model against all 17 tasks")
    parser.add_argument("model", help="Ollama model name (e.g. qwen2.5-coder:14b)")
    parser.add_argument("--tasks", nargs="*", help="Specific task IDs to run (default: all)")
    args = parser.parse_args()

    model = args.model
    task_ids = args.tasks or TASK_IDS

    safe_name = model_to_filename(model)
    output_file = f"/home/ai24mtech02001/.openclaw/workspace/meta-hackathon/results/{safe_name}.json"

    print(f"SQLab — Model Test Run")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Model: {model}")
    print(f"Environment: {ENV_URL}")
    print(f"Ollama: {OLLAMA_URL}")
    print(f"Output: {output_file}")

    # Verify environment is ready
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=10).json()
        print(f"Health: {health}")
    except Exception as e:
        print(f"ERROR: Environment not ready: {e}")
        sys.exit(1)

    # Get task list
    try:
        tasks_resp = requests.get(f"{ENV_URL}/tasks", timeout=10).json()
        tasks = {t["id"]: t for t in tasks_resp["tasks"]}
        print(f"Available tasks: {len(tasks)}")
    except Exception as e:
        print(f"ERROR: Cannot fetch tasks: {e}")
        sys.exit(1)

    # Verify Ollama is ready with this model
    try:
        test_resp = requests.post(
            f"{OLLAMA_URL}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "SELECT 1"}],
                "max_tokens": 10,
            },
            timeout=OLLAMA_TIMEOUT,
        )
        test_resp.raise_for_status()
        print(f"Ollama OK: model={model}")
    except Exception as e:
        print(f"ERROR: Ollama not ready with model '{model}': {e}")
        sys.exit(1)

    # Run tasks
    results = []
    for task_id in task_ids:
        if task_id not in tasks:
            print(f"\nSKIPPED: {task_id} (not in registry)")
            continue

        try:
            result = run_task(model, task_id, tasks[task_id])
            results.append(result)
        except Exception as e:
            print(f"\nFAILED: {task_id}: {e}")
            traceback.print_exc()
            results.append({
                "task_id": task_id,
                "task_name": tasks[task_id]["name"],
                "difficulty": tasks[task_id]["difficulty"],
                "error": str(e),
                "grader_score": 0.0,
                "is_resolved": False,
                "steps_used": 0,
                "elapsed_s": 0,
            })

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"SUMMARY — {model} — {datetime.now().isoformat()}")
    print(f"{'='*70}")
    print(f"{'Task':<10} {'Name':<40} {'Diff':<12} {'Score':>6} {'Resolved':>9} {'Steps':>6} {'Time':>7}")
    print(f"{'-'*10} {'-'*40} {'-'*12} {'-'*6} {'-'*9} {'-'*6} {'-'*7}")

    total_score = 0
    resolved_count = 0
    for r in results:
        score = r.get("grader_score", 0) or 0
        total_score += score
        if r.get("is_resolved"):
            resolved_count += 1
        print(
            f"{r['task_id']:<10} {r.get('task_name','?'):<40} "
            f"{r.get('difficulty','?'):<12} {score:>6.3f} "
            f"{'YES' if r.get('is_resolved') else 'NO':>9} "
            f"{r.get('steps_used',0):>6} "
            f"{r.get('elapsed_s',0):>6.1f}s"
        )

    print(f"\nTotal score: {total_score:.3f} / {len(results)}.000")
    print(f"Average score: {total_score/max(len(results),1):.3f}")
    print(f"Resolved: {resolved_count} / {len(results)}")

    # Save detailed results
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tasks": len(results),
                "total_score": round(total_score, 4),
                "average_score": round(total_score / max(len(results), 1), 4),
                "resolved_count": resolved_count,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
