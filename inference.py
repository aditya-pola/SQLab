#!/usr/bin/env python3
"""
SQLab Inference Script
===================================
Runs an LLM agent against all 17 SQLab tasks (PostgreSQL incident response)
and reports per-task scores in the mandatory OpenEnv stdout format.

Environment variables (MANDATORY):
    API_BASE_URL   The API endpoint for the LLM (default: HF router)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key (or API_KEY)
    IMAGE_NAME     Docker image name for the SQLab environment

Usage:
    IMAGE_NAME=sqlab MODEL_NAME=Qwen/Qwen2.5-72B-Instruct python -m sqlab.inference
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

from sqlab.client import DBSreEnv
from sqlab.models import DBSreAction

# ---------------------------------------------------------------------------
# Configuration — reads from environment variables per hackathon spec
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("IMAGE_NAME", "sqlab")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "sqlab"
MAX_STEPS = 15
TEMPERATURE = 0.0  # Deterministic for reproducibility
MAX_TOKENS = 500   # Sufficient for any single SQL command

# All 17 tasks ordered by difficulty (easy -> medium -> hard)
ALL_TASKS = [
    "task_1", "task_2", "task_3", "task_4", "task_5",       # Easy
    "task_6", "task_7", "task_8", "task_9", "task_10",       # Medium
    "task_11", "task_12", "task_13", "task_14", "task_15",    # Medium + Hard
    "task_16", "task_17",                                     # Hard
]

# ---------------------------------------------------------------------------
# System prompt — deliberately minimal to test diagnostic ability
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line per mandatory stdout format."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line per mandatory stdout format."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Truncate action to avoid very long lines
    action_short = action.replace("\n", " ")[:200]
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line per mandatory stdout format."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(obs_data: Dict[str, Any]) -> str:
    """Build the user prompt from an observation dict.

    Includes the alert, last command output, error, metrics, and step count.
    Mirrors real SRE incident context: observable symptoms + time pressure.
    """
    parts = [f"ALERT: {obs_data.get('alert', 'No alert')}"]

    if obs_data.get("command_output"):
        parts.append(f"\nLast command output:\n{obs_data['command_output']}")
    if obs_data.get("error"):
        parts.append(f"\nError: {obs_data['error']}")

    metrics = obs_data.get("metrics", {})
    if metrics:
        parts.append(f"\nCurrent metrics: {json.dumps(metrics, indent=2, default=str)}")

    step = obs_data.get("step_number", 0)
    max_steps = obs_data.get("max_steps", MAX_STEPS)
    parts.append(f"\nStep {step}/{max_steps}")
    parts.append("\nRespond with a single SQL command:")

    return "\n".join(parts)


def extract_sql(text: str) -> str:
    """Extract SQL from model response, stripping markdown code blocks if present."""
    text = text.strip()
    if "```" in text:
        blocks = text.split("```")
        if len(blocks) >= 2:
            code = blocks[1].strip()
            if code.lower().startswith("sql"):
                code = code[3:].strip()
            return code
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_episode(
    env: DBSreEnv,
    client: OpenAI,
    task_id: str,
) -> Dict[str, Any]:
    """Run a single episode against one task.

    Uses the OpenEnv client pattern (env.reset / env.step) with typed
    DBSreAction actions and DBSreObservation observations.
    """
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # Reset environment to the specified task
        result = await env.reset(seed=None, task_id=task_id)
        obs = result.observation

        obs_data = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Build prompt from observation and get model response
            prompt = build_prompt(obs_data)
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )

            raw_response = (completion.choices[0].message.content or "").strip()
            sql = extract_sql(raw_response)

            # Execute the SQL command
            result = await env.step(DBSreAction(command=sql))
            obs = result.observation
            obs_data = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()

            reward = result.reward or 0.0
            done = result.done
            error = obs_data.get("error")

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=sql, reward=reward, done=done, error=error)

            if done:
                break

        # Extract final score from metadata
        metadata = obs_data.get("metadata", {})
        score = metadata.get("grader_score", 0.0) or 0.0
        success = metadata.get("resolved", False)

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "steps": steps_taken,
        "success": success,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def async_main() -> None:
    if not API_KEY:
        raise SystemExit(
            "HF_TOKEN (or API_KEY) must be set to query the model.\n"
            "  export HF_TOKEN=your_token_here"
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Connect to SQLab environment via OpenEnv client
    async with DBSreEnv.from_docker_image(IMAGE_NAME) as env:
        results = []
        for task_id in ALL_TASKS:
            episode_result = await run_episode(env, client, task_id)
            results.append(episode_result)

        # Print summary
        print(f"\n{'=' * 60}", flush=True)
        print("SUMMARY", flush=True)
        print(f"{'=' * 60}", flush=True)

        total_score = sum(r["score"] for r in results)
        resolved = sum(1 for r in results if r["success"])
        avg_score = total_score / len(results) if results else 0.0

        for r in results:
            status = "RESOLVED" if r["success"] else "FAILED"
            print(
                f"  {r['task_id']:>8}: score={r['score']:.3f}  steps={r['steps']}  {status}",
                flush=True,
            )

        print(f"\n  Total:    {total_score:.3f} / {len(results)}", flush=True)
        print(f"  Average:  {avg_score:.3f}", flush=True)
        print(f"  Resolved: {resolved} / {len(results)}", flush=True)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
