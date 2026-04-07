#!/usr/bin/env python3
"""
SQLab Inference Script
===================================
Runs an LLM agent against a single SQLab task (PostgreSQL incident response)
and reports the result in the mandatory OpenEnv stdout format.

Environment variables:
    API_BASE_URL  API endpoint for the LLM        (default: HF router)
    MODEL_NAME    Model identifier for inference   (default: Qwen2.5-72B)
    HF_TOKEN      Hugging Face / API key           (required, no default)
    IMAGE_NAME    Docker image for SQLab env       (required, no default)
    TASK_NAME     Which task to run                (default: task_12)

Available tasks:
    Easy:   task_1  (Missing Index)
            task_2  (Stale Statistics)
            task_3  (Long-Running Transaction / Lock)
            task_4  (Connection Exhaustion)
            task_5  (Bad Configuration)
    Medium: task_6  (Redundant Indexes)
            task_7  (Lock Contention — UPDATE vs SELECT)
            task_8  (Table Bloat / Vacuum Stuck)
            task_9  (Over-Indexing)
            task_10 (Index Bloat / Fragmented Index)
            task_11 (Wrong Index Column Order)
    Hard:   task_12 (Compound: Stale Stats + Missing Index)
            task_13 (Compound: Lock + Bloat)
            task_14 (Deadlock Chain)
            task_15 (Query Plan Flip)
            task_16 (Cascading Bloat — Multi-Table)
            task_17 (Compound: Connection Exhaustion + Deadlock)

Usage:
    TASK_NAME=task_1 IMAGE_NAME=sqlab HF_TOKEN=xxx python -m sqlab.inference
    TASK_NAME=task_12 IMAGE_NAME=sqlab HF_TOKEN=xxx python -m sqlab.inference
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from sqlab.client import DBSreEnv
from sqlab.models import DBSreAction

# ---------------------------------------------------------------------------
# Configuration — reads from environment variables per hackathon spec
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("IMAGE_NAME")  # No default — must be set explicitly
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "task_12")

BENCHMARK = "sqlab"
MAX_STEPS = 15
TEMPERATURE = 0.0  # Deterministic for reproducibility
MAX_TOKENS = 500   # Sufficient for any single SQL command

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
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_short = action.replace("\n", " ")[:200]
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(obs_data: Dict[str, Any]) -> str:
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
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "HF_TOKEN (or API_KEY) must be set to query the model.\n"
            "  export HF_TOKEN=your_token_here"
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await DBSreEnv.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(seed=None, task_id=TASK_NAME)
        obs = result.observation
        obs_data = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

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

        metadata = obs_data.get("metadata", {})
        score = metadata.get("grader_score", 0.0) or 0.0
        success = metadata.get("resolved", False)

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
