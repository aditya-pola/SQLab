#!/usr/bin/env python3
"""
SQLab Inference Script
===================================
Runs an LLM agent against a single SQLab task (PostgreSQL incident response)
and reports the result in the mandatory OpenEnv stdout format.

Environment variables:
    API_BASE_URL  API endpoint for the LLM        (default: HF router)
    MODEL_NAME    Model identifier for inference   (default: Qwen2.5-72B)
    API_KEY       API key for the LLM              (required, no default)
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
    TASK_NAME=task_1 API_KEY=xxx python inference.py
    TASK_NAME=task_12 API_KEY=xxx python inference.py
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — reads from environment variables per hackathon spec
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME", "task_12")

ENV_URL = os.environ.get("ENV_URL", "https://stvident-sqlab.hf.space")

BENCHMARK = "sqlab"
MAX_STEPS = 15
TEMPERATURE = 0.0
MAX_TOKENS = 500

# ---------------------------------------------------------------------------
# System prompt
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
# Stdout logging helpers
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


# ---------------------------------------------------------------------------
# Prompt / SQL helpers
# ---------------------------------------------------------------------------


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
# HTTP environment client (connects to HF Space directly)
# ---------------------------------------------------------------------------


class EnvClient:
    """Thin HTTP client that talks to the SQLab server's /reset and /step."""

    def __init__(self, base_url: str, timeout: int = 60):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def reset(self, task_id: str) -> Dict[str, Any]:
        r = self.session.post(
            f"{self.base}/reset",
            json={"task_id": task_id},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def step(self, command: str) -> Dict[str, Any]:
        r = self.session.post(
            f"{self.base}/step",
            json={"action": {"command": command}},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def close(self):
        self.session.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not API_KEY:
        raise SystemExit(
            "API_KEY must be set to query the model.\n"
            "  export API_KEY=your_token_here"
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(ENV_URL)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        resp = env.reset(TASK_NAME)
        obs_data = resp.get("observation", {})
        done = resp.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
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

            resp = env.step(sql)
            obs_data = resp.get("observation", {})
            reward = resp.get("reward", 0.0) or 0.0
            done = resp.get("done", False)
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
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
