"""
SQLab — FastAPI application exposing the OpenEnv-compatible HTTP + WebSocket API.

Serves the complete SQLab environment with:
- Standard OpenEnv protocol: /reset, /step, /state (HTTP) and /ws (WebSocket)
  provided by openenv-core's create_app(), enabling EnvClient connections
- Custom endpoints: /tasks, /grader, /baseline for hackathon spec compliance
- Interactive Gradio UI mounted at /

Architecture: create_app() handles per-session environment instances for WebSocket
connections (each EnvClient gets its own DBSreEnvironment). The Gradio UI and
/baseline endpoint share a persistent singleton instance for interactive use.

This design enables both programmatic agent evaluation (WebSocket/HTTP) and
interactive exploration (Gradio playground) against the same live PostgreSQL database.
"""

import asyncio
import logging
import os
import threading
from typing import Optional, Any, Dict

import gradio as gr
from pathlib import Path

from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from openenv.core.env_server.http_server import create_app

from sqlab.models import DBSreAction, DBSreObservation
from sqlab.server.environment import DBSreEnvironment
from sqlab.server.tasks import TASK_REGISTRY
from sqlab.server.gradio_ui import create_gradio_app

logger = logging.getLogger(__name__)

# ── Create app via openenv-core ──────────────────────────────────
# create_app() wires up /reset, /step, /state (HTTP), /ws (WebSocket),
# /health, /mcp, and schema endpoints. Each WebSocket connection gets
# its own DBSreEnvironment instance via the factory pattern.
app = create_app(
    DBSreEnvironment,
    DBSreAction,
    DBSreObservation,
    env_name="sqlab",
    max_concurrent_envs=1,
)

# Persistent singleton for Gradio UI and /baseline endpoint.
# Separate from the per-session WebSocket instances above.
_env = DBSreEnvironment()
_env_lock = threading.Lock()


def _serialize_observation(obs: DBSreObservation) -> dict:
    """Serialize a DBSreObservation to a JSON-friendly dict."""
    d = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
    return d


# ── Custom endpoints (hackathon spec) ────────────────────────────

@app.get("/tasks")
async def list_tasks():
    """Return all available tasks with their metadata and action schema."""
    tasks = []
    for tid, task in TASK_REGISTRY.items():
        tasks.append({
            "id": tid,
            "name": task["name"],
            "difficulty": task["difficulty"],
            "description": task["description"],
            "fault_type": task["fault_type"],
        })
    return {
        "tasks": tasks,
        "action_schema": {"command": "string (SQL command to execute)"},
        "max_steps": 15,
    }


@app.get("/grader")
async def get_grader_score():
    """Return the grader score for the current/last episode.

    Note: This endpoint uses the most recently completed episode's score.
    In a concurrent environment, this returns the last graded result.
    """
    result = DBSreEnvironment.last_grader_result
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No episode has been graded yet. Complete an episode first."},
        )
    return result


@app.post("/baseline")
async def run_baseline():
    """Run baseline LLM agent against all 17 tasks and return scores.

    Requires OPENAI_API_KEY (or HF_TOKEN) and optionally MODEL_NAME
    environment variables. Runs each task sequentially using the
    persistent environment instance.
    """
    import openai as _openai
    import json as _json

    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse(
            status_code=400,
            content={"error": "No API key found. Set HF_TOKEN, API_KEY, or OPENAI_API_KEY."},
        )

    base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    client = _openai.OpenAI(base_url=base_url, api_key=api_key)

    system_prompt = (
        "You are an expert PostgreSQL DBA and Site Reliability Engineer.\n"
        "You are responding to a database incident. Diagnose the root cause and fix it.\n"
        "RULES: Respond with ONLY a single SQL command. No explanations, no markdown.\n"
        "Start by diagnosing (EXPLAIN, pg_stat_activity, pg_locks, etc.), then fix."
    )

    def _build_prompt(obs_dict):
        parts = [f"ALERT: {obs_dict.get('alert', '')}"]
        if obs_dict.get("command_output"):
            parts.append(f"\nOutput:\n{obs_dict['command_output']}")
        if obs_dict.get("error"):
            parts.append(f"\nError: {obs_dict['error']}")
        m = obs_dict.get("metrics", {})
        if m:
            parts.append(f"\nMetrics: {_json.dumps(m, default=str)}")
        parts.append(f"\nStep {obs_dict.get('step_number', 0)}/{obs_dict.get('max_steps', 15)}")
        parts.append("\nRespond with a single SQL command:")
        return "\n".join(parts)

    def _extract_sql(text):
        text = text.strip()
        if "```" in text:
            blocks = text.split("```")
            if len(blocks) >= 2:
                code = blocks[1].strip()
                if code.lower().startswith("sql"):
                    code = code[3:].strip()
                return code
        return text

    loop = asyncio.get_event_loop()
    task_ids = list(TASK_REGISTRY.keys())

    def _run_all():
        results = []
        for tid in task_ids:
            with _env_lock:
                obs = _env.reset(task_id=tid)
            obs_dict = _serialize_observation(obs)

            rewards = []
            steps = 0

            while not obs_dict.get("done", False):
                prompt = _build_prompt(obs_dict)
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=500,
                        temperature=0.0,
                    )
                    sql = _extract_sql(completion.choices[0].message.content or "SELECT 1")
                except Exception:
                    sql = "SELECT 1"

                action = DBSreAction(command=sql)
                with _env_lock:
                    obs = _env.step(action)
                obs_dict = _serialize_observation(obs)
                rewards.append(obs_dict.get("reward", 0.0))
                steps += 1

            meta = obs_dict.get("metadata", {})
            results.append({
                "task_id": tid,
                "score": meta.get("grader_score", 0.0) or 0.0,
                "resolved": meta.get("resolved", False),
                "steps": steps,
            })

        total = sum(r["score"] for r in results)
        resolved = sum(1 for r in results if r["resolved"])
        return {
            "model": model,
            "results": results,
            "total_score": round(total, 3),
            "average_score": round(total / len(results), 3) if results else 0.0,
            "resolved": f"{resolved}/{len(results)}",
        }

    return await loop.run_in_executor(None, _run_all)


# ── Static files + Gradio UI ────────────────────────────────────

# Serve static files (diagrams, images)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Mount Gradio UI at root — MUST be after all API routes to avoid catchall interference
_gradio_app = create_gradio_app(_env, _env_lock)
app = gr.mount_gradio_app(app, _gradio_app, path="/")


def main():
    """Entry point for running the SQLab server."""
    import uvicorn
    uvicorn.run("sqlab.server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
