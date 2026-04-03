"""
SQLab — Core Environment class.

Models the workflow of a production SRE responding to a PostgreSQL incident.
Each episode: receive alert, diagnose with SQL, fix, verify. Clean state
guarantee: each reset() fully reverses the previous fault before injecting
the next, ensuring episode independence for RL training. Pre-baked injection
optimization reduces reset latency from ~120s to ~2-5s, enabling hundreds
of training episodes per hour.

Implements the OpenEnv Environment interface: reset(), step(), state.
Orchestrates fault injection, SQL execution, reward shaping, and grading.

Performance optimization: Pre-baked fault states (Option D).
Instead of live SQL injection on every reset(), we:
1. Run cleanup SQL from the PREVIOUS fault (fast, targeted)
2. Run inject SQL for the NEW fault (fast, targeted)
3. Only fall back to live injection for thread-only faults
This reduces reset time from ~120s average to ~2-5s.
"""

import logging
import random
import time
import threading
import uuid
from typing import Optional, Any

import psycopg2

from openenv.core.env_server.interfaces import Environment

from sqlab.models import DBSreAction, DBSreObservation, DBSreState
from sqlab.server.db import (
    get_admin_connection,
    get_agent_connection,
    get_connection_params,
    execute_agent_sql,
    get_db_metrics,
    BackgroundConnectionManager,
)
from sqlab.server.tasks import TASK_REGISTRY, get_task, list_task_ids
from sqlab.server.fault_injector import get_injector
from sqlab.server.reward import compute_step_reward
from sqlab.server.grader import grade_episode

logger = logging.getLogger(__name__)

# 15-step budget forces efficient triage — mirrors real incident SLAs where
# resolution time matters. Frontier models must prioritize high-value
# diagnostic queries over exploratory ones.
MAX_STEPS = 15

# Safety guardrails prevent catastrophic actions (DROP SCHEMA, VACUUM FULL).
# These mirror real production runbook restrictions where SREs cannot
# unilaterally destroy data or perform operations that block all queries.
# Global destructive patterns — fatal for ALL tasks unless in allowed_dangerous
GLOBAL_FATAL_PATTERNS = [
    "ALTER USER",
    "WITH SUPERUSER",
    "ALTER SYSTEM RESET ALL",
    "DROP INDEX",       # dropping primary keys, etc.
    "LOCK TABLE",
    "DROP SCHEMA",
    "VACUUM FULL",
]


class DBSreEnvironment(Environment[DBSreAction, DBSreObservation, DBSreState]):
    """PostgreSQL incident-response training environment.

    Each episode:
    1. reset() picks a task, injects a fault, returns initial observation
    2. step() executes agent SQL, computes reward, checks resolution
    3. state property returns current episode metadata

    Performance: Uses pre-baked SQL for fast fault injection/cleanup.
    """

    # Class-level storage for the /grader endpoint
    last_grader_result: Optional[dict] = None

    def __init__(self):
        super().__init__()

        # DB connections (lazily opened)
        self._admin_conn = None
        self._agent_conn = None

        # Background manager for threads/connections used by faults
        self._bg_manager = BackgroundConnectionManager()

        # Episode state
        self._episode_id: str = ""
        self._task_id: str = ""
        self._task: dict = {}
        self._fault_type: str = ""
        self._inject_meta: dict = {}
        self._step_count: int = 0
        self._done: bool = True
        self._is_resolved: bool = False
        self._cumulative_reward: float = 0.0
        self._grader_score: Optional[float] = None
        self._action_history: list[str] = []
        self._error_history: list[bool] = []
        self._alert: str = ""
        self._seed: Optional[int] = None
        self._rewarded_set: set = set()  # dedup for per-step rewards

        # Pre-bake tracking: remember previous fault's prebake SQL for fast cleanup
        self._previous_prebake_sql: Optional[dict] = None
        self._previous_fault_type: str = ""

    # ── Connection management ────────────────────────────────────

    def _ensure_admin_conn(self):
        """Get or reconnect the admin connection."""
        if self._admin_conn is None or self._admin_conn.closed:
            self._admin_conn = get_admin_connection()
        return self._admin_conn

    def _ensure_agent_conn(self):
        """Get or reconnect the agent connection."""
        if self._agent_conn is None or self._agent_conn.closed:
            self._agent_conn = get_agent_connection()
        return self._agent_conn

    # ── Pre-bake helpers ─────────────────────────────────────────

    def _run_sql_list(self, conn, sql_list: list[str], label: str = ""):
        """Execute a list of SQL statements on the admin connection."""
        for sql in sql_list:
            try:
                cur = conn.cursor()
                cur.execute(sql)
            except Exception as e:
                logger.warning("Prebake SQL error (%s): %s — SQL: %s", label, e, sql[:200])

    def _start_hybrid_threads(self, fault_type: str, params: dict,
                               bg_manager: BackgroundConnectionManager) -> dict:
        """Start background threads for hybrid faults (data pre-baked, threads live).

        Returns partial metadata from the thread setup (e.g. blocker_pid).
        """
        conn_params = get_connection_params()

        if fault_type == "table_bloat":
            # Start a long-running transaction that blocks autovacuum
            blocker_conn = psycopg2.connect(**conn_params)
            blocker_conn.autocommit = False
            bg_manager.add_connection(blocker_conn)
            blocker_pid = [None]

            def hold_tx():
                try:
                    cur = blocker_conn.cursor()
                    cur.execute("BEGIN")
                    cur.execute("SELECT txid_current()")
                    cur.execute("SELECT pg_backend_pid()")
                    blocker_pid[0] = cur.fetchone()[0]
                    while not bg_manager.stop_event.wait(timeout=1.0):
                        pass
                except Exception as e:
                    logger.debug("Prebake table_bloat hold_tx ended: %s", e)

            t = threading.Thread(target=hold_tx, daemon=True)
            t.start()
            bg_manager.add_thread(t)
            time.sleep(0.5)
            return {"blocker_pid": blocker_pid[0]}

        elif fault_type == "cascading_bloat":
            # Start REPEATABLE READ transaction holding snapshot
            blocker_conn = psycopg2.connect(**conn_params)
            blocker_conn.autocommit = False
            bg_manager.add_connection(blocker_conn)
            blocker_pid = [None]

            def hold_snapshot():
                try:
                    cur = blocker_conn.cursor()
                    cur.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
                    cur.execute("SELECT txid_current()")
                    cur.execute("SELECT pg_backend_pid()")
                    blocker_pid[0] = cur.fetchone()[0]
                    cur.execute("SELECT count(*) FROM bookings.bookings")
                    while not bg_manager.stop_event.wait(timeout=1.0):
                        pass
                except Exception as e:
                    logger.debug("Prebake cascading_bloat snapshot thread ended: %s", e)

            t = threading.Thread(target=hold_snapshot, daemon=True)
            t.start()
            bg_manager.add_thread(t)
            time.sleep(1.0)
            return {"blocker_pid": blocker_pid[0]}

        elif fault_type == "compound_lock_bloat":
            # Single blocker: holds row lock AND keeps tx open
            blocker_ref = params.get("book_refs", ["361A07"])[0]
            table = params.get("target_table", "bookings")
            num_waiters = params.get("num_waiters", 3)

            blocker_conn = psycopg2.connect(**conn_params)
            blocker_conn.autocommit = False
            bg_manager.add_connection(blocker_conn)
            blocker_pid = [None]

            def hold_lock_and_tx():
                try:
                    cur = blocker_conn.cursor()
                    cur.execute("BEGIN")
                    cur.execute("SELECT txid_current()")
                    cur.execute(f"UPDATE bookings.{table} SET total_amount = total_amount WHERE book_ref = '{blocker_ref}'")
                    cur.execute("SELECT pg_backend_pid()")
                    blocker_pid[0] = cur.fetchone()[0]
                    while not bg_manager.stop_event.wait(timeout=1.0):
                        pass
                except Exception as e:
                    logger.debug("Prebake compound_lock_bloat blocker ended: %s", e)

            t = threading.Thread(target=hold_lock_and_tx, daemon=True)
            t.start()
            bg_manager.add_thread(t)
            time.sleep(1.0)

            # Start waiters
            for i in range(num_waiters):
                try:
                    wconn = psycopg2.connect(**conn_params)
                    wconn.autocommit = False
                    bg_manager.add_connection(wconn)

                    def wait_on_lock(c=wconn, ref=blocker_ref):
                        try:
                            cur = c.cursor()
                            cur.execute("BEGIN")
                            cur.execute("SET lock_timeout = '30s'")
                            cur.execute(f"UPDATE bookings.{table} SET total_amount = total_amount WHERE book_ref = '{ref}'")
                        except Exception as e:
                            logger.debug("Prebake compound waiter ended: %s", e)

                    wt = threading.Thread(target=wait_on_lock, daemon=True)
                    wt.start()
                    bg_manager.add_thread(wt)
                except Exception as e:
                    logger.warning("Prebake compound: failed to create waiter %d: %s", i, e)

            time.sleep(0.5)
            return {
                "blocker_pid": blocker_pid[0],
                "blocker_ref": blocker_ref,
                "num_waiters": num_waiters,
            }

        return {}

    # ── OpenEnv interface ────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DBSreObservation:
        """Start a new episode.

        Uses pre-baked SQL for fast fault injection when available.
        Falls back to live injection for thread-only faults.

        Args:
            seed: Random seed for reproducibility.
            episode_id: Optional episode ID (auto-generated if not given).
            **kwargs: May include 'task_id' to select a specific task.
        """
        t0 = time.time()
        self._reset_rubric()

        # Clean up any previous episode
        self._cleanup_previous()

        # Seed
        self._seed = seed
        if seed is not None:
            random.seed(seed)

        # Pick task — 17 tasks span 3 difficulty tiers. Easy tasks test
        # single-fault diagnosis (missing index, stale stats). Hard tasks
        # (compound_lock_bloat, cascading_bloat) require multi-root-cause
        # analysis — a capability gap in current frontier models.
        task_id = kwargs.get("task_id")
        if task_id is None:
            task_id = random.choice(list_task_ids())
        self._task_id = task_id
        self._task = get_task(task_id)
        self._fault_type = self._task["fault_type"]
        self._alert = self._task["alert"]

        # Episode bookkeeping
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._is_resolved = False
        self._cumulative_reward = 0.0
        self._grader_score = None
        self._action_history = []
        self._error_history = []
        self._rewarded_set = set()

        # Create fresh background manager
        self._bg_manager = BackgroundConnectionManager()

        # ── Inject the fault (fast path vs slow path) ──
        # Pre-baked SQL injection: fault state expressed as idempotent SQL
        # rather than live thread manipulation. Enables sub-5-second resets
        # critical for RL training throughput (GRPO needs thousands of episodes).
        admin = self._ensure_admin_conn()
        injector = get_injector(self._fault_type)
        prebake = injector.get_prebake_sql()

        if prebake is not None:
            # ═══ FAST PATH: Pre-baked SQL injection ═══
            self._run_sql_list(admin, prebake["inject"], f"inject:{self._fault_type}")

            # Use pre-baked metadata
            self._inject_meta = dict(prebake.get("meta", {}))

            # For hybrid faults, start threads after SQL injection
            if prebake.get("needs_threads", False):
                params = dict(self._task["params"])
                thread_meta = self._start_hybrid_threads(
                    self._fault_type, params, self._bg_manager
                )
                self._inject_meta.update(thread_meta)

            # For index_bloat, measure sizes post-injection
            if self._fault_type == "index_bloat":
                try:
                    cur = admin.cursor()
                    cur.execute("SELECT pg_relation_size('bookings.idx_ticket_flights_flight')")
                    self._inject_meta["bloated_size"] = cur.fetchone()[0]
                    # initial_size is unknown for prebake, use 80% of bloated as heuristic
                    self._inject_meta["initial_size"] = int(self._inject_meta["bloated_size"] * 0.7)
                except Exception:
                    pass

            # Store prebake SQL for fast cleanup next time
            self._previous_prebake_sql = prebake
            self._previous_fault_type = self._fault_type

            logger.info(
                "Episode %s started (PREBAKED): task=%s fault=%s seed=%s elapsed=%.1fs",
                self._episode_id, self._task_id, self._fault_type, seed, time.time() - t0,
            )
        else:
            # ═══ SLOW PATH: Live injection (thread-only faults) ═══
            params = dict(self._task["params"])
            self._inject_meta = injector.inject(admin, params, self._bg_manager)

            # No prebake SQL to cache
            self._previous_prebake_sql = None
            self._previous_fault_type = self._fault_type

            logger.info(
                "Episode %s started (LIVE): task=%s fault=%s seed=%s elapsed=%.1fs",
                self._episode_id, self._task_id, self._fault_type, seed, time.time() - t0,
            )

        # Collect initial metrics
        metrics = self._safe_metrics()

        # Build initial observation with concrete schema context.
        # Observation includes concrete schema hint with row counts. Mirrors
        # real SRE tooling (runbook context pages, PagerDuty annotations) and
        # helps the agent reason about query plans and table sizes without
        # wasting diagnostic steps on information-gathering queries.
        schema_hint = (
            "Database: demo (PostgreSQL 16, Airlines booking system)\n"
            "Schema: bookings\n"
            "Tables: bookings (~2.1M rows), tickets (~2.9M), flights (~214K), "
            "ticket_flights (~8.4M), boarding_passes (~7.9M), "
            "airports_data (104), aircrafts_data (9), seats (1.3K)\n"
            "You have superuser access. Use SQL to diagnose and fix the issue."
        )

        return DBSreObservation(
            command_output=schema_hint,
            error=None,
            alert=self._alert,
            metrics=metrics,
            step_number=0,
            max_steps=MAX_STEPS,
            done=False,
            reward=0.0,
            metadata={"task_id": self._task_id, "difficulty": self._task["difficulty"]},
        )

    def step(
        self,
        action: DBSreAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DBSreObservation:
        """Execute one agent action (SQL command) and return observation."""
        if self._done:
            return self._terminal_observation("Episode is already done. Call reset() to start a new one.")

        self._step_count += 1
        command = action.command.strip()
        self._action_history.append(command)

        # Execute SQL
        agent_conn = self._ensure_agent_conn()
        output, error = execute_agent_sql(agent_conn, command)
        self._error_history.append(error is not None)

        # ── Fatal action detection (task-aware) ──
        # Fatal action detection is task-aware: each task specifies
        # fatal_patterns and allowed_dangerous overrides. E.g., REINDEX is
        # fatal for most tasks but allowed for index_bloat. This prevents
        # agents from learning shortcut policies while allowing legitimate fixes.
        cmd_upper = command.upper()
        task_fatal = self._task.get("fatal_patterns", [])
        task_allowed = self._task.get("allowed_dangerous", [])

        is_fatal = False
        for pattern in GLOBAL_FATAL_PATTERNS + task_fatal:
            if pattern in cmd_upper:
                is_fatal = True
                break
        # Allow if it's in the task's allowlist
        for allowed in task_allowed:
            if allowed in cmd_upper:
                is_fatal = False
                break

        if is_fatal:
            self._done = True
            self._is_resolved = False
            self._cumulative_reward -= 0.5
            self._grader_score = self._run_grader()
            metrics = self._safe_metrics()
            return DBSreObservation(
                command_output=output or f"Command executed: {command[:80]}",
                error=f"FATAL: Destructive action detected. Episode terminated with penalty.",
                alert=self._alert,
                metrics=metrics,
                step_number=self._step_count,
                max_steps=MAX_STEPS,
                done=True,
                reward=-0.5,
                metadata={
                    "task_id": self._task_id,
                    "difficulty": self._task["difficulty"],
                    "is_resolved": False,
                    "cumulative_reward": round(self._cumulative_reward, 4),
                    "grader_score": self._grader_score,
                    "fatal_action": True,
                },
            )

        # Compute per-step reward
        step_reward = compute_step_reward(
            command=command,
            output=output,
            error=error,
            fault_type=self._fault_type,
            action_history=self._action_history,
            inject_meta=self._inject_meta,
            rewarded_set=self._rewarded_set,
        )
        self._cumulative_reward += step_reward
        self._cumulative_reward = max(0.0, min(1.0, self._cumulative_reward))

        # Resolution verified by querying actual database state, not
        # pattern-matching agent commands. The injector's check_resolved()
        # inspects pg_catalog / pg_stat_* views. This makes grading robust
        # against reward hacking — the agent must actually fix the problem.
        admin = self._ensure_admin_conn()
        injector = get_injector(self._fault_type)
        try:
            self._is_resolved = injector.check_resolved(admin, self._inject_meta)
        except Exception as e:
            logger.warning("check_resolved error: %s", e)
            self._is_resolved = False

        # Check done conditions
        done = False
        if self._is_resolved:
            done = True
        if self._step_count >= MAX_STEPS:
            done = True
        self._done = done

        # Collect metrics
        metrics = self._safe_metrics()

        # If done, compute final grader score
        completion_bonus = None
        if done:
            self._grader_score = self._run_grader()
            # Add completion bonus based on grader score
            if self._grader_score is not None:
                completion_bonus = round(self._grader_score * 0.5, 4)
                step_reward += completion_bonus
                self._cumulative_reward += completion_bonus

        return DBSreObservation(
            command_output=output,
            error=error,
            alert=self._alert,
            metrics=metrics,
            step_number=self._step_count,
            max_steps=MAX_STEPS,
            done=done,
            reward=step_reward,
            metadata={
                "task_id": self._task_id,
                "difficulty": self._task["difficulty"],
                "is_resolved": self._is_resolved,
                "cumulative_reward": round(self._cumulative_reward, 4),
                "grader_score": self._grader_score,
                "completion_bonus": completion_bonus,
            },
        )

    @property
    def state(self) -> DBSreState:
        """Return current episode state.

        Episode metadata including cumulative_reward, grader_score, and
        difficulty tier. Useful for curriculum learning: trainers can filter
        episodes by difficulty or score range, and RL algorithms (GRPO, PPO)
        can condition value estimates on task difficulty.
        """
        return DBSreState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            task_name=self._task.get("name", ""),
            difficulty=self._task.get("difficulty", ""),
            fault_type=self._fault_type,
            is_resolved=self._is_resolved,
            cumulative_reward=round(self._cumulative_reward, 4),
            grader_score=self._grader_score,
        )

    def close(self) -> None:
        """Clean up all resources."""
        self._cleanup_previous()
        for conn in (self._admin_conn, self._agent_conn):
            if conn and not conn.closed:
                try:
                    conn.close()
                except Exception:
                    pass
        self._admin_conn = None
        self._agent_conn = None

    # ── Internal helpers ─────────────────────────────────────────

    def _cleanup_previous(self):
        """Clean up the previous episode's fault injection.

        Bulk-terminate all non-admin backends before cleanup. Guarantees
        clean state between episodes regardless of what the agent did —
        essential for reproducible RL training where episode independence
        is a hard requirement (no state leakage between rollouts).

        Uses fast pre-baked cleanup SQL when available, falls back to
        live cleanup for thread-only faults.
        """
        if not self._fault_type:
            self._bg_manager.cleanup()
            return

        admin = self._ensure_admin_conn()

        # Terminate ALL non-admin backends to release locks/transactions fast.
        # get_pids() on busy connections can block, so use a SQL query instead.
        try:
            cur = admin.cursor()
            cur.execute("""
                SELECT pg_terminate_backend(pid)
                FROM pg_stat_activity
                WHERE datname = current_database()
                  AND pid != pg_backend_pid()
                  AND backend_type = 'client backend'
                  AND query NOT LIKE '%pg_terminate_backend%'
            """)
            time.sleep(0.3)
        except Exception as e:
            logger.warning("Bulk terminate error: %s", e)

        # Agent conn was killed by bulk terminate — discard it so
        # _ensure_agent_conn() creates a fresh one on next step().
        if self._agent_conn is not None:
            try:
                self._agent_conn.close()
            except Exception:
                pass
            self._agent_conn = None

        # Stop background threads/connections (should be fast now)
        self._bg_manager.cleanup()

        if self._previous_prebake_sql is not None:
            # ═══ FAST PATH: Run pre-baked cleanup SQL ═══
            t0 = time.time()
            self._run_sql_list(
                admin,
                self._previous_prebake_sql["cleanup"],
                f"cleanup:{self._previous_fault_type}",
            )
            logger.info(
                "Prebake cleanup for %s took %.1fs",
                self._previous_fault_type, time.time() - t0,
            )
        elif self._inject_meta:
            # ═══ SLOW PATH: Live cleanup ═══
            try:
                injector = get_injector(self._fault_type)
                injector.cleanup(admin, self._inject_meta, self._bg_manager)
            except Exception as e:
                logger.warning("Live cleanup error: %s", e)

        # Reset tracking
        self._previous_prebake_sql = None
        self._previous_fault_type = ""

    def _safe_metrics(self) -> dict:
        """Collect DB metrics, returning empty dict on error."""
        try:
            admin = self._ensure_admin_conn()
            return get_db_metrics(admin)
        except Exception as e:
            logger.warning("Metrics collection error: %s", e)
            return {"error": str(e)}

    def _run_grader(self) -> float:
        """Run the deterministic grader and store result."""
        try:
            admin = self._ensure_admin_conn()
            score, breakdown = grade_episode(
                conn=admin,
                fault_type=self._fault_type,
                inject_meta=self._inject_meta,
                action_history=self._action_history,
                error_history=self._error_history,
                steps_used=self._step_count,
            )
            # Store for /grader endpoint
            DBSreEnvironment.last_grader_result = {
                "task_id": self._task_id,
                "episode_id": self._episode_id,
                "score": round(score, 4),
                "breakdown": breakdown,
                "steps_used": self._step_count,
                "is_resolved": self._is_resolved,
            }
            logger.info(
                "Graded episode %s: score=%.3f breakdown=%s",
                self._episode_id, score, breakdown,
            )
            return round(score, 4)
        except Exception as e:
            logger.error("Grader error: %s", e)
            return 0.0

    def _terminal_observation(self, message: str) -> DBSreObservation:
        """Return an observation for a terminal/error state."""
        return DBSreObservation(
            command_output=message,
            error=None,
            alert=self._alert,
            metrics={},
            step_number=self._step_count,
            max_steps=MAX_STEPS,
            done=True,
            reward=0.0,
            metadata={
                "task_id": self._task_id,
                "grader_score": self._grader_score,
            },
        )
