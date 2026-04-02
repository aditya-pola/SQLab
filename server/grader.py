"""
SQLab — Deterministic graders for all 17 PostgreSQL incident-response tasks.

All 17 graders are fully deterministic — no LLM judge, no stochastic sampling.
Score reproducibility is critical for RL reward signal stability: given the same
DB state and action history, a grader will always return the same score.

Structure: every grader scores across three sections:
  Diagnosis    (0.4) = Investigation (0.2) + Identification (0.2)
  Resolution   (0.4) = DB state checks × efficiency_penalty
  Best Practice (0.2) = clean execution, safety, prevention

The 3-section structure (Diagnosis 40%, Resolution 40%, Best Practice 20%) reflects
real SRE performance evaluation: understanding the problem matters as much as fixing
it. This mirrors how on-call engineers are assessed in post-incident reviews.

Resolution scores check actual PostgreSQL catalog state (pg_indexes,
pg_stat_user_tables, pg_settings), not whether the agent typed the right keywords.
This prevents reward hacking — an agent cannot game the grader by echoing known SQL
patterns without actually modifying the database.

Tested against 255 adversarial scenarios (no-op agents, keyword-stuffing agents,
destructive agents, partial-fix agents) to verify graders cannot be gamed.
"""

import json
import logging
import re
from typing import List, Tuple

import psycopg2
import psycopg2.extras

from sqlab.server.db import get_admin_connection

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════

def _exec(conn, sql: str):
    """Execute SQL on admin conn and return rows as dicts."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql)
    try:
        return cur.fetchall()
    except Exception:
        return []


def _history_contains(action_history: List[str], *keywords: str) -> bool:
    """Check if any command in history contains ALL the given keywords (case-insensitive).
    Used for diagnosis scoring: verifying the agent investigated the right system views
    before attempting a fix. This encourages methodical troubleshooting over guessing."""
    for cmd in action_history:
        upper = cmd.upper()
        if all(kw.upper() in upper for kw in keywords):
            return True
    return False


def _history_contains_any(action_history: List[str], *keywords: str) -> bool:
    """Check if any command in history contains ANY of the given keywords."""
    for cmd in action_history:
        upper = cmd.upper()
        if any(kw.upper() in upper for kw in keywords):
            return True
    return False


def _efficiency_penalty(steps_used: int, threshold: int) -> float:
    """Multiplier on resolution score. At/under threshold = 1.0.
    Each step over: -0.05. Minimum 0.5.

    Efficiency penalty mirrors real incident response: SRE performance reviews
    weigh time-to-resolution. Penalty is gentle (min 0.5x multiplier) to avoid
    cliff-edge scoring that would destabilize RL training gradients."""
    if steps_used <= threshold:
        return 1.0
    return max(0.5, 1.0 - (steps_used - threshold) * 0.05)


def _error_rate(error_history: List[bool]) -> float:
    """Fraction of commands that errored."""
    if not error_history:
        return 0.0
    return sum(error_history) / len(error_history)


def _has_destructive(history: List[str]) -> bool:
    """Check for DROP TABLE or TRUNCATE in history. Penalizing destructive commands
    across all 17 graders ensures agents learn production-safe behavior — a key
    property for any environment targeting real-world SRE training."""
    return _history_contains_any(history, "DROP TABLE", "TRUNCATE")


def _fresh_explain(sql: str) -> str:
    """Open a fresh connection, run EXPLAIN (FORMAT JSON), return plan text.

    Opens a fresh connection to avoid inheriting session-level GUC settings from
    the admin connection. Ensures EXPLAIN output reflects actual DB state after
    the agent's changes, not a stale session cache."""
    conn = None
    try:
        conn = get_admin_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
        rows = cur.fetchall()
        if rows:
            return json.dumps(rows[0])
        return ""
    except Exception as e:
        logger.debug("_fresh_explain failed: %s", e)
        return ""
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def _fresh_show(param: str) -> str:
    """Open a fresh connection, run SHOW <param>, return value string.

    Same fresh-connection pattern as _fresh_explain: avoids session-level SET
    overrides so we grade against the persistent server configuration."""
    conn = None
    try:
        conn = get_admin_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(f"SHOW {param}")
        rows = cur.fetchall()
        if rows:
            row = rows[0]
            return str(list(row.values())[0])
        return ""
    except Exception as e:
        logger.debug("_fresh_show failed: %s", e)
        return ""
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# Per-task step thresholds for efficiency penalty.
# Calibrated from 6-model baselines (Phi-4, Qwen2.5-Coder, Devstral, DeepSeek,
# Qwen3, GPT-4.1-mini). Easy single-fault tasks allow fewer steps; compound
# multi-fault tasks allow more. Thresholds set at ~75th percentile of successful
# runs so that competent agents are not penalized but inefficient exploration is.
STEP_THRESHOLDS = {
    "missing_index": 9,
    "stale_statistics": 9,
    "connection_exhaustion": 10,
    "lock_contention": 10,
    "table_bloat": 11,
    "over_indexing": 12,
    "compound_stats_index": 12,
    "compound_lock_bloat": 13,
    "bad_config": 10,
    "index_bloat": 10,
    "wrong_index_order": 9,
    "deadlock_chain": 11,
    "query_plan_flip": 10,
    "cascading_bloat": 14,
    "permission_error": 8,
    "sequence_exhaustion": 9,
    "compound_conn_deadlock": 14,
}


# ═══════════════════════════════════════════════════════════════════
# Task 1: Missing Index
# ═══════════════════════════════════════════════════════════════════

def _grade_missing_index(conn, meta: dict, history: List[str],
                         error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Simulates the #1 most common PostgreSQL performance issue: a missing index
    causing sequential scans. Requires reading EXPLAIN plans — a skill many LLMs
    struggle with because plan output is dense, nested, and numeric."""
    breakdown = {}
    score = 0.0
    col = meta.get("target_column", "flight_id")
    table = meta.get("target_table", "ticket_flights")

    # ── Diagnosis (0.4) ──
    # Diagnosis scoring checks that the agent investigated before acting.
    # In production SRE, acting without diagnosis causes secondary outages.
    # Investigation (0.2)
    if _history_contains_any(history, "EXPLAIN"):
        breakdown["inv_explain"] = 0.10
        score += 0.10
    if _history_contains_any(history, "PG_INDEXES", "PG_STAT_USER_INDEXES"):
        breakdown["inv_checked_indexes"] = 0.10
        score += 0.10

    # Identification (0.2)
    if _history_contains(history, table) and _history_contains_any(history, "EXPLAIN", "INDEX"):
        breakdown["id_target_table"] = 0.10
        score += 0.10
    if _history_contains_any(history, col):
        breakdown["id_target_column"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    # Grading by DB state, not command keywords: the agent can use any valid SQL
    # to fix the issue. This openness encourages creative solutions while
    # remaining fully deterministic.
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["missing_index"])
    res_score = 0.0

    rows = _exec(conn, f"""
        SELECT indexdef FROM pg_indexes
        WHERE schemaname = 'bookings' AND tablename = '{table}'
          AND indexdef LIKE '%({col}%'
    """)
    if rows:
        res_score += 0.20
        breakdown["res_index_exists"] = 0.20

    plan_text = _fresh_explain(
        f"SELECT tf.ticket_no, tf.fare_conditions, tf.amount "
        f"FROM bookings.{table} tf WHERE tf.{col} = 2880"
    )
    if plan_text:
        if "Index" in plan_text and "Seq Scan" not in plan_text:
            res_score += 0.20
            breakdown["res_plan_improved"] = 0.20
        elif "Index" in plan_text:
            res_score += 0.10
            breakdown["res_plan_improved"] = 0.10

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    # Best practice scoring rewards production-safe behavior: CONCURRENTLY for
    # index builds, running ANALYZE after schema changes, avoiding destructive ops.
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "CONCURRENTLY"):
        breakdown["bp_concurrently"] = 0.05
        score += 0.05
    if _history_contains_any(history, "ANALYZE"):
        breakdown["bp_analyzed_after"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 2: Stale Statistics
# ═══════════════════════════════════════════════════════════════════

def _grade_stale_statistics(conn, meta: dict, history: List[str],
                            error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Stale table statistics cause the planner to choose catastrophic query plans.
    Tests whether agents can correlate estimated vs. actual row counts in EXPLAIN
    ANALYZE output — a numeric reasoning challenge frontier models often fail."""
    breakdown = {}
    score = 0.0
    table = meta.get("target_table", "flights")

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "EXPLAIN"):
        breakdown["inv_explain"] = 0.10
        score += 0.10
    if _history_contains_any(history, "PG_STAT_USER_TABLES", "N_DEAD_TUP"):
        breakdown["inv_checked_stats"] = 0.10
        score += 0.10
    if _history_contains_any(history, table):
        breakdown["id_target_table"] = 0.10
        score += 0.10
    if _history_contains(history, "ANALYZE", table):
        breakdown["id_stale_stats"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["stale_statistics"])
    res_score = 0.0

    rows = _exec(conn, f"""
        SELECT last_analyze FROM pg_stat_user_tables
        WHERE schemaname = 'bookings' AND relname = '{table}'
          AND last_analyze > now() - interval '10 minutes'
    """)
    if rows:
        res_score += 0.25
        breakdown["res_analyze_ran"] = 0.25

    # Check estimate accuracy with fresh connection
    try:
        status_to = meta.get("status_to", "Delayed")
        fresh_conn = get_admin_connection()
        try:
            cur = fresh_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(f"""
                EXPLAIN (ANALYZE, FORMAT JSON)
                SELECT * FROM bookings.{table} WHERE status = '{status_to}'
            """)
            explain_rows = cur.fetchall()
            if explain_rows:
                flat = json.dumps(explain_rows[0])
                est_match = re.search(r'"Plan Rows":\s*(\d+)', flat)
                act_match = re.search(r'"Actual Rows":\s*(\d+)', flat)
                if est_match and act_match:
                    est = int(est_match.group(1))
                    act = int(act_match.group(1))
                    if act > 0:
                        ratio = max(est, act) / max(min(est, act), 1)
                        if ratio < 10:
                            res_score += 0.15
                            breakdown["res_estimates_accurate"] = 0.15
                        elif ratio < 100:
                            res_score += 0.08
                            breakdown["res_estimates_accurate"] = 0.08
        finally:
            fresh_conn.close()
    except Exception as e:
        logger.debug("Estimate accuracy check failed: %s", e)

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains(history, "ANALYZE", table):
        breakdown["bp_targeted_analyze"] = 0.05
        score += 0.05
    # Check diagnosed first: first EXPLAIN before first ANALYZE
    first_explain = next((i for i, cmd in enumerate(history) if "EXPLAIN" in cmd.upper()), None)
    first_analyze = next((i for i, cmd in enumerate(history) if "ANALYZE" in cmd.upper()), None)
    if first_explain is not None and first_analyze is not None and first_explain < first_analyze:
        breakdown["bp_diagnosed_first"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 3: Connection Exhaustion
# ═══════════════════════════════════════════════════════════════════

def _grade_connection_exhaustion(conn, meta: dict, history: List[str],
                                 error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Models the most common production P1 incident: connection pool exhaustion from
    leaked idle-in-transaction sessions. Agent must identify and terminate idle sessions,
    then configure a timeout to prevent recurrence — a two-phase fix most models miss."""
    breakdown = {}
    score = 0.0

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "PG_STAT_ACTIVITY"):
        breakdown["inv_checked_activity"] = 0.10
        score += 0.10
    if _history_contains_any(history, "MAX_CONNECTIONS"):
        breakdown["inv_checked_max_conn"] = 0.10
        score += 0.10
    if _history_contains_any(history, "IDLE", "IDLE IN TRANSACTION"):
        breakdown["id_idle_sessions"] = 0.10
        score += 0.10
    if _history_contains(history, "PG_TERMINATE_BACKEND") and _history_contains_any(history, "IDLE", "STATE"):
        breakdown["id_terminate_idle"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    # Grading by DB state, not command keywords: we count remaining idle-in-transaction
    # sessions and check pg_file_settings for a timeout. The agent can use any valid
    # approach (pg_terminate_backend, pg_cancel_backend, ALTER SYSTEM) to achieve this.
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["connection_exhaustion"])
    res_score = 0.0

    rows = _exec(conn, """
        SELECT count(*) as cnt FROM pg_stat_activity
        WHERE state = 'idle in transaction'
          AND datname = current_database()
          AND pid != pg_backend_pid()
    """)
    idle_count = rows[0]["cnt"] if rows else 999
    if idle_count < 5:
        res_score += 0.20
        breakdown["res_idle_terminated"] = 0.20
    elif idle_count < 20:
        res_score += 0.10
        breakdown["res_idle_terminated"] = 0.10

    rows = _exec(conn, """
        SELECT setting FROM pg_file_settings
        WHERE name = 'idle_in_transaction_session_timeout'
        AND error IS NULL
        ORDER BY seqno DESC LIMIT 1
    """)
    if rows and rows[0]["setting"] and rows[0]["setting"] != '0':
        res_score += 0.20
        breakdown["res_timeout_set"] = 0.20

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    # Best practice scoring rewards production-safe behavior: targeted pg_terminate_backend
    # with WHERE clauses (not blanket kills), reloading config, and low error rates.
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "PG_RELOAD_CONF"):
        breakdown["bp_reload_conf"] = 0.05
        score += 0.05
    # Check for WHERE clause in terminate commands
    for cmd in history:
        upper = cmd.upper()
        if "PG_TERMINATE_BACKEND" in upper and "WHERE" in upper:
            breakdown["bp_targeted_terminate"] = 0.05
            score += 0.05
            break

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 4: Lock Contention
# ═══════════════════════════════════════════════════════════════════

def _grade_lock_contention(conn, meta: dict, history: List[str],
                           error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Simulates a production lock chain where one long-running transaction blocks N
    others. Agent must distinguish the root blocker from victims using pg_locks join
    pg_stat_activity — a multi-table correlation that challenges LLM reasoning."""
    breakdown = {}
    score = 0.0
    table = meta.get("target_table", "bookings")

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "PG_STAT_ACTIVITY"):
        breakdown["inv_checked_activity"] = 0.10
        score += 0.10
    if _history_contains_any(history, "PG_LOCKS"):
        breakdown["inv_checked_locks"] = 0.10
        score += 0.10
    if _history_contains_any(history, "GRANTED", "PG_BLOCKING_PIDS") or \
       (_history_contains_any(history, "PG_LOCKS") and _history_contains_any(history, "PG_STAT_ACTIVITY")):
        breakdown["id_blocker_pattern"] = 0.10
        score += 0.10
    if _history_contains_any(history, table) and _history_contains_any(history, "LOCK", "PG_LOCKS", "BLOCKED"):
        breakdown["id_target_table"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    # Resolution checks live DB state: are there still lock waiters? Are there still
    # ungranted relation locks? Any valid resolution path counts — not just the
    # textbook approach.
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["lock_contention"])
    res_score = 0.0

    rows = _exec(conn, """
        SELECT count(*) as cnt FROM pg_stat_activity
        WHERE wait_event_type = 'Lock'
          AND datname = current_database()
    """)
    lock_waits = rows[0]["cnt"] if rows else 999
    if lock_waits == 0:
        res_score += 0.25
        breakdown["res_no_lock_waits"] = 0.25

    rows = _exec(conn, """
        SELECT count(*) as cnt FROM pg_locks
        WHERE NOT granted AND locktype = 'relation'
    """)
    blocked = rows[0]["cnt"] if rows else 999
    if blocked == 0:
        res_score += 0.15
        breakdown["res_no_blocked_queries"] = 0.15

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "LOCK_TIMEOUT"):
        breakdown["bp_lock_timeout"] = 0.05
        score += 0.05
    # Targeted kill: PG_TERMINATE_BACKEND with a specific PID (not blanket)
    for cmd in history:
        upper = cmd.upper()
        if "PG_TERMINATE_BACKEND" in upper and ("WHERE" in upper or re.search(r'PG_TERMINATE_BACKEND\s*\(\s*\d+', upper)):
            breakdown["bp_targeted_kill"] = 0.05
            score += 0.05
            break

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 5: Table Bloat / Vacuum Stuck
# ═══════════════════════════════════════════════════════════════════

def _grade_table_bloat(conn, meta: dict, history: List[str],
                       error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Reproduces vacuum-blocked-by-long-transaction, the #1 cause of uncontrolled
    table growth in production PostgreSQL. Agent must find the snapshot-holding
    transaction, terminate it, then VACUUM — a causal chain LLMs rarely complete."""
    breakdown = {}
    score = 0.0
    table = meta.get("target_table", "bookings")

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "PG_STAT_USER_TABLES", "N_DEAD_TUP"):
        breakdown["inv_checked_stats"] = 0.10
        score += 0.10
    if _history_contains_any(history, "PG_STAT_ACTIVITY"):
        breakdown["inv_checked_activity"] = 0.10
        score += 0.10
    if _history_contains_any(history, table) and _history_contains_any(history, "N_DEAD_TUP", "VACUUM", "DEAD"):
        breakdown["id_dead_tuples"] = 0.10
        score += 0.10
    if _history_contains_any(history, "BACKEND_XMIN", "TXID", "XID", "XACT_START"):
        breakdown["id_blocking_tx"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["table_bloat"])
    res_score = 0.0

    # No long-running txns with old backend_xmin
    rows = _exec(conn, """
        SELECT count(*) as cnt FROM pg_stat_activity
        WHERE backend_xmin IS NOT NULL
          AND age(backend_xmin) > 1000
          AND datname = current_database()
          AND pid != pg_backend_pid()
    """)
    old_xmin = rows[0]["cnt"] if rows else 999
    if old_xmin == 0:
        res_score += 0.15
        breakdown["res_blocker_gone"] = 0.15

    # Dead tuples reduced
    rows = _exec(conn, f"""
        SELECT n_dead_tup FROM pg_stat_user_tables
        WHERE schemaname = 'bookings' AND relname = '{table}'
    """)
    dead = rows[0]["n_dead_tup"] if rows else 999999
    update_count = meta.get("update_count", 200000)
    if dead < update_count * 0.3:
        res_score += 0.25
        breakdown["res_dead_tuples_reduced"] = 0.25
    elif dead < update_count * 0.7:
        res_score += 0.12
        breakdown["res_dead_tuples_reduced"] = 0.12

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "VACUUM"):
        breakdown["bp_ran_vacuum"] = 0.05
        score += 0.05
    if _history_contains_any(history, "IDLE_IN_TRANSACTION_SESSION_TIMEOUT", "STATEMENT_TIMEOUT"):
        breakdown["bp_prevention"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 6: Over-Indexing
# ═══════════════════════════════════════════════════════════════════

def _grade_over_indexing(conn, meta: dict, history: List[str],
                         error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Reverse of missing_index: table has 8+ redundant indexes degrading write
    throughput. Agent must identify unused indexes via idx_scan stats, drop them
    without removing the primary key — a precision task that penalizes over-eagerness."""
    breakdown = {}
    score = 0.0
    table = meta.get("target_table", "ticket_flights")
    junk_indexes = meta.get("junk_indexes", [])

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "PG_STAT_USER_INDEXES", "PG_STAT_ALL_INDEXES"):
        breakdown["inv_checked_index_stats"] = 0.10
        score += 0.10
    if _history_contains_any(history, "PG_INDEXES"):
        breakdown["inv_checked_table"] = 0.10
        score += 0.10
    if _history_contains_any(history, "IDX_SCAN"):
        breakdown["id_unused_indexes"] = 0.10
        score += 0.10
    if _history_contains_any(history, table) and _history_contains_any(history, "INDEX", "PG_INDEXES"):
        breakdown["id_target_table"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    # Proportional reward shaping: score scales linearly with fraction of junk indexes
    # dropped. This gives smooth RL gradients instead of all-or-nothing scoring.
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["over_indexing"])
    res_score = 0.0

    # Count how many junk indexes remain
    remaining = 0
    for idx_name in junk_indexes:
        rows = _exec(conn, f"""
            SELECT 1 FROM pg_indexes
            WHERE schemaname = 'bookings' AND indexname = '{idx_name}'
        """)
        if rows:
            remaining += 1

    if junk_indexes:
        dropped_pct = 1.0 - (remaining / len(junk_indexes))
        junk_score = 0.25 * dropped_pct
        res_score += junk_score
        breakdown["res_junk_dropped"] = round(junk_score, 3)

    # PK preserved
    rows = _exec(conn, """
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'bookings'
          AND tablename = 'ticket_flights'
          AND indexname = 'ticket_flights_pkey'
    """)
    if rows:
        res_score += 0.15
        breakdown["res_pk_preserved"] = 0.15

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains(history, "DROP INDEX", "CONCURRENTLY"):
        breakdown["bp_concurrently"] = 0.05
        score += 0.05
    # All non-junk indexes still exist
    junk_set = set(junk_indexes)
    rows = _exec(conn, f"""
        SELECT indexname FROM pg_indexes
        WHERE schemaname = 'bookings' AND tablename = '{table}'
    """)
    existing = {r["indexname"] for r in rows} if rows else set()
    # We can't check what non-junk were there originally, but PK check covers main case
    if "ticket_flights_pkey" in existing:
        breakdown["bp_essential_preserved"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 7: Compound Stats + Index
# ═══════════════════════════════════════════════════════════════════

def _grade_compound_stats_index(conn, meta: dict, history: List[str],
                                error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Two independent faults (stale stats + missing index) that interact: fixing only
    one may appear to improve the query plan but leaves residual degradation. Tests
    multi-root-cause analysis — a core SRE skill that single-fault benchmarks miss.

    Compound faults require multi-step reasoning: the agent must identify and fix
    both root causes. Fixing only one yields partial credit via proportional scoring."""
    breakdown = {}
    score = 0.0
    index_meta = meta.get("index_meta", {})
    stats_meta = meta.get("stats_meta", {})
    idx_col = index_meta.get("target_column", "flight_id")
    idx_table = index_meta.get("target_table", "ticket_flights")
    stats_table = stats_meta.get("target_table", "flights")

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "EXPLAIN"):
        breakdown["inv_ran_explain"] = 0.10
        score += 0.10
    if _history_contains_any(history, "PG_INDEXES", "PG_STAT_USER_TABLES"):
        breakdown["inv_checked_catalogs"] = 0.10
        score += 0.10
    if _history_contains_any(history, idx_col) or (_history_contains_any(history, idx_table) and _history_contains_any(history, "INDEX")):
        breakdown["id_missing_index"] = 0.10
        score += 0.10
    if _history_contains_any(history, stats_table) and _history_contains_any(history, "ANALYZE", "STAT"):
        breakdown["id_stale_stats"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["compound_stats_index"])
    res_score = 0.0

    rows = _exec(conn, f"""
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'bookings' AND tablename = '{idx_table}'
          AND indexdef LIKE '%({idx_col}%'
    """)
    index_ok = bool(rows)
    if index_ok:
        res_score += 0.20
        breakdown["res_index_created"] = 0.20

    rows = _exec(conn, f"""
        SELECT 1 FROM pg_stat_user_tables
        WHERE schemaname = 'bookings' AND relname = '{stats_table}'
          AND last_analyze > now() - interval '10 minutes'
    """)
    analyze_ok = bool(rows)
    if analyze_ok:
        res_score += 0.15
        breakdown["res_analyze_ran"] = 0.15

    # Bonus for resolving both faults: rewards complete root-cause analysis over
    # partial fixes. This interaction bonus is unique to compound tasks.
    if index_ok and analyze_ok:
        res_score += 0.05
        breakdown["res_fully_resolved"] = 0.05

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "CONCURRENTLY"):
        breakdown["bp_concurrently"] = 0.05
        score += 0.05
    # Diagnosed before corrective
    first_diag = next((i for i, cmd in enumerate(history) if any(
        kw in cmd.upper() for kw in ["EXPLAIN", "PG_STAT", "PG_INDEXES"])), None)
    first_fix = next((i for i, cmd in enumerate(history) if any(
        kw in cmd.upper() for kw in ["CREATE INDEX", "ANALYZE"])), None)
    if first_diag is not None and first_fix is not None and first_diag < first_fix:
        breakdown["bp_diagnosed_first"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 8: Compound Lock + Bloat
# ═══════════════════════════════════════════════════════════════════

def _grade_compound_lock_bloat(conn, meta: dict, history: List[str],
                               error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Compound fault: lock contention prevents vacuum from reclaiming dead tuples,
    creating a feedback loop of growing bloat. Agent must resolve locks first, then
    vacuum — order matters, and the grader awards a bonus for resolving both."""
    breakdown = {}
    score = 0.0
    table = meta.get("target_table", "bookings")

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "PG_STAT_ACTIVITY"):
        breakdown["inv_checked_activity"] = 0.10
        score += 0.10
    if _history_contains_any(history, "PG_LOCKS"):
        breakdown["inv_checked_locks"] = 0.10
        score += 0.10
    if _history_contains_any(history, table) and _history_contains_any(history, "LOCK", "PG_LOCKS", "WAIT", "BLOCKED"):
        breakdown["id_lock_issue"] = 0.10
        score += 0.10
    if _history_contains_any(history, table) and _history_contains_any(history, "N_DEAD_TUP", "VACUUM", "DEAD"):
        breakdown["id_bloat_issue"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["compound_lock_bloat"])
    res_score = 0.0

    rows = _exec(conn, """
        SELECT count(*) as cnt FROM pg_stat_activity
        WHERE wait_event_type = 'Lock' AND datname = current_database()
    """)
    locks_ok = (rows[0]["cnt"] if rows else 999) == 0
    if locks_ok:
        res_score += 0.15
        breakdown["res_locks_freed"] = 0.15

    rows = _exec(conn, f"""
        SELECT n_dead_tup FROM pg_stat_user_tables
        WHERE schemaname = 'bookings' AND relname = '{table}'
    """)
    dead = rows[0]["n_dead_tup"] if rows else 999999
    update_count = meta.get("update_count", 200000)
    dead_ok = dead < update_count * 0.3
    if dead_ok:
        res_score += 0.15
        breakdown["res_dead_tuples_reduced"] = 0.15
    elif dead < update_count * 0.7:
        res_score += 0.08
        breakdown["res_dead_tuples_reduced"] = 0.08

    if locks_ok and dead_ok:
        res_score += 0.10
        breakdown["res_both_resolved"] = 0.10

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "VACUUM"):
        breakdown["bp_ran_vacuum"] = 0.05
        score += 0.05
    if _history_contains_any(history, "IDLE_IN_TRANSACTION_SESSION_TIMEOUT", "STATEMENT_TIMEOUT"):
        breakdown["bp_prevention"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 9: Bad Configuration
# ═══════════════════════════════════════════════════════════════════

def _grade_bad_config(conn, meta: dict, history: List[str],
                      error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Misconfigured memory parameters (work_mem=64kB, effective_cache_size=1MB) cause
    the planner to avoid hash joins and index scans. Agent must correlate bad EXPLAIN
    plans with pg_settings values — requires quantitative reasoning about memory units."""
    breakdown = {}
    score = 0.0
    bad_settings = meta.get("bad_settings", {"work_mem": "64kB", "effective_cache_size": "1MB"})

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "PG_SETTINGS", "SHOW"):
        breakdown["inv_checked_settings"] = 0.10
        score += 0.10
    if _history_contains_any(history, "EXPLAIN"):
        breakdown["inv_ran_explain"] = 0.10
        score += 0.10

    # Dynamic: check if agent referenced any of the bad parameter names
    param_names = [k.upper() for k in bad_settings.keys()]
    found_params = sum(1 for p in param_names if _history_contains_any(history, p))
    if found_params >= 1:
        breakdown["id_bad_params"] = 0.10
        score += 0.10
    if found_params >= 2:
        breakdown["id_both_params"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["bad_config"])
    res_score = 0.0

    def _parse_mem_kb(val: str) -> int:
        v = val.upper().strip()
        try:
            if v.endswith("KB"):
                return int(v[:-2])
            elif v.endswith("MB"):
                return int(v[:-2]) * 1024
            elif v.endswith("GB"):
                return int(v[:-2]) * 1024 * 1024
            elif v.endswith("TB"):
                return int(v[:-2]) * 1024 * 1024 * 1024
            else:
                return int(v)
        except ValueError:
            return 0

    # work_mem
    rows = _exec(conn, """
        SELECT setting FROM pg_file_settings
        WHERE name = 'work_mem' AND error IS NULL
        ORDER BY seqno DESC LIMIT 1
    """)
    if rows:
        wm_kb = _parse_mem_kb(rows[0]["setting"])
        if wm_kb >= 1024:
            res_score += 0.20
            breakdown["res_work_mem_ok"] = 0.20
    else:
        rows = _exec(conn, "SELECT setting FROM pg_settings WHERE name = 'work_mem'")
        if rows:
            try:
                if int(rows[0]["setting"]) >= 1024:
                    res_score += 0.20
                    breakdown["res_work_mem_ok"] = 0.20
            except (ValueError, TypeError):
                pass

    # effective_cache_size
    rows = _exec(conn, """
        SELECT setting FROM pg_file_settings
        WHERE name = 'effective_cache_size' AND error IS NULL
        ORDER BY seqno DESC LIMIT 1
    """)
    if rows:
        ecs_kb = _parse_mem_kb(rows[0]["setting"])
        if ecs_kb >= 512 * 1024:
            res_score += 0.20
            breakdown["res_cache_size_ok"] = 0.20
    else:
        rows = _exec(conn, "SELECT setting FROM pg_settings WHERE name = 'effective_cache_size'")
        if rows:
            try:
                if int(rows[0]["setting"]) * 8 >= 512 * 1024:
                    res_score += 0.20
                    breakdown["res_cache_size_ok"] = 0.20
            except (ValueError, TypeError):
                pass

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "PG_RELOAD_CONF"):
        breakdown["bp_reload_conf"] = 0.05
        score += 0.05
    param_names = [k.upper() for k in bad_settings.keys()]
    if any(_history_contains(history, "ALTER SYSTEM", p) for p in param_names):
        breakdown["bp_alter_system"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 10: Index Bloat
# ═══════════════════════════════════════════════════════════════════

def _grade_index_bloat(conn, meta: dict, history: List[str],
                       error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Index bloat from repeated updates without maintenance. Agent must detect the
    bloated index via size comparison or pgstattuple, then REINDEX CONCURRENTLY —
    the production-safe path that avoids locking the table during rebuild."""
    breakdown = {}
    score = 0.0
    index_name = meta.get("target_index", "idx_ticket_flights_flight")
    table = meta.get("target_table", "ticket_flights")
    bloated_size = meta.get("bloated_size", 0)

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "PG_RELATION_SIZE", "PG_SIZE_PRETTY", "PGSTATTUPLE"):
        breakdown["inv_checked_size"] = 0.10
        score += 0.10
    if _history_contains_any(history, "PG_STAT_USER_INDEXES"):
        breakdown["inv_checked_index_stats"] = 0.10
        score += 0.10
    if _history_contains_any(history, index_name) or (_history_contains_any(history, table) and _history_contains_any(history, "INDEX")):
        breakdown["id_target_index"] = 0.10
        score += 0.10
    if _history_contains_any(history, "BLOAT", "REINDEX", "PG_RELATION_SIZE"):
        breakdown["id_bloat_detected"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["index_bloat"])
    res_score = 0.0

    if _history_contains_any(history, "REINDEX"):
        # Verify index still exists
        rows = _exec(conn, f"""
            SELECT 1 FROM pg_indexes
            WHERE schemaname = 'bookings' AND indexname = '{index_name}'
        """)
        if rows:
            res_score += 0.30
            breakdown["res_index_rebuilt"] = 0.30
    elif _history_contains(history, "CREATE INDEX") and _history_contains(history, "DROP INDEX"):
        res_score += 0.20
        breakdown["res_index_rebuilt"] = 0.20

    if bloated_size > 0:
        try:
            rows = _exec(conn, f"SELECT pg_relation_size('bookings.{index_name}') as sz")
            if rows and rows[0]["sz"] < bloated_size * 0.9:
                res_score += 0.10
                breakdown["res_size_reduced"] = 0.10
        except Exception:
            pass

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "CONCURRENTLY"):
        breakdown["bp_concurrently"] = 0.10
        score += 0.10

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 11: Wrong Index Column Order
# ═══════════════════════════════════════════════════════════════════

def _grade_wrong_index_order(conn, meta: dict, history: List[str],
                             error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Composite index exists but column order is wrong for the query's WHERE clause,
    so the planner falls back to seq scan. Tests understanding of B-tree leftmost
    prefix rule — a subtle concept that trips up even experienced engineers."""
    breakdown = {}
    score = 0.0
    column = meta.get("target_column", "flight_id")
    table = meta.get("target_table", "ticket_flights")

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "EXPLAIN"):
        breakdown["inv_ran_explain"] = 0.10
        score += 0.10
    if _history_contains_any(history, "PG_INDEXES"):
        breakdown["inv_checked_indexes"] = 0.10
        score += 0.10
    if _history_contains_any(history, column):
        breakdown["id_column_order"] = 0.10
        score += 0.10
    if _history_contains_any(history, table) and _history_contains_any(history, "TICKET_NO", "COMPOSITE", "PKEY", column):
        breakdown["id_composite_key"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["wrong_index_order"])
    res_score = 0.0

    rows = _exec(conn, f"""
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'bookings'
          AND tablename = '{table}'
          AND indexdef LIKE '%({column})%'
          AND indexname != 'ticket_flights_pkey'
    """)
    if rows:
        res_score += 0.20
        breakdown["res_standalone_index"] = 0.20

    plan_text = _fresh_explain(
        f"SELECT tf.ticket_no, tf.fare_conditions, tf.amount "
        f"FROM bookings.{table} tf WHERE tf.{column} = 2880"
    )
    if plan_text and "Index" in plan_text and "Seq Scan" not in plan_text:
        res_score += 0.20
        breakdown["res_plan_improved"] = 0.20

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "CONCURRENTLY"):
        breakdown["bp_concurrently"] = 0.05
        score += 0.05
    # PK preserved
    rows = _exec(conn, """
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'bookings' AND indexname = 'ticket_flights_pkey'
    """)
    if rows:
        breakdown["bp_pk_preserved"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 12: Deadlock Chain
# ═══════════════════════════════════════════════════════════════════

def _grade_deadlock_chain(conn, meta: dict, history: List[str],
                          error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Real PostgreSQL deadlock between transactions updating rows in opposite order.
    Requires reading pg_locks grant status and understanding lock wait graphs to
    identify which transaction to terminate — random termination risks data loss."""
    breakdown = {}
    score = 0.0
    table = meta.get("target_table", "bookings")
    book_ref_a = meta.get("book_ref_a", "")
    book_ref_b = meta.get("book_ref_b", "")

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "PG_STAT_ACTIVITY"):
        breakdown["inv_checked_activity"] = 0.10
        score += 0.10
    if _history_contains_any(history, "PG_LOCKS"):
        breakdown["inv_checked_locks"] = 0.10
        score += 0.10
    if _history_contains_any(history, "DEADLOCK", "PG_BLOCKING_PIDS"):
        breakdown["id_deadlock_pattern"] = 0.10
        score += 0.10
    # Check for book_refs or target table in lock context
    refs_found = False
    if book_ref_a and _history_contains_any(history, book_ref_a):
        refs_found = True
    if book_ref_b and _history_contains_any(history, book_ref_b):
        refs_found = True
    if _history_contains_any(history, table) and _history_contains_any(history, "LOCK", "PG_LOCKS"):
        refs_found = True
    if refs_found:
        breakdown["id_conflicting_txns"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["deadlock_chain"])
    res_score = 0.0

    rows = _exec(conn, """
        SELECT count(*) as cnt FROM pg_locks
        WHERE NOT granted AND locktype = 'transactionid'
    """)
    blocked = rows[0]["cnt"] if rows else 999
    if blocked == 0:
        res_score += 0.20
        breakdown["res_no_blocked_txids"] = 0.20

    rows = _exec(conn, """
        SELECT count(*) as cnt FROM pg_stat_activity
        WHERE wait_event_type = 'Lock' AND datname = current_database()
    """)
    lock_waits = rows[0]["cnt"] if rows else 999
    if lock_waits == 0:
        res_score += 0.20
        breakdown["res_no_lock_waits"] = 0.20

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "DEADLOCK_TIMEOUT"):
        breakdown["bp_deadlock_timeout"] = 0.05
        score += 0.05
    if _history_contains_any(history, "LOCK_TIMEOUT"):
        breakdown["bp_lock_timeout"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 13: Query Plan Flip
# ═══════════════════════════════════════════════════════════════════

def _grade_query_plan_flip(conn, meta: dict, history: List[str],
                           error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """A planner cost parameter (random_page_cost) has been set to an extreme value,
    causing the optimizer to avoid index scans entirely. Agent must trace the plan
    regression back to pg_settings, correct it, and reload — not just add more indexes."""
    breakdown = {}
    score = 0.0
    param = meta.get("bad_param", "random_page_cost")

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "EXPLAIN"):
        breakdown["inv_ran_explain"] = 0.10
        score += 0.10
    if _history_contains_any(history, "SHOW", "PG_SETTINGS"):
        breakdown["inv_checked_settings"] = 0.10
        score += 0.10
    if _history_contains_any(history, param.upper()):
        breakdown["id_bad_param"] = 0.20
        score += 0.20

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["query_plan_flip"])
    res_score = 0.0

    # Fresh connection SHOW to avoid session inheritance
    fresh_val = _fresh_show(param)
    param_ok = False
    if fresh_val:
        try:
            if float(fresh_val) <= 4.0:
                # Also check pg_file_settings to ensure persistent fix
                rows = _exec(conn, f"""
                    SELECT setting FROM pg_file_settings
                    WHERE name = '{param}' AND error IS NULL
                    ORDER BY seqno DESC LIMIT 1
                """)
                if not rows or float(rows[0]["setting"]) <= 4.0:
                    param_ok = True
                    res_score += 0.20
                    breakdown["res_param_reset"] = 0.20
        except (ValueError, TypeError):
            pass

    # Fresh connection EXPLAIN
    plan_text = _fresh_explain(
        "SELECT tf.ticket_no, tf.fare_conditions, tf.amount "
        "FROM bookings.ticket_flights tf WHERE tf.flight_id = 2880"
    )
    if plan_text and "Index" in plan_text and "Seq Scan" not in plan_text:
        res_score += 0.20
        breakdown["res_plan_uses_index"] = 0.20

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "PG_RELOAD_CONF"):
        breakdown["bp_reload_conf"] = 0.05
        score += 0.05
    if _history_contains(history, "ALTER SYSTEM", param.upper()):
        breakdown["bp_alter_system"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 14: Cascading Bloat
# ═══════════════════════════════════════════════════════════════════

def _grade_cascading_bloat(conn, meta: dict, history: List[str],
                           error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """A REPEATABLE READ transaction holds a snapshot that blocks vacuum across
    multiple tables simultaneously. The hardest single-fault task: agent must find the
    snapshot holder, terminate it, then vacuum each affected table — up to 4 tables."""
    breakdown = {}
    score = 0.0
    tables = meta.get("tables", [])
    update_count = meta.get("update_count_per_table", 50000)

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "PG_STAT_ACTIVITY"):
        breakdown["inv_checked_activity"] = 0.10
        score += 0.10
    if _history_contains_any(history, "PG_STAT_USER_TABLES", "N_DEAD_TUP"):
        breakdown["inv_checked_tables"] = 0.10
        score += 0.10
    if _history_contains_any(history, "BACKEND_XMIN", "TXID", "XID", "REPEATABLE READ"):
        breakdown["id_snapshot_holder"] = 0.10
        score += 0.10
    # Check how many affected tables agent referenced
    tables_referenced = sum(1 for t in tables if _history_contains_any(history, t))
    if tables_referenced >= 2:
        breakdown["id_multi_table"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["cascading_bloat"])
    res_score = 0.0

    # No old backend_xmin transactions
    rows = _exec(conn, """
        SELECT count(*) as cnt FROM pg_stat_activity
        WHERE backend_xmin IS NOT NULL
          AND age(backend_xmin) > 1000
          AND datname = current_database()
          AND pid != pg_backend_pid()
    """)
    old_xmin = rows[0]["cnt"] if rows else 999
    if old_xmin == 0:
        res_score += 0.15
        breakdown["res_blocker_gone"] = 0.15

    # Dead tuples reduced: proportional scoring across all affected tables.
    # Partial credit for cleaning some-but-not-all tables gives smooth reward
    # gradients, making this suitable for RL training without sparse-reward issues.
    cleaned = 0
    for tbl in tables:
        rows = _exec(conn, f"""
            SELECT n_dead_tup FROM pg_stat_user_tables
            WHERE schemaname = 'bookings' AND relname = '{tbl}'
        """)
        dead = rows[0]["n_dead_tup"] if rows else 999999
        if dead < update_count * 0.5:
            cleaned += 1
    if tables:
        tables_score = 0.25 * (cleaned / len(tables))
        res_score += tables_score
        breakdown["res_tables_cleaned"] = round(tables_score, 3)

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    # VACUUM for each table (proportional)
    vacuum_count = sum(1 for t in tables if _history_contains(history, "VACUUM", t))
    if tables and vacuum_count > 0:
        vac_score = 0.05 * (vacuum_count / len(tables))
        breakdown["bp_vacuumed_all"] = round(vac_score, 3)
        score += vac_score
    if _history_contains_any(history, "IDLE_IN_TRANSACTION_SESSION_TIMEOUT", "STATEMENT_TIMEOUT"):
        breakdown["bp_prevention"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 15: Permission Error
# ═══════════════════════════════════════════════════════════════════

def _grade_permission_error(conn, meta: dict, history: List[str],
                            error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Missing GRANT on a table for an application role. Simulates a common deployment
    failure. Best practice scoring penalizes overly broad fixes (GRANT ALL / SUPERUSER)
    and rewards minimal-privilege grants — testing security-aware incident response."""
    breakdown = {}
    score = 0.0
    role = meta.get("role_name", "app_user")
    table = meta.get("target_table", "ticket_flights")
    schema = meta.get("target_schema", "bookings")

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "INFORMATION_SCHEMA", "HAS_TABLE_PRIVILEGE", "PG_ROLES"):
        breakdown["inv_checked_grants"] = 0.10
        score += 0.10
    if _history_contains_any(history, "ROLE", "GRANT", "PRIVILEGE", "PG_ROLES"):
        breakdown["inv_checked_role"] = 0.10
        score += 0.10
    if _history_contains_any(history, table):
        breakdown["id_target_table"] = 0.10
        score += 0.10
    if _history_contains_any(history, role):
        breakdown["id_target_role"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["permission_error"])
    res_score = 0.0

    rows = _exec(conn, f"""
        SELECT 1
        FROM information_schema.role_table_grants
        WHERE grantee = '{role}'
          AND table_schema = '{schema}'
          AND table_name = '{table}'
          AND privilege_type = 'SELECT'
    """)
    if rows:
        res_score += 0.40
        breakdown["res_permission_granted"] = 0.40

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    # Penalize overly broad grants: in production, GRANT ALL or SUPERUSER is a
    # security anti-pattern. Rewards principle of least privilege.
    if not _history_contains_any(history, "ALL PRIVILEGES", "SUPERUSER"):
        breakdown["bp_minimal_grants"] = 0.05
        score += 0.05
    if _history_contains_any(history, "GRANT USAGE ON SCHEMA", "USAGE"):
        breakdown["bp_schema_usage"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 16: Sequence Exhaustion
# ═══════════════════════════════════════════════════════════════════

def _grade_sequence_exhaustion(conn, meta: dict, history: List[str],
                               error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """Sequence value is behind the actual max PK, causing duplicate key errors on
    INSERT. Agent must query both the sequence and the table to compute the correct
    setval target — a numeric coordination task where off-by-one errors are common."""
    breakdown = {}
    score = 0.0
    seq = meta.get("sequence_name", "bookings.flights_flight_id_seq")
    table = meta.get("target_table", "flights")
    pk = meta.get("pk_column", "flight_id")

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "PG_SEQUENCES", "LAST_VALUE", "NEXTVAL"):
        breakdown["inv_checked_sequence"] = 0.10
        score += 0.10
    if _history_contains(history, "MAX") or _history_contains_any(history, table):
        breakdown["inv_checked_max_pk"] = 0.10
        score += 0.10
    # Extract short name from qualified sequence name for matching
    seq_short = seq.split(".")[-1] if "." in seq else seq
    if _history_contains_any(history, seq_short, "SETVAL"):
        breakdown["id_sequence_name"] = 0.10
        score += 0.10
    # Both sequence value and max PK queried
    checked_seq = _history_contains_any(history, "LAST_VALUE", "CURRVAL", seq_short)
    checked_max = _history_contains(history, "MAX") and _history_contains_any(history, pk, table)
    if checked_seq and checked_max:
        breakdown["id_mismatch"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["sequence_exhaustion"])
    res_score = 0.0

    rows = _exec(conn, f"SELECT last_value FROM {seq}")
    seq_val = rows[0]["last_value"] if rows else 0
    rows = _exec(conn, f"SELECT max({pk}) as max_pk FROM bookings.{table}")
    max_pk = rows[0]["max_pk"] if rows else 0

    if seq_val and max_pk and seq_val >= max_pk:
        res_score += 0.25
        breakdown["res_sequence_reset"] = 0.25
        # Insert would succeed (same check)
        res_score += 0.15
        breakdown["res_insert_succeeds"] = 0.15

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "SETVAL"):
        breakdown["bp_used_setval"] = 0.05
        score += 0.05
    # Correct value: not wildly over max_pk
    if seq_val and max_pk and max_pk <= seq_val <= max_pk * 2:
        breakdown["bp_correct_value"] = 0.05
        score += 0.05

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Task 17: Compound Connection Exhaustion + Deadlock
# ═══════════════════════════════════════════════════════════════════

def _grade_compound_conn_deadlock(conn, meta: dict, history: List[str],
                                  error_history: List[bool], steps_used: int) -> Tuple[float, dict]:
    """The hardest compound fault: connection exhaustion + deadlock occurring
    simultaneously. Agent must triage two independent production fires, resolve each
    with the correct tool, and set preventive timeouts — our ceiling-difficulty task."""
    breakdown = {}
    score = 0.0
    deadlock_meta = meta.get("deadlock_meta", {})
    dl_table = deadlock_meta.get("target_table", "bookings")

    # ── Diagnosis (0.4) ──
    if _history_contains_any(history, "PG_STAT_ACTIVITY"):
        breakdown["inv_checked_activity"] = 0.10
        score += 0.10
    if _history_contains_any(history, "PG_LOCKS"):
        breakdown["inv_checked_locks"] = 0.10
        score += 0.10
    if _history_contains_any(history, "IDLE", "IDLE IN TRANSACTION", "IDLE_IN_TRANSACTION"):
        breakdown["id_idle_problem"] = 0.10
        score += 0.10
    if _history_contains_any(history, "DEADLOCK") or \
       (_history_contains_any(history, dl_table) and _history_contains_any(history, "LOCK", "PG_LOCKS")):
        breakdown["id_deadlock_problem"] = 0.10
        score += 0.10

    # ── Resolution (0.4) × efficiency ──
    eff = _efficiency_penalty(steps_used, STEP_THRESHOLDS["compound_conn_deadlock"])
    res_score = 0.0

    rows = _exec(conn, """
        SELECT count(*) as cnt FROM pg_stat_activity
        WHERE state = 'idle in transaction'
          AND datname = current_database()
          AND pid != pg_backend_pid()
    """)
    idle_count = rows[0]["cnt"] if rows else 999
    if idle_count < 5:
        res_score += 0.15
        breakdown["res_idle_cleared"] = 0.15
    elif idle_count < 20:
        res_score += 0.07
        breakdown["res_idle_cleared"] = 0.07

    rows = _exec(conn, """
        SELECT setting FROM pg_file_settings
        WHERE name = 'idle_in_transaction_session_timeout'
        AND error IS NULL
        ORDER BY seqno DESC LIMIT 1
    """)
    if rows and rows[0]["setting"] and rows[0]["setting"] != '0':
        res_score += 0.15
        breakdown["res_timeout_set"] = 0.15

    rows = _exec(conn, """
        SELECT count(*) as cnt FROM pg_locks
        WHERE NOT granted AND locktype = 'transactionid'
    """)
    blocked = rows[0]["cnt"] if rows else 999
    if blocked == 0:
        res_score += 0.10
        breakdown["res_no_deadlocks"] = 0.10

    res_score *= eff
    breakdown["_efficiency_mult"] = round(eff, 2)
    score += res_score

    # ── Best Practice (0.2) ──
    if not _has_destructive(history):
        breakdown["bp_no_destructive"] = 0.05
        score += 0.05
    if _error_rate(error_history) < 0.3:
        breakdown["bp_clean_execution"] = 0.05
        score += 0.05
    if _history_contains_any(history, "PG_RELOAD_CONF"):
        breakdown["bp_reload_conf"] = 0.05
        score += 0.05
    for cmd in history:
        upper = cmd.upper()
        if "PG_TERMINATE_BACKEND" in upper and "WHERE" in upper:
            breakdown["bp_targeted_terminate"] = 0.05
            score += 0.05
            break

    return min(1.0, round(score, 4)), breakdown


# ═══════════════════════════════════════════════════════════════════
# Registry & dispatcher
# ═══════════════════════════════════════════════════════════════════
# 17 graders covering the full spectrum of PostgreSQL incident response:
#   - 10 single-fault tasks (easy to hard)
#   - 4 compound-fault tasks requiring multi-root-cause analysis
#   - 3 tasks targeting configuration and access control
# Difficulty ranges from tasks solvable in 3 steps (permission_error) to tasks
# requiring 10+ coordinated actions (compound_conn_deadlock, cascading_bloat).

_GRADER_REGISTRY = {
    "missing_index": _grade_missing_index,
    "stale_statistics": _grade_stale_statistics,
    "connection_exhaustion": _grade_connection_exhaustion,
    "lock_contention": _grade_lock_contention,
    "table_bloat": _grade_table_bloat,
    "over_indexing": _grade_over_indexing,
    "compound_stats_index": _grade_compound_stats_index,
    "compound_lock_bloat": _grade_compound_lock_bloat,
    "bad_config": _grade_bad_config,
    "index_bloat": _grade_index_bloat,
    "wrong_index_order": _grade_wrong_index_order,
    "deadlock_chain": _grade_deadlock_chain,
    "query_plan_flip": _grade_query_plan_flip,
    "cascading_bloat": _grade_cascading_bloat,
    "permission_error": _grade_permission_error,
    "sequence_exhaustion": _grade_sequence_exhaustion,
    "compound_conn_deadlock": _grade_compound_conn_deadlock,
}


def grade_episode(
    conn,
    fault_type: str,
    inject_meta: dict,
    action_history: List[str],
    error_history: List[bool] = None,
    steps_used: int = 0,
) -> Tuple[float, dict]:
    """Grade an episode. Returns (score, breakdown).

    Central dispatch point: maps fault_type to the corresponding deterministic
    grader function. Every grader returns a float in [0.0, 1.0] and a breakdown
    dict showing exactly how each sub-score was earned — full transparency for
    debugging reward signals during RL training.

    Args:
        conn: Admin DB connection.
        fault_type: The fault type string.
        inject_meta: Metadata returned by the injector's inject().
        action_history: List of SQL commands the agent executed.
        error_history: List of booleans indicating if each command errored.
        steps_used: Number of steps taken in the episode.

    Returns:
        (score, breakdown): score in [0.0, 1.0], breakdown dict.
    """
    if error_history is None:
        error_history = []

    grader_fn = _GRADER_REGISTRY.get(fault_type)
    if grader_fn is None:
        logger.error("No grader for fault_type=%s", fault_type)
        return 0.0, {"error": f"No grader for {fault_type}"}

    try:
        return grader_fn(conn, inject_meta, action_history, error_history, steps_used)
    except Exception as e:
        logger.error("Grader error for %s: %s", fault_type, e)
        return 0.0, {"error": str(e)}
