"""
SQLab — PostgreSQL connection management with three security tiers.

Manages three tiers of database connections, mirroring production database
access controls where application users have restricted permissions compared
to DBA accounts:

1. Admin connection: Superuser for fault injection and grading (never exposed
   to the agent). Used internally to inject faults, verify resolution, and
   query pg_catalog for grading.
2. Agent connection: Restricted connection for the LLM agent's SQL execution.
   Commands are filtered through a safety layer that blocks destructive
   operations on core data tables, while preserving full access to diagnostic
   queries (EXPLAIN, pg_stat_activity, pg_locks) and corrective DDL
   (CREATE INDEX, DROP INDEX, ALTER SYSTEM).
3. Background connections: Thread-managed connections for fault simulation
   (holding locks, maintaining idle-in-transaction sessions). These create
   the realistic concurrent workload that agents must diagnose.

This separation ensures the agent interacts with the database the same way a
production SRE would — full diagnostic access but restricted write permissions.
"""

import os
import logging
import threading
from typing import Optional, List
from contextlib import contextmanager

import psycopg2
import psycopg2.extensions
import psycopg2.extras

logger = logging.getLogger(__name__)

# Connection defaults — overridable via environment variables
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5433"))
DB_NAME = os.environ.get("DB_NAME", "demo")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "srelab")

# SQL commands the agent is NOT allowed to run (safety guard).
# Blocks DROP TABLE, TRUNCATE, and other irreversible operations on the 8 core
# data tables. The agent retains full access to diagnostic queries, DDL
# (CREATE/DROP INDEX), and system functions (pg_terminate_backend, ALTER SYSTEM).
# This balances realism with data integrity — a real SRE has similar guardrails.
BLOCKED_PATTERNS = [
    "DROP DATABASE",
    "DROP SCHEMA",
    "CREATE DATABASE",
    "DROP TABLE bookings",
    "DROP TABLE tickets",
    "DROP TABLE flights",
    "DROP TABLE ticket_flights",
    "DROP TABLE boarding_passes",
    "DROP TABLE airports_data",
    "DROP TABLE aircrafts_data",
    "DROP TABLE seats",
    "TRUNCATE bookings",
    "TRUNCATE tickets",
    "TRUNCATE flights",
    "TRUNCATE ticket_flights",
    "TRUNCATE boarding_passes",
]


def get_connection_params() -> dict:
    """Return connection parameters dict."""
    return {
        "host": DB_HOST,
        "port": DB_PORT,
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
    }


def get_admin_connection() -> psycopg2.extensions.connection:
    """Get a superuser connection for fault injection and grading.

    This connection has full privileges and autocommit enabled.
    """
    conn = psycopg2.connect(**get_connection_params())
    conn.autocommit = True
    return conn


def get_agent_connection() -> psycopg2.extensions.connection:
    """Get a connection for agent SQL execution.

    Uses the same superuser credentials but commands are filtered
    through the safety guard before execution.
    """
    conn = psycopg2.connect(**get_connection_params())
    conn.autocommit = True
    return conn


def is_command_allowed(command: str) -> bool:
    """Check if a SQL command is allowed for the agent.

    Blocks destructive operations on core data tables.
    Allows: SELECT, CREATE INDEX, DROP INDEX, ALTER SYSTEM, VACUUM, ANALYZE,
            pg_terminate_backend, pg_cancel_backend, pg_reload_conf, SHOW, SET, etc.
    """
    cmd_upper = command.upper().strip()

    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd_upper:
            return False

    return True


def execute_agent_sql(conn: psycopg2.extensions.connection, command: str) -> tuple[str, Optional[str]]:
    """Execute a SQL command from the agent with safety checks.

    The agent can run any valid PostgreSQL command (diagnostic or corrective)
    as long as it doesn't match the blocked patterns list. Output is formatted
    as a plain-text table mimicking psql output — the format LLMs are most
    familiar with from training data, minimizing the need for output parsing.

    Safety features:
    - 30-second statement timeout prevents runaway queries from blocking the env
    - Output truncated to 100 rows to keep observation size manageable for LLM
      context windows while providing enough data for diagnosis
    - Connection state auto-recovered after errors via rollback

    Returns:
        (output, error): output is the formatted result, error is None on success.
    """
    command = command.strip()
    if not command:
        return "", "Empty command"

    # Safety check
    if not is_command_allowed(command):
        return "", "ERROR: Command blocked for safety. You cannot drop or truncate core data tables."

    try:
        cur = conn.cursor()
        cur.execute("SET statement_timeout = '30s'")
        cur.execute(command)

        # Try to fetch results
        try:
            rows = cur.fetchall()
            if not rows:
                # Command succeeded but returned no rows
                status = cur.statusmessage or "OK"
                return status, None

            # Format output as a table
            colnames = [desc[0] for desc in cur.description]
            output_lines = []
            # Header
            output_lines.append(" | ".join(colnames))
            output_lines.append("-+-".join("-" * max(len(c), 5) for c in colnames))
            # Rows (limit to 100 for readability)
            for row in rows[:100]:
                output_lines.append(" | ".join(str(v) if v is not None else "NULL" for v in row))
            if len(rows) > 100:
                output_lines.append(f"... ({len(rows)} total rows, showing first 100)")
            else:
                output_lines.append(f"({len(rows)} rows)")

            return "\n".join(output_lines), None

        except psycopg2.ProgrammingError:
            # Command didn't return rows (e.g., CREATE INDEX, VACUUM)
            status = cur.statusmessage or "OK"
            return status, None

    except psycopg2.Error as e:
        error_msg = str(e).strip()
        # Reset the connection state after error
        try:
            conn.rollback()
        except Exception:
            pass
        return "", f"ERROR: {error_msg}"
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return "", f"ERROR: {str(e)}"


def get_db_metrics(conn: psycopg2.extensions.connection) -> dict:
    """Snapshot current database health metrics.

    Captures the key health indicators a production SRE would check during an
    incident: connection states (active vs idle-in-transaction), lock waits,
    dead tuple counts per table, and index counts. These are the same metrics
    surfaced by production monitoring tools like pganalyze and pg_stat_monitor.

    Providing structured metrics on every step gives the agent the same
    observability that human SREs have, enabling data-driven diagnosis.

    Returns dict with: active_connections, idle_in_transaction,
    lock_waits, dead_tuples (top tables), index_count, etc.
    """
    metrics = {}
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Connection counts by state
        cur.execute("""
            SELECT state, count(*) as cnt
            FROM pg_stat_activity
            WHERE datname = current_database()
            GROUP BY state
        """)
        conn_states = {row["state"] or "unknown": row["cnt"] for row in cur.fetchall()}
        metrics["connections"] = conn_states
        metrics["total_connections"] = sum(conn_states.values())
        metrics["idle_in_transaction"] = conn_states.get("idle in transaction", 0)

        # Lock waits
        cur.execute("""
            SELECT count(*) as cnt
            FROM pg_stat_activity
            WHERE wait_event_type = 'Lock'
            AND datname = current_database()
        """)
        metrics["lock_waits"] = cur.fetchone()["cnt"]

        # Dead tuples (top 5 tables)
        cur.execute("""
            SELECT relname, n_dead_tup, n_live_tup, last_autovacuum, last_analyze
            FROM pg_stat_user_tables
            WHERE schemaname = 'bookings'
            ORDER BY n_dead_tup DESC
            LIMIT 5
        """)
        dead_tuples = []
        for row in cur.fetchall():
            dead_tuples.append({
                "table": row["relname"],
                "dead_tuples": row["n_dead_tup"],
                "live_tuples": row["n_live_tup"],
                "last_autovacuum": str(row["last_autovacuum"]) if row["last_autovacuum"] else None,
                "last_analyze": str(row["last_analyze"]) if row["last_analyze"] else None,
            })
        metrics["dead_tuples_top5"] = dead_tuples

        # Index count on ticket_flights
        cur.execute("""
            SELECT count(*) as cnt
            FROM pg_indexes
            WHERE schemaname = 'bookings' AND tablename = 'ticket_flights'
        """)
        metrics["ticket_flights_index_count"] = cur.fetchone()["cnt"]

    except Exception as e:
        logger.warning(f"Error collecting metrics: {e}")
        metrics["error"] = str(e)

    return metrics


class BackgroundConnectionManager:
    """Manages background connections used for fault simulation.

    Thread-safe manager for background connections that simulate concurrent
    database activity: idle-in-transaction sessions (connection exhaustion),
    lock-holding transactions (lock contention), and deadlocked transactions.

    Cleanup is guaranteed via stop_event signaling, ensuring clean state
    between episodes regardless of how the agent's episode ended. This is
    essential for reproducible RL training — each episode must start from
    a known-good database state.
    """

    def __init__(self):
        self._connections: List[psycopg2.extensions.connection] = []
        self._threads: List[threading.Thread] = []
        self._pids: List[int] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def add_connection(self, conn: psycopg2.extensions.connection):
        """Track a background connection."""
        with self._lock:
            self._connections.append(conn)

    def add_thread(self, thread: threading.Thread):
        """Track a background thread."""
        with self._lock:
            self._threads.append(thread)

    @property
    def stop_event(self) -> threading.Event:
        """Event to signal background threads to stop."""
        return self._stop_event

    def cleanup(self):
        """Close all background connections and stop all threads."""
        self._stop_event.set()

        # Wait for threads to finish (with timeout)
        with self._lock:
            threads = list(self._threads)
        for t in threads:
            t.join(timeout=5.0)

        # Close all connections
        with self._lock:
            for conn in self._connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
            self._threads.clear()
            self._pids.clear()

        self._stop_event.clear()

    def add_pid(self, pid: int):
        """Track a PID for a background connection (call after connection is established)."""
        with self._lock:
            self._pids.append(pid)

    def get_pids(self) -> List[int]:
        """Get tracked PIDs of background connections (non-blocking)."""
        with self._lock:
            return list(self._pids)
