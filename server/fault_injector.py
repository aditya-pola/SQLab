"""
SQLab — Fault injectors for PostgreSQL failure simulation.

8 fault injector classes simulate real PostgreSQL failure modes encountered
in production. Each modeled on documented incident patterns (missing indexes,
connection exhaustion, lock chains, bloated tables, misconfigured GUCs).
Every injector provides:
    inject()         — create the fault in a live PostgreSQL instance
    check_resolved() — verify the fix via actual DB state (pg_catalog queries)
    cleanup()        — restore DB state for episode independence
    get_prebake_sql() — optional fast idempotent injection for RL throughput

Pre-bake architecture: faults expressible as pure SQL provide get_prebake_sql(),
enabling sub-5-second resets instead of ~120s live injection. Faults requiring
background threads (lock contention, connection exhaustion) use a hybrid
approach: pre-baked data setup + live thread creation.

Resolution verification queries actual PostgreSQL catalog state in every case.
The agent can use any valid approach to fix the problem; the grader only checks
end state. This makes the environment robust against reward hacking.
"""

import logging
import random
import threading
import time
from typing import Dict, Any, Optional

import psycopg2

from sqlab.server.db import (
    get_connection_params,
    BackgroundConnectionManager,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════

class BaseFaultInjector:
    """Base class for fault injectors."""

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        raise NotImplementedError

    def check_resolved(self, conn, meta: dict) -> bool:
        raise NotImplementedError

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        raise NotImplementedError

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        """Return {"inject": [...], "cleanup": [...]} or None if not pre-bakeable.

        Pre-bake interface for RL training throughput. Returns idempotent SQL
        that avoids live thread setup on every reset. Pre-bakeable faults are
        those whose entire fault state can be expressed as SQL statements (no
        background threads needed). Subclasses override this to provide their SQL.
        """
        return None

    # ── helpers ──────────────────────────────────────────────────
    @staticmethod
    def _exec(conn, sql: str, fetch: bool = False):
        """Execute SQL on an autocommit connection. Optionally fetch results."""
        cur = conn.cursor()
        cur.execute(sql)
        if fetch:
            return cur.fetchall()
        return None


# ═══════════════════════════════════════════════════════════════════
# 1. Missing Index
# ═══════════════════════════════════════════════════════════════════

class MissingIndexInjector(BaseFaultInjector):
    """Models the #1 most common PostgreSQL performance issue in production.

    Drops index on ticket_flights(flight_id), forcing sequential scans on
    8.4M rows. Real-world analogue: post-migration index omission or
    accidental DROP INDEX in a deployment script.
    """

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        return {
            "inject": [
                "DROP INDEX IF EXISTS bookings.idx_ticket_flights_flight",
                # Drop ALL non-PK indexes on (flight_id) — leftovers from other tasks
                """DO $$ DECLARE r RECORD; BEGIN
                    FOR r IN SELECT indexname FROM pg_indexes
                        WHERE schemaname = 'bookings' AND tablename = 'ticket_flights'
                        AND indexdef LIKE '%(flight_id)%'
                        AND indexname != 'ticket_flights_pkey'
                    LOOP EXECUTE 'DROP INDEX IF EXISTS bookings.' || r.indexname; END LOOP;
                END $$""",
            ],
            "cleanup": [
                "CREATE INDEX IF NOT EXISTS idx_ticket_flights_flight ON bookings.ticket_flights(flight_id)",
            ],
            "meta": {
                "index_name": "idx_ticket_flights_flight",
                "target_table": "ticket_flights",
                "target_column": "flight_id",
            },
        }

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        index_name = params["index_name"]
        target_column = params["target_column"]
        target_table = params["target_table"]

        # Drop ALL non-PK indexes on the target column (not just the named one)
        self._exec(conn, f"""
            DO $$ DECLARE r RECORD; BEGIN
                FOR r IN SELECT indexname FROM pg_indexes
                    WHERE schemaname = 'bookings' AND tablename = '{target_table}'
                    AND indexdef LIKE '%({target_column})%'
                    AND indexname != 'ticket_flights_pkey'
                LOOP EXECUTE 'DROP INDEX IF EXISTS bookings.' || r.indexname; END LOOP;
            END $$
        """)

        logger.info("MissingIndex: dropped all %s indexes on %s", target_column, target_table)
        return {
            "index_name": index_name,
            "target_table": target_table,
            "target_column": target_column,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Resolution check queries pg_indexes for any index on the target column.
        Agent can use CREATE INDEX with any name or column list — grader only
        checks that an efficient access path exists, not the exact command used.
        """
        rows = self._exec(conn, f"""
            SELECT 1 FROM pg_indexes
            WHERE schemaname = 'bookings'
              AND tablename = '{meta["target_table"]}'
              AND indexdef LIKE '%({meta["target_column"]}%'
        """, fetch=True)
        return bool(rows)

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Re-create the index if it's still missing."""
        try:
            self._exec(conn,
                f"CREATE INDEX IF NOT EXISTS {meta['index_name']} "
                f"ON bookings.{meta['target_table']}({meta['target_column']})"
            )
        except Exception as e:
            logger.warning("MissingIndex cleanup error: %s", e)


# ═══════════════════════════════════════════════════════════════════
# 2. Stale Statistics
# ═══════════════════════════════════════════════════════════════════

class StaleStatsInjector(BaseFaultInjector):
    """Simulates post-migration statistics drift.

    Mass UPDATE flights + delete pg_statistic entries, forcing the query
    planner to use default selectivity estimates. Causes catastrophic plan
    regression (hash joins where nested loops are optimal). Real-world
    analogue: large batch ETL that changes data distribution without ANALYZE.
    """

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        return {
            "inject": [
                "UPDATE bookings.flights SET status = 'Delayed' WHERE flight_id IN (SELECT flight_id FROM bookings.flights WHERE status = 'Arrived' LIMIT 100000)",
                "DELETE FROM pg_statistic WHERE starelid = 'bookings.flights'::regclass",
                "SELECT pg_stat_reset_single_table_counters('bookings.flights'::regclass)",
            ],
            "cleanup": [
                "UPDATE bookings.flights SET status = 'Arrived' WHERE status = 'Delayed'",
                "ANALYZE bookings.flights",
            ],
            "meta": {
                "target_table": "flights",
                "status_from": "Arrived",
                "status_to": "Delayed",
                "update_count": 100000,
            },
        }

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        table = params["target_table"]
        status_from = params["update_status_from"]
        status_to = params["update_status_to"]
        count = params["update_count"]

        # Mass update to change status distribution
        self._exec(conn, f"""
            UPDATE bookings.{table} SET status = '{status_to}'
            WHERE flight_id IN (
                SELECT flight_id FROM bookings.{table}
                WHERE status = '{status_from}' LIMIT {count}
            )
        """)

        # Delete statistics for the flights table to make planner use defaults
        # This makes estimates wildly off
        self._exec(conn, f"""
            DELETE FROM pg_statistic
            WHERE starelid = 'bookings.{table}'::regclass
        """)

        # Clear last_analyze timestamp so check_resolved doesn't see stale value
        # from a previous episode's cleanup ANALYZE
        self._exec(conn, f"SELECT pg_stat_reset_single_table_counters('bookings.{table}'::regclass)")

        logger.info("StaleStats: updated %d rows %s→%s, deleted pg_statistic", count, status_from, status_to)
        return {
            "target_table": table,
            "status_from": status_from,
            "status_to": status_to,
            "update_count": count,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Resolution verified by querying pg_stat_user_tables.last_analyze.
        Agent can run ANALYZE on any subset of columns — grader checks
        timestamp, not the specific ANALYZE command used.
        """
        rows = self._exec(conn, f"""
            SELECT last_analyze FROM pg_stat_user_tables
            WHERE schemaname = 'bookings' AND relname = '{meta["target_table"]}'
              AND last_analyze > now() - interval '30 minutes'
        """, fetch=True)
        return bool(rows)

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Revert the mass update and re-analyze."""
        table = meta["target_table"]
        status_from = meta["status_from"]
        status_to = meta["status_to"]
        try:
            self._exec(conn, f"""
                UPDATE bookings.{table} SET status = '{status_from}'
                WHERE status = '{status_to}'
            """)
            self._exec(conn, f"ANALYZE bookings.{table}")
        except Exception as e:
            logger.warning("StaleStats cleanup error: %s", e)


# ═══════════════════════════════════════════════════════════════════
# 3. Connection Exhaustion
# ═══════════════════════════════════════════════════════════════════

class ConnectionExhaustionInjector(BaseFaultInjector):
    """Creates N idle-in-transaction connections consuming connection slots.

    Models the most common production P1 incident: connection pool exhaustion
    from leaked connections or missing idle_in_transaction_session_timeout.
    Agent must both terminate existing sessions AND set preventive timeout.
    """

    # Thread-only fault — not pre-bakeable. Connection exhaustion requires
    # actual open TCP connections, which cannot be expressed as SQL. Falls
    # back to live injection (~3s) which is still fast enough for training.

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        base = params["num_connections_base"]
        rng = params.get("num_connections_range", 10)
        num_conns = base + random.randint(0, rng)
        # Cap to avoid exceeding max_connections entirely (leave room for admin)
        num_conns = min(num_conns, 90)

        conn_params = get_connection_params()
        opened = 0
        for i in range(num_conns):
            try:
                c = psycopg2.connect(**conn_params)
                c.autocommit = False
                cur = c.cursor()
                cur.execute("BEGIN")
                cur.execute("SELECT 1")
                # Connection is now in "idle in transaction" state
                bg_manager.add_connection(c)
                opened += 1
            except psycopg2.OperationalError:
                # max_connections reached
                logger.info("ConnectionExhaustion: stopped at %d (max reached)", opened)
                break

        logger.info("ConnectionExhaustion: opened %d idle-in-tx connections", opened)
        return {
            "num_connections": opened,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Two-part resolution: idle-in-transaction count < 5 AND preventive
        timeout configured via ALTER SYSTEM. Requires both remediation AND
        prevention — mirrors real SRE practice of fixing now + preventing
        recurrence.
        """
        rows = self._exec(conn, """
            SELECT count(*) FROM pg_stat_activity
            WHERE state = 'idle in transaction'
              AND datname = current_database()
              AND pid != pg_backend_pid()
        """, fetch=True)
        idle_count = rows[0][0] if rows else 0
        if idle_count >= 5:
            return False

        # Also require timeout to be set (preventive measure)
        rows = self._exec(conn, """
            SELECT setting FROM pg_file_settings
            WHERE name = 'idle_in_transaction_session_timeout'
              AND error IS NULL
            ORDER BY seqno DESC LIMIT 1
        """, fetch=True)
        if rows and rows[0][0] and rows[0][0] != '0':
            return True

        return False

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Close all background connections and reset timeout."""
        bg_manager.cleanup()
        try:
            self._exec(conn, "ALTER SYSTEM RESET idle_in_transaction_session_timeout")
            self._exec(conn, "SELECT pg_reload_conf()")
        except Exception as e:
            logger.warning("ConnectionExhaustion cleanup error: %s", e)


# ═══════════════════════════════════════════════════════════════════
# 4. Lock Contention
# ═══════════════════════════════════════════════════════════════════

class LockContentionInjector(BaseFaultInjector):
    """Simulates production lock chain: one blocker holds row lock, N waiters
    queue behind it. Agent must identify the root blocker via pg_locks /
    pg_stat_activity, not just kill victim sessions. Real-world analogue:
    long-running admin query holding AccessExclusiveLock during peak traffic.
    """

    # Thread-only fault — not pre-bakeable. Lock contention requires actual
    # backend processes holding row locks, which cannot be faked with SQL.

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        book_refs = params["book_refs"]
        num_waiters = params.get("num_waiters", 3)
        # Pick a book_ref for the blocker
        blocker_ref = book_refs[0]

        conn_params = get_connection_params()

        # Start blocker thread — holds a row lock and stays idle
        blocker_conn = psycopg2.connect(**conn_params)
        blocker_conn.autocommit = False
        bg_manager.add_connection(blocker_conn)

        blocker_pid = [None]

        def hold_lock():
            try:
                cur = blocker_conn.cursor()
                cur.execute("BEGIN")
                cur.execute(f"UPDATE bookings.bookings SET total_amount = total_amount WHERE book_ref = '{blocker_ref}'")
                cur.execute("SELECT pg_backend_pid()")
                blocker_pid[0] = cur.fetchone()[0]
                # Hold lock until stop event
                while not bg_manager.stop_event.wait(timeout=1.0):
                    pass
            except Exception as e:
                logger.debug("Blocker thread ended: %s", e)

        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        bg_manager.add_thread(t)
        # Wait for blocker to acquire the lock
        time.sleep(1.0)

        # Start waiter threads that will be blocked
        # Use short lock_timeout so they auto-cancel after blocker dies
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
                        cur.execute(f"UPDATE bookings.bookings SET total_amount = total_amount WHERE book_ref = '{ref}'")
                    except Exception as e:
                        logger.debug("Waiter thread ended: %s", e)
                    finally:
                        try:
                            c.rollback()
                        except Exception:
                            pass

                wt = threading.Thread(target=wait_on_lock, daemon=True)
                wt.start()
                bg_manager.add_thread(wt)
            except Exception as e:
                logger.warning("Failed to create waiter %d: %s", i, e)

        time.sleep(0.5)

        logger.info("LockContention: blocker PID %s on book_ref=%s, %d waiters",
                     blocker_pid[0], blocker_ref, num_waiters)
        return {
            "blocker_pid": blocker_pid[0],
            "blocker_ref": blocker_ref,
            "num_waiters": num_waiters,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Resolution verified by checking system-wide lock state — no lock
        waiters and no ungranted relation locks. Matches grader logic.
        """
        rows = self._exec(conn, """
            SELECT count(*) FROM pg_stat_activity
            WHERE wait_event_type = 'Lock'
              AND datname = current_database()
        """, fetch=True)
        lock_waits = rows[0][0] if rows else 999
        if lock_waits > 0:
            return False

        rows = self._exec(conn, """
            SELECT count(*) FROM pg_locks
            WHERE NOT granted AND locktype = 'relation'
        """, fetch=True)
        blocked = rows[0][0] if rows else 999
        return blocked == 0

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Stop background threads and close connections."""
        bg_manager.cleanup()


# ═══════════════════════════════════════════════════════════════════
# 5. Table Bloat / Vacuum Stuck
# ═══════════════════════════════════════════════════════════════════

class TableBloatInjector(BaseFaultInjector):
    """Creates 200K+ dead tuples while long-running transaction holds
    backend_xmin, preventing autovacuum from reclaiming space. Models batch
    jobs with forgotten open transactions — a common production pattern where
    a developer's debug session or reporting query blocks vacuum for hours.
    """

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        """Hybrid pre-bake: mass UPDATE expressed as idempotent SQL, but the
        transaction-holding thread must be created live (needs_threads=True).
        This hybrid approach gives ~80% of the speedup of full pre-baking.
        """
        return {
            "inject": [
                "UPDATE bookings.bookings SET total_amount = total_amount + 0.01 WHERE book_ref IN (SELECT book_ref FROM bookings.bookings LIMIT 10000)",
                "SELECT pg_stat_force_next_flush()",
            ],
            "cleanup": [
                # No VACUUM needed — re-running inject just adds more dead tuples.
                # The agent is expected to VACUUM as part of solving the fault.
                "SELECT 1",
            ],
            "needs_threads": True,
            "meta": {
                "target_table": "bookings",
                "update_count": 10000,
            },
        }

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        table = params["target_table"]
        dead_base = params["dead_tuple_count_base"]
        dead_range = params.get("dead_tuple_count_range", 50000)
        update_count = dead_base + random.randint(0, dead_range)

        conn_params = get_connection_params()

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
                # Hold transaction open
                while not bg_manager.stop_event.wait(timeout=1.0):
                    pass
            except Exception as e:
                logger.debug("Blocker tx thread ended: %s", e)

        t = threading.Thread(target=hold_tx, daemon=True)
        t.start()
        bg_manager.add_thread(t)
        time.sleep(0.5)

        # Mass update to create dead tuples (done on admin conn, committed)
        self._exec(conn, f"""
            UPDATE bookings.{table} SET total_amount = total_amount + 0.01
            WHERE book_ref IN (
                SELECT book_ref FROM bookings.{table} LIMIT {update_count}
            )
        """)

        # Force stats collector to update
        self._exec(conn, f"SELECT pg_stat_force_next_flush()")
        time.sleep(0.5)

        logger.info("TableBloat: %d dead tuples in %s, blocker PID %s",
                     update_count, table, blocker_pid[0])
        return {
            "target_table": table,
            "update_count": update_count,
            "blocker_pid": blocker_pid[0],
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Resolution checks both: (1) no old backend_xmin transactions, and
        (2) dead tuples reduced by 70%+ via pg_stat_user_tables. Matches grader
        thresholds to prevent resolved/score mismatch.
        """
        table = meta["target_table"]
        # Check no long-running txns with old backend_xmin (matches grader)
        rows = self._exec(conn, """
            SELECT count(*) FROM pg_stat_activity
            WHERE backend_xmin IS NOT NULL
              AND age(backend_xmin) > 1000
              AND datname = current_database()
              AND pid != pg_backend_pid()
        """, fetch=True)
        old_xmin = rows[0][0] if rows else 999
        if old_xmin > 0:
            return False

        # Check dead tuples reduced (threshold matches grader's 0.3)
        rows = self._exec(conn, f"""
            SELECT n_dead_tup FROM pg_stat_user_tables
            WHERE schemaname = 'bookings' AND relname = '{table}'
        """, fetch=True)
        dead = rows[0][0] if rows else 0
        return dead < meta.get("update_count", 200000) * 0.3

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Stop blocker, vacuum the table."""
        bg_manager.cleanup()
        table = meta["target_table"]
        try:
            self._exec(conn, f"VACUUM bookings.{table}")
        except Exception as e:
            logger.warning("TableBloat cleanup vacuum error: %s", e)


# ═══════════════════════════════════════════════════════════════════
# 6. Over-Indexing
# ═══════════════════════════════════════════════════════════════════

class OverIndexingInjector(BaseFaultInjector):
    """Creates 8-12 unnecessary indexes with zero scans on ticket_flights.

    Tests whether the agent can distinguish useful indexes from dead weight
    using pg_stat_user_indexes (idx_scan = 0). Real-world analogue: ORM
    auto-generated indexes or cargo-culted index creation over years of
    schema evolution. Over-indexing wastes write I/O and bloats WAL.
    """

    # Fixed set of junk indexes for pre-baking (no randomization).
    # 8 indexes on the full table — slower to create but matches the live fault closely.
    PREBAKE_JUNK_INDEXES = [
        ("idx_tf_junk1", "CREATE INDEX idx_tf_junk1 ON bookings.ticket_flights(amount) WHERE flight_id < 10000"),
        ("idx_tf_junk2", "CREATE INDEX idx_tf_junk2 ON bookings.ticket_flights(fare_conditions) WHERE flight_id < 10000"),
        ("idx_tf_junk3", "CREATE INDEX idx_tf_junk3 ON bookings.ticket_flights(amount, fare_conditions) WHERE flight_id < 10000"),
    ]

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        # Use IF NOT EXISTS so re-running is fast if indexes already exist
        inject_sql = []
        cleanup_sql = []
        junk_names = []
        for idx_name, create_sql in cls.PREBAKE_JUNK_INDEXES:
            inject_sql.append(create_sql.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS "))
            cleanup_sql.append(f"DROP INDEX IF EXISTS bookings.{idx_name}")
            junk_names.append(idx_name)
        inject_sql.append("SELECT pg_stat_reset()")
        return {
            "inject": inject_sql,
            "cleanup": cleanup_sql,
            "meta": {
                "target_table": "ticket_flights",
                "junk_indexes": junk_names,
            },
        }

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        num_base = params.get("num_junk_indexes_base", 8)
        num_range = params.get("num_junk_indexes_range", 5)
        num_junk = num_base + random.randint(0, num_range)
        pool = params["junk_pool"]

        # Select a random subset
        selected = random.sample(pool, min(num_junk, len(pool)))

        created = []
        for idx_name, create_sql in selected:
            try:
                self._exec(conn, f"DROP INDEX IF EXISTS bookings.{idx_name}")
                self._exec(conn, create_sql)
                created.append(idx_name)
            except Exception as e:
                logger.warning("OverIndexing: failed to create %s: %s", idx_name, e)

        # Reset index usage stats so all junk indexes show idx_scan=0
        self._exec(conn, "SELECT pg_stat_reset()")

        logger.info("OverIndexing: created %d junk indexes: %s", len(created), created)
        return {
            "target_table": "ticket_flights",
            "junk_indexes": created,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Check that at least 70% of junk indexes dropped AND PK preserved.
        Matches grader logic which checks both proportional drops and PK.
        """
        junk = meta.get("junk_indexes", [])
        if not junk:
            return True
        remaining = 0
        for idx_name in junk:
            rows = self._exec(conn, f"""
                SELECT 1 FROM pg_indexes
                WHERE schemaname = 'bookings' AND indexname = '{idx_name}'
            """, fetch=True)
            if rows:
                remaining += 1
        if remaining > len(junk) * 0.3:
            return False

        # PK must be preserved (matches grader's res_pk_preserved check)
        rows = self._exec(conn, """
            SELECT 1 FROM pg_indexes
            WHERE schemaname = 'bookings'
              AND tablename = 'ticket_flights'
              AND indexname = 'ticket_flights_pkey'
        """, fetch=True)
        return bool(rows)

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Drop all junk indexes."""
        for idx_name in meta.get("junk_indexes", []):
            try:
                self._exec(conn, f"DROP INDEX IF EXISTS bookings.{idx_name}")
            except Exception as e:
                logger.warning("OverIndexing cleanup: %s: %s", idx_name, e)


# ═══════════════════════════════════════════════════════════════════
# 7. Compound: Stale Stats + Missing Index
# ═══════════════════════════════════════════════════════════════════

class CompoundStatsIndexInjector(BaseFaultInjector):
    """Combines two independent faults that interact: missing index AND stale
    statistics. Fixing only one leaves residual degradation — the planner
    still chooses bad plans. Tests multi-root-cause analysis, a capability
    gap in current frontier models that tend to stop after the first fix.
    """

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        return {
            "inject": [
                # Missing index part — drop ALL non-PK indexes on (flight_id)
                "DROP INDEX IF EXISTS bookings.idx_ticket_flights_flight",
                """DO $$ DECLARE r RECORD; BEGIN
                    FOR r IN SELECT indexname FROM pg_indexes
                        WHERE schemaname = 'bookings' AND tablename = 'ticket_flights'
                        AND indexdef LIKE '%(flight_id)%'
                        AND indexname != 'ticket_flights_pkey'
                    LOOP EXECUTE 'DROP INDEX IF EXISTS bookings.' || r.indexname; END LOOP;
                END $$""",
                # Stale stats part
                "UPDATE bookings.flights SET status = 'Delayed' WHERE flight_id IN (SELECT flight_id FROM bookings.flights WHERE status = 'Arrived' LIMIT 100000)",
                "DELETE FROM pg_statistic WHERE starelid = 'bookings.flights'::regclass",
                "SELECT pg_stat_reset_single_table_counters('bookings.flights'::regclass)",
            ],
            "cleanup": [
                # Restore index
                "CREATE INDEX IF NOT EXISTS idx_ticket_flights_flight ON bookings.ticket_flights(flight_id)",
                # Restore stats
                "UPDATE bookings.flights SET status = 'Arrived' WHERE status = 'Delayed'",
                "ANALYZE bookings.flights",
            ],
            "meta": {
                "index_meta": {
                    "index_name": "idx_ticket_flights_flight",
                    "target_table": "ticket_flights",
                    "target_column": "flight_id",
                },
                "stats_meta": {
                    "target_table": "flights",
                    "status_from": "Arrived",
                    "status_to": "Delayed",
                    "update_count": 100000,
                },
            },
        }

    def __init__(self):
        self._index_injector = MissingIndexInjector()
        self._stats_injector = StaleStatsInjector()

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        # Inject missing index
        index_params = {
            "index_name": params["index_name"],
            "target_table": params["target_table_index"],
            "target_column": params["target_column"],
        }
        index_meta = self._index_injector.inject(conn, index_params, bg_manager)

        # Inject stale stats
        stats_params = {
            "target_table": params["target_table_stats"],
            "update_status_from": params["update_status_from"],
            "update_status_to": params["update_status_to"],
            "update_count": params["update_count"],
        }
        stats_meta = self._stats_injector.inject(conn, stats_params, bg_manager)

        logger.info("CompoundStatsIndex: both faults injected")
        return {
            "index_meta": index_meta,
            "stats_meta": stats_meta,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Both sub-faults must be resolved independently. Fixing only the
        index still leaves stale stats (bad plans), and vice versa. This
        AND-logic prevents partial-fix reward hacking."""
        idx_ok = self._index_injector.check_resolved(conn, meta["index_meta"])
        stats_ok = self._stats_injector.check_resolved(conn, meta["stats_meta"])
        return idx_ok and stats_ok

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        self._index_injector.cleanup(conn, meta["index_meta"], bg_manager)
        self._stats_injector.cleanup(conn, meta["stats_meta"], bg_manager)


# ═══════════════════════════════════════════════════════════════════
# 8. Compound: Lock + Bloat
# ═══════════════════════════════════════════════════════════════════

class CompoundLockBloatInjector(BaseFaultInjector):
    """A single long transaction causes BOTH lock contention AND table bloat.

    One background connection holds a row lock (blocking others) AND also
    holds a transaction open that prevents vacuum. Mass UPDATE creates dead
    tuples. This compound fault requires the agent to resolve both symptoms
    from a single root cause — the pattern most often seen in production
    where one bad actor creates cascading degradation.
    """

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        """Hybrid: pre-bake the mass UPDATE, but threads (lock+waiters) stay live."""
        return {
            "inject": [
                "UPDATE bookings.bookings SET total_amount = total_amount + 0.01 WHERE book_ref IN (SELECT book_ref FROM bookings.bookings LIMIT 10000)",
                "SELECT pg_stat_force_next_flush()",
            ],
            "cleanup": [
                "SELECT 1",
            ],
            "needs_threads": True,
            "meta": {
                "target_table": "bookings",
                "update_count": 10000,
            },
        }

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        table = params["target_table"]
        book_refs = params["book_refs"]
        num_waiters = params.get("num_waiters", 3)
        dead_base = params.get("dead_tuple_count_base", 200000)
        dead_range = params.get("dead_tuple_count_range", 50000)
        update_count = dead_base + random.randint(0, dead_range)
        blocker_ref = book_refs[0]

        conn_params = get_connection_params()

        # Single blocker: holds row lock AND keeps tx open (blocking vacuum)
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
                logger.debug("Compound blocker thread ended: %s", e)

        t = threading.Thread(target=hold_lock_and_tx, daemon=True)
        t.start()
        bg_manager.add_thread(t)
        time.sleep(1.0)

        # Mass update to create dead tuples
        self._exec(conn, f"""
            UPDATE bookings.{table} SET total_amount = total_amount + 0.01
            WHERE book_ref IN (
                SELECT book_ref FROM bookings.{table} LIMIT {update_count}
            )
        """)

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
                        logger.debug("Compound waiter ended: %s", e)

                wt = threading.Thread(target=wait_on_lock, daemon=True)
                wt.start()
                bg_manager.add_thread(wt)
            except Exception as e:
                logger.warning("Compound: failed to create waiter %d: %s", i, e)

        time.sleep(0.5)

        try:
            self._exec(conn, "SELECT pg_stat_force_next_flush()")
        except Exception:
            pass

        logger.info("CompoundLockBloat: blocker PID %s, %d dead tuples, %d waiters",
                     blocker_pid[0], update_count, num_waiters)
        return {
            "target_table": table,
            "blocker_pid": blocker_pid[0],
            "blocker_ref": blocker_ref,
            "update_count": update_count,
            "num_waiters": num_waiters,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Both lock waits gone AND dead tuples reduced. Thresholds match
        grader (0.3 for dead tuples, system-wide lock check).
        """
        # Check no lock waits
        rows = self._exec(conn, """
            SELECT count(*) FROM pg_stat_activity
            WHERE wait_event_type = 'Lock'
              AND datname = current_database()
        """, fetch=True)
        lock_waits = rows[0][0] if rows else 0
        if lock_waits > 0:
            return False

        # Check dead tuples reduced (threshold matches grader's 0.3)
        table = meta["target_table"]
        rows = self._exec(conn, f"""
            SELECT n_dead_tup FROM pg_stat_user_tables
            WHERE schemaname = 'bookings' AND relname = '{table}'
        """, fetch=True)
        dead = rows[0][0] if rows else 0
        return dead < meta.get("update_count", 200000) * 0.3

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        bg_manager.cleanup()
        table = meta["target_table"]
        try:
            self._exec(conn, f"VACUUM bookings.{table}")
        except Exception as e:
            logger.warning("CompoundLockBloat cleanup: %s", e)


# ═══════════════════════════════════════════════════════════════════
# 9. Bad Configuration (work_mem / effective_cache_size)
# ═══════════════════════════════════════════════════════════════════

class BadConfigInjector(BaseFaultInjector):
    """Sets work_mem and effective_cache_size to pathologically low values.

    Models misconfigured GUC parameters after a config management deploy or
    a restore from a dev snapshot. Agent must identify the bad settings via
    pg_settings, apply correct values with ALTER SYSTEM, and reload config.
    """

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        return {
            "inject": [
                "ALTER SYSTEM SET work_mem = '64kB'",
                "ALTER SYSTEM SET effective_cache_size = '1MB'",
                "SELECT pg_reload_conf()",
            ],
            "cleanup": [
                "ALTER SYSTEM RESET work_mem",
                "ALTER SYSTEM RESET effective_cache_size",
                "SELECT pg_reload_conf()",
            ],
            "meta": {
                "bad_settings": {"work_mem": "64kB", "effective_cache_size": "1MB"},
                "original_settings": {"work_mem": None, "effective_cache_size": None},
            },
        }

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        bad_settings = params["bad_settings"]
        # Save original values
        originals = {}
        for param_name in bad_settings:
            rows = self._exec(conn, f"SHOW {param_name}", fetch=True)
            originals[param_name] = rows[0][0] if rows else None

        # Apply bad settings
        for param_name, bad_value in bad_settings.items():
            self._exec(conn, f"ALTER SYSTEM SET {param_name} = '{bad_value}'")
        self._exec(conn, "SELECT pg_reload_conf()")

        logger.info("BadConfig: set %s", bad_settings)
        return {
            "bad_settings": bad_settings,
            "original_settings": originals,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Check work_mem >= 1MB and effective_cache_size >= 512MB.
        Matches grader logic: pg_file_settings first, pg_settings fallback
        with unit conversion (effective_cache_size is in 8kB pages).
        """
        for param_name, min_kb in [("work_mem", 1024), ("effective_cache_size", 512 * 1024)]:
            rows = self._exec(conn, f"""
                SELECT setting FROM pg_file_settings
                WHERE name = '{param_name}' AND error IS NULL
                ORDER BY seqno DESC LIMIT 1
            """, fetch=True)
            if rows and rows[0][0]:
                val_kb = self._parse_mem_to_kb(rows[0][0])
                if val_kb < min_kb:
                    return False
            else:
                # Fallback: pg_settings (matches grader unit conversion)
                rows = self._exec(conn, f"""
                    SELECT setting FROM pg_settings WHERE name = '{param_name}'
                """, fetch=True)
                if rows:
                    setting_val = int(rows[0][0])
                    # effective_cache_size is in 8kB pages, work_mem in kB
                    if param_name == "effective_cache_size":
                        setting_val = setting_val * 8  # convert 8kB pages to kB
                    if setting_val < min_kb:
                        return False
        return True

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Reset to original or sensible defaults."""
        originals = meta.get("original_settings", {})
        for param_name, orig_value in originals.items():
            try:
                if orig_value:
                    self._exec(conn, f"ALTER SYSTEM SET {param_name} = '{orig_value}'")
                else:
                    self._exec(conn, f"ALTER SYSTEM RESET {param_name}")
            except Exception as e:
                logger.warning("BadConfig cleanup %s: %s", param_name, e)
        try:
            self._exec(conn, "SELECT pg_reload_conf()")
        except Exception:
            pass

    @staticmethod
    def _parse_mem_to_kb(value: str) -> int:
        """Parse a PostgreSQL memory value to kilobytes."""
        value = value.strip().upper()
        try:
            if value.endswith("KB"):
                return int(value[:-2])
            elif value.endswith("MB"):
                return int(value[:-2]) * 1024
            elif value.endswith("GB"):
                return int(value[:-2]) * 1024 * 1024
            elif value.endswith("TB"):
                return int(value[:-2]) * 1024 * 1024 * 1024
            else:
                # Assume kB
                return int(value)
        except ValueError:
            return 0


# ═══════════════════════════════════════════════════════════════════
# 10. Index Bloat / Fragmented Index
# ═══════════════════════════════════════════════════════════════════

class IndexBloatInjector(BaseFaultInjector):
    """Mass-update rows to create index bloat via B-tree page splits.

    Models gradual index degradation from high-churn UPDATE workloads.
    Agent must detect bloated index size and perform REINDEX. Resolution
    verified by checking pg_relation_size decrease, not command matching.
    """

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        # Reduced rounds/batch for faster prebake (~10s instead of 3min)
        inject_sql = [
            "CREATE INDEX IF NOT EXISTS idx_ticket_flights_flight ON bookings.ticket_flights(flight_id)",
        ]
        for i in range(2):
            inject_sql.append(
                "UPDATE bookings.ticket_flights SET amount = amount + 0.01 "
                "WHERE ctid IN (SELECT ctid FROM bookings.ticket_flights LIMIT 50000)"
            )
        return {
            "inject": inject_sql,
            "cleanup": [
                "REINDEX INDEX bookings.idx_ticket_flights_flight",
                "VACUUM bookings.ticket_flights",
            ],
            "meta": {
                "target_table": "ticket_flights",
                "target_index": "idx_ticket_flights_flight",
                "target_column": "flight_id",
                "initial_size": 0,   # Will be filled at inject time
                "bloated_size": 0,   # Will be filled at inject time
            },
        }

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        table = params["target_table"]
        index_name = params["target_index"]
        column = params["target_column"]
        rounds = params.get("update_rounds", 3)
        batch_size = params.get("update_batch_size", 100000)

        # Ensure the index exists
        try:
            self._exec(conn, f"CREATE INDEX IF NOT EXISTS {index_name} ON bookings.{table}({column})")
        except Exception:
            pass

        # Record initial index size
        rows = self._exec(conn, f"""
            SELECT pg_relation_size('bookings.{index_name}') AS idx_size
        """, fetch=True)
        initial_size = rows[0][0] if rows else 0

        # Mass update in rounds to create index churn
        for i in range(rounds):
            self._exec(conn, f"""
                UPDATE bookings.{table} SET amount = amount + 0.01
                WHERE ctid IN (
                    SELECT ctid FROM bookings.{table} LIMIT {batch_size}
                )
            """)
            logger.info("IndexBloat: round %d/%d done (%d rows)", i + 1, rounds, batch_size)

        # Record bloated index size
        rows = self._exec(conn, f"""
            SELECT pg_relation_size('bookings.{index_name}') AS idx_size
        """, fetch=True)
        bloated_size = rows[0][0] if rows else 0

        logger.info("IndexBloat: index %s grew %d → %d bytes", index_name, initial_size, bloated_size)
        return {
            "target_table": table,
            "target_index": index_name,
            "target_column": column,
            "initial_size": initial_size,
            "bloated_size": bloated_size,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Check that index exists and size decreased by at least 10%.
        Matches grader's res_size_reduced threshold (bloated_size * 0.9).
        """
        index_name = meta["target_index"]
        bloated_size = meta.get("bloated_size", 0)
        if bloated_size == 0:
            return True

        # Index must still exist
        rows = self._exec(conn, f"""
            SELECT 1 FROM pg_indexes
            WHERE schemaname = 'bookings' AND indexname = '{index_name}'
        """, fetch=True)
        if not rows:
            return False

        rows = self._exec(conn, f"""
            SELECT pg_relation_size('bookings.{index_name}') AS idx_size
        """, fetch=True)
        current_size = rows[0][0] if rows else bloated_size

        # Matches grader's threshold: size must decrease by at least 10%
        return current_size < bloated_size * 0.9

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Reindex to clean up."""
        index_name = meta["target_index"]
        try:
            self._exec(conn, f"REINDEX INDEX bookings.{index_name}")
        except Exception as e:
            logger.warning("IndexBloat cleanup: %s", e)
        # Vacuum to clean dead tuples from the updates
        table = meta["target_table"]
        try:
            self._exec(conn, f"VACUUM bookings.{table}")
        except Exception as e:
            logger.warning("IndexBloat cleanup vacuum: %s", e)


# ═══════════════════════════════════════════════════════════════════
# 11. Wrong Index Column Order
# ═══════════════════════════════════════════════════════════════════

class WrongIndexOrderInjector(BaseFaultInjector):
    """Drop standalone index on flight_id, forcing queries to use composite PK
    (ticket_no, flight_id) which can't efficiently filter on flight_id alone.

    Models a subtle indexing mistake: the composite PK exists but its column
    order makes leading-column queries on flight_id inefficient. Agent must
    understand B-tree index ordering to diagnose the plan regression.
    """

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        return {
            "inject": [
                "DROP INDEX IF EXISTS bookings.idx_ticket_flights_flight",
                # Drop ALL non-PK standalone indexes on (flight_id)
                """DO $$ DECLARE r RECORD; BEGIN
                    FOR r IN SELECT indexname FROM pg_indexes
                        WHERE schemaname = 'bookings' AND tablename = 'ticket_flights'
                        AND indexdef LIKE '%(flight_id)%'
                        AND indexname != 'ticket_flights_pkey'
                    LOOP EXECUTE 'DROP INDEX IF EXISTS bookings.' || r.indexname; END LOOP;
                END $$""",
            ],
            "cleanup": [
                "CREATE INDEX IF NOT EXISTS idx_ticket_flights_flight ON bookings.ticket_flights(flight_id)",
            ],
            "meta": {
                "target_table": "ticket_flights",
                "target_column": "flight_id",
                "dropped_indexes": ["idx_ticket_flights_flight"],
            },
        }

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        table = params["target_table"]
        column = params["target_column"]
        index_to_drop = params["index_to_drop"]

        # Drop ALL standalone indexes that start with flight_id
        # (there may be multiple from previous test runs or other tasks)
        rows = self._exec(conn, f"""
            SELECT indexname FROM pg_indexes
            WHERE schemaname = 'bookings'
              AND tablename = '{table}'
              AND indexdef LIKE '%({column})%'
              AND indexname != '{table}_pkey'
        """, fetch=True)
        dropped = []
        for row in (rows or []):
            idx = row[0]
            try:
                self._exec(conn, f"DROP INDEX IF EXISTS bookings.{idx}")
                dropped.append(idx)
            except Exception as e:
                logger.warning("WrongIndexOrder: failed to drop %s: %s", idx, e)

        if not dropped:
            # Nothing to drop — the fault condition already exists
            self._exec(conn, f"DROP INDEX IF EXISTS bookings.{index_to_drop}")
            dropped.append(index_to_drop)

        logger.info("WrongIndexOrder: dropped %s — queries on %s must use composite PK",
                     dropped, column)
        return {
            "target_table": table,
            "target_column": column,
            "dropped_indexes": dropped,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Check that a standalone index on flight_id exists."""
        column = meta["target_column"]
        table = meta["target_table"]
        rows = self._exec(conn, f"""
            SELECT 1 FROM pg_indexes
            WHERE schemaname = 'bookings'
              AND tablename = '{table}'
              AND indexdef LIKE '%({column})%'
              AND indexname != 'ticket_flights_pkey'
        """, fetch=True)
        return bool(rows)

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Re-create the standalone index."""
        table = meta["target_table"]
        column = meta["target_column"]
        # Restore at least one standalone index
        dropped = meta.get("dropped_indexes", [meta.get("dropped_index", "idx_ticket_flights_flight")])
        if dropped:
            idx_name = dropped[0]
            try:
                self._exec(conn, f"CREATE INDEX IF NOT EXISTS {idx_name} ON bookings.{table}({column})")
            except Exception as e:
                logger.warning("WrongIndexOrder cleanup: %s", e)


# ═══════════════════════════════════════════════════════════════════
# 12. Deadlock Chain
# ═══════════════════════════════════════════════════════════════════

class DeadlockChainInjector(BaseFaultInjector):
    """Creates a real PostgreSQL deadlock between transactions updating rows
    in opposite order. Deadlock timeout is set to 300s per-session to prevent
    PostgreSQL from auto-resolving. Agent must identify the deadlock from
    pg_locks and pg_stat_activity, then terminate the appropriate backend.
    """

    # Thread-only fault — not pre-bakeable
    # get_prebake_sql() returns None (inherited from base)

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        table = params["target_table"]
        ref_a = params["book_ref_a"]
        ref_b = params["book_ref_b"]

        conn_params = get_connection_params()
        deadlock_detected = [False]
        pids = {"thread1": None, "thread2": None}
        deadlock_error = [None]

        def thread1_fn():
            try:
                c = psycopg2.connect(**conn_params)
                c.autocommit = False
                bg_manager.add_connection(c)
                cur = c.cursor()
                cur.execute("SELECT pg_backend_pid()")
                pids["thread1"] = cur.fetchone()[0]
                cur.execute("BEGIN")
                cur.execute("SET LOCAL deadlock_timeout = '300s'")
                cur.execute(f"UPDATE bookings.{table} SET total_amount = total_amount WHERE book_ref = '{ref_a}'")
                time.sleep(1.5)  # Wait for thread2 to lock ref_b
                cur.execute(f"UPDATE bookings.{table} SET total_amount = total_amount WHERE book_ref = '{ref_b}'")
                c.commit()
            except psycopg2.errors.DeadlockDetected as e:
                deadlock_detected[0] = True
                deadlock_error[0] = str(e)
                logger.info("DeadlockChain: thread1 was the deadlock victim")
                try:
                    c.rollback()
                except Exception:
                    pass
            except Exception as e:
                logger.debug("DeadlockChain thread1 error: %s", e)

        def thread2_fn():
            try:
                c = psycopg2.connect(**conn_params)
                c.autocommit = False
                bg_manager.add_connection(c)
                cur = c.cursor()
                cur.execute("SELECT pg_backend_pid()")
                pids["thread2"] = cur.fetchone()[0]
                cur.execute("BEGIN")
                cur.execute("SET LOCAL deadlock_timeout = '300s'")
                cur.execute(f"UPDATE bookings.{table} SET total_amount = total_amount WHERE book_ref = '{ref_b}'")
                time.sleep(1.5)  # Wait for thread1 to lock ref_a
                cur.execute(f"UPDATE bookings.{table} SET total_amount = total_amount WHERE book_ref = '{ref_a}'")
                c.commit()
            except psycopg2.errors.DeadlockDetected as e:
                deadlock_detected[0] = True
                deadlock_error[0] = str(e)
                logger.info("DeadlockChain: thread2 was the deadlock victim")
                try:
                    c.rollback()
                except Exception:
                    pass
            except Exception as e:
                logger.debug("DeadlockChain thread2 error: %s", e)

        t1 = threading.Thread(target=thread1_fn, daemon=True)
        t2 = threading.Thread(target=thread2_fn, daemon=True)
        t1.start()
        time.sleep(0.3)  # Slight stagger so thread1 locks ref_a first
        t2.start()
        bg_manager.add_thread(t1)
        bg_manager.add_thread(t2)

        # Wait for deadlock to establish (both threads grab first lock, block on second)
        time.sleep(3.0)

        logger.info("DeadlockChain: deadlock established (timeout=300s), pids=%s", pids)
        return {
            "target_table": table,
            "book_ref_a": ref_a,
            "book_ref_b": ref_b,
            "deadlock_detected": deadlock_detected[0],
            "deadlock_error": deadlock_error[0],
            "pids": pids,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Check live DB state: no ungranted transactionid locks and no lock
        waiters. Matches grader logic instead of relying on static metadata.
        """
        rows = self._exec(conn, """
            SELECT count(*) FROM pg_locks
            WHERE NOT granted AND locktype = 'transactionid'
        """, fetch=True)
        blocked = rows[0][0] if rows else 999
        if blocked > 0:
            return False

        rows = self._exec(conn, """
            SELECT count(*) FROM pg_stat_activity
            WHERE wait_event_type = 'Lock' AND datname = current_database()
        """, fetch=True)
        lock_waits = rows[0][0] if rows else 999
        return lock_waits == 0

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Terminate deadlocked backends and clean up connections."""
        pids = meta.get("pids", {})
        for label in ("thread1", "thread2"):
            pid = pids.get(label)
            if pid:
                try:
                    self._exec(conn, f"SELECT pg_terminate_backend({pid})")
                except Exception as e:
                    logger.debug("DeadlockChain cleanup terminate %s (pid=%s): %s", label, pid, e)
        time.sleep(0.5)
        bg_manager.cleanup()


# ═══════════════════════════════════════════════════════════════════
# 13. Query Plan Flip (random_page_cost)
# ═══════════════════════════════════════════════════════════════════

class QueryPlanFlipInjector(BaseFaultInjector):
    """Sets random_page_cost to extreme value (100) to force planner to prefer
    Seq Scans over Index Scans even when indexes exist. Models misconfigured
    planner cost parameters — common after migrating from HDD to SSD storage
    without updating cost settings.
    """

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        return {
            "inject": [
                "CREATE INDEX IF NOT EXISTS idx_ticket_flights_flight ON bookings.ticket_flights(flight_id)",
                "ALTER DATABASE demo SET random_page_cost = 100",
            ],
            "cleanup": [
                "ALTER DATABASE demo RESET random_page_cost",
            ],
            "meta": {
                "bad_param": "random_page_cost",
                "bad_value": "100",
                "original_value": "4",
            },
        }

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        bad_param = params["bad_param"]
        bad_value = params["bad_value"]

        # Ensure the index exists first (so there IS an index to ignore)
        try:
            self._exec(conn, "CREATE INDEX IF NOT EXISTS idx_ticket_flights_flight ON bookings.ticket_flights(flight_id)")
        except Exception:
            pass

        # Save original value
        rows = self._exec(conn, f"SHOW {bad_param}", fetch=True)
        original_value = rows[0][0] if rows else None

        # Set extreme value at database level
        self._exec(conn, f"ALTER DATABASE demo SET {bad_param} = {bad_value}")

        logger.info("QueryPlanFlip: set %s = %s (was %s)", bad_param, bad_value, original_value)
        return {
            "bad_param": bad_param,
            "bad_value": bad_value,
            "original_value": original_value,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Check that random_page_cost is back to a reasonable value (<= 4).
        Matches grader: checks database-level setting, pg_file_settings, and
        fresh SHOW value — all must be <= 4.0.
        """
        param = meta["bad_param"]
        # Check database-level setting (ALTER DATABASE demo SET ...)
        rows = self._exec(conn, f"""
            SELECT setconfig FROM pg_db_role_setting
            WHERE setdatabase = (SELECT oid FROM pg_database WHERE datname = 'demo')
              AND setrole = 0
        """, fetch=True)
        if rows:
            for row in rows:
                configs = row[0] if row[0] else []
                for cfg in configs:
                    if cfg.startswith(f"{param}="):
                        val = float(cfg.split("=")[1])
                        if val > 4.0:
                            return False

        # Check pg_file_settings (ALTER SYSTEM)
        rows = self._exec(conn, f"""
            SELECT setting FROM pg_file_settings
            WHERE name = '{param}' AND error IS NULL
            ORDER BY seqno DESC LIMIT 1
        """, fetch=True)
        if rows and rows[0][0]:
            try:
                if float(rows[0][0]) > 4.0:
                    return False
            except (ValueError, TypeError):
                pass

        # Check current session value
        rows = self._exec(conn, f"SHOW {param}", fetch=True)
        if rows:
            try:
                val = float(rows[0][0])
                if val > 4.0:
                    return False
            except ValueError:
                pass

        return True

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Reset the parameter."""
        param = meta["bad_param"]
        try:
            self._exec(conn, f"ALTER DATABASE demo RESET {param}")
        except Exception as e:
            logger.warning("QueryPlanFlip cleanup: %s", e)


# ═══════════════════════════════════════════════════════════════════
# 14. Cascading Bloat (Multi-Table)
# ═══════════════════════════════════════════════════════════════════

class CascadingBloatInjector(BaseFaultInjector):
    """Open REPEATABLE READ transaction + UPDATE multiple tables to bloat them all.

    Models cascading bloat from a long-running analytics query holding a
    snapshot while OLTP writes continue. Agent must identify the snapshot-
    holding backend, terminate it, then VACUUM all affected tables. This is
    a hard-tier task requiring multi-step, multi-table remediation.
    """

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        """Hybrid: pre-bake the mass UPDATEs, but snapshot-holding thread stays live."""
        return {
            "inject": [
                "UPDATE bookings.bookings SET total_amount = total_amount + 0.01 WHERE book_ref IN (SELECT book_ref FROM bookings.bookings LIMIT 50000)",
                "UPDATE bookings.flights SET status = status WHERE flight_id IN (SELECT flight_id FROM bookings.flights LIMIT 50000)",
                "UPDATE bookings.ticket_flights SET amount = amount + 0.01 WHERE ctid IN (SELECT ctid FROM bookings.ticket_flights LIMIT 50000)",
                "UPDATE bookings.tickets SET passenger_name = passenger_name WHERE ticket_no IN (SELECT ticket_no FROM bookings.tickets LIMIT 50000)",
                "SELECT pg_stat_force_next_flush()",
            ],
            "cleanup": [
                "VACUUM bookings.bookings",
                "VACUUM bookings.flights",
                "VACUUM bookings.ticket_flights",
                "VACUUM bookings.tickets",
            ],
            "needs_threads": True,
            "meta": {
                "tables": ["bookings", "flights", "ticket_flights", "tickets"],
                "update_count_per_table": 50000,
            },
        }

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        tables = params["tables"]
        update_count = params.get("update_count_per_table", 50000)

        conn_params = get_connection_params()

        # Start long-running REPEATABLE READ transaction to hold snapshot
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
                # Do a read to establish the snapshot
                cur.execute("SELECT count(*) FROM bookings.bookings")
                # Hold transaction open
                while not bg_manager.stop_event.wait(timeout=1.0):
                    pass
            except Exception as e:
                logger.debug("CascadingBloat snapshot thread ended: %s", e)

        t = threading.Thread(target=hold_snapshot, daemon=True)
        t.start()
        bg_manager.add_thread(t)
        time.sleep(1.0)

        # Update each table to create dead tuples (committed on admin conn)
        update_sqls = {
            "bookings": f"""
                UPDATE bookings.bookings SET total_amount = total_amount + 0.01
                WHERE book_ref IN (SELECT book_ref FROM bookings.bookings LIMIT {update_count})
            """,
            "flights": f"""
                UPDATE bookings.flights SET status = status
                WHERE flight_id IN (SELECT flight_id FROM bookings.flights LIMIT {update_count})
            """,
            "ticket_flights": f"""
                UPDATE bookings.ticket_flights SET amount = amount + 0.01
                WHERE ctid IN (SELECT ctid FROM bookings.ticket_flights LIMIT {update_count})
            """,
            "tickets": f"""
                UPDATE bookings.tickets SET passenger_name = passenger_name
                WHERE ticket_no IN (SELECT ticket_no FROM bookings.tickets LIMIT {update_count})
            """,
        }

        updated_tables = []
        for tbl in tables:
            if tbl in update_sqls:
                try:
                    self._exec(conn, update_sqls[tbl])
                    updated_tables.append(tbl)
                    logger.info("CascadingBloat: updated %d rows in %s", update_count, tbl)
                except Exception as e:
                    logger.warning("CascadingBloat: failed to update %s: %s", tbl, e)

        try:
            self._exec(conn, "SELECT pg_stat_force_next_flush()")
        except Exception:
            pass
        time.sleep(0.5)

        logger.info("CascadingBloat: blocker PID %s, updated tables: %s",
                     blocker_pid[0], updated_tables)
        return {
            "tables": updated_tables,
            "update_count_per_table": update_count,
            "blocker_pid": blocker_pid[0],
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Check no old backend_xmin transactions and dead tuples reduced
        across at least half the tables. Matches grader logic.
        """
        # Check no long-running txns with old backend_xmin (matches grader)
        rows = self._exec(conn, """
            SELECT count(*) FROM pg_stat_activity
            WHERE backend_xmin IS NOT NULL
              AND age(backend_xmin) > 1000
              AND datname = current_database()
              AND pid != pg_backend_pid()
        """, fetch=True)
        old_xmin = rows[0][0] if rows else 999
        if old_xmin > 0:
            return False

        # Check dead tuples reduced on at least half the tables
        # (threshold 0.5 matches grader's per-table threshold)
        tables = meta.get("tables", [])
        update_count = meta.get("update_count_per_table", 50000)
        cleaned = 0
        for tbl in tables:
            rows = self._exec(conn, f"""
                SELECT n_dead_tup FROM pg_stat_user_tables
                WHERE schemaname = 'bookings' AND relname = '{tbl}'
            """, fetch=True)
            dead = rows[0][0] if rows else 999999
            if dead < update_count * 0.5:
                cleaned += 1

        return cleaned >= len(tables) * 0.5

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Kill blocker and vacuum all tables."""
        bg_manager.cleanup()
        for tbl in meta.get("tables", []):
            try:
                self._exec(conn, f"VACUUM bookings.{tbl}")
            except Exception as e:
                logger.warning("CascadingBloat cleanup vacuum %s: %s", tbl, e)


# ═══════════════════════════════════════════════════════════════════
# 15. Permission / Role Error
# ═══════════════════════════════════════════════════════════════════

class PermissionErrorInjector(BaseFaultInjector):
    """Creates an app_user role and revokes SELECT on a critical table.

    Models permission regression after a role migration or GRANT cleanup.
    Agent must inspect information_schema.role_table_grants to find the
    missing privilege and re-grant it. Tests RBAC diagnostic reasoning.
    """

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        return {
            "inject": [
                "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_user') THEN CREATE ROLE app_user LOGIN PASSWORD 'apppass'; END IF; END $$",
                "GRANT CONNECT ON DATABASE demo TO app_user",
                "GRANT USAGE ON SCHEMA bookings TO app_user",
                "GRANT SELECT ON ALL TABLES IN SCHEMA bookings TO app_user",
                "REVOKE SELECT ON bookings.ticket_flights FROM app_user",
            ],
            "cleanup": [
                "GRANT SELECT ON bookings.ticket_flights TO app_user",
            ],
            "meta": {
                "role_name": "app_user",
                "target_table": "ticket_flights",
                "target_schema": "bookings",
                "revoked_privilege": "SELECT",
            },
        }

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        role_name = params["role_name"]
        role_password = params["role_password"]
        target_table = params["target_table"]
        target_schema = params["target_schema"]

        # Create the role if it doesn't exist
        try:
            self._exec(conn, f"CREATE ROLE {role_name} LOGIN PASSWORD '{role_password}'")
        except Exception:
            # Role may already exist
            pass

        # Grant baseline permissions
        try:
            self._exec(conn, f"GRANT CONNECT ON DATABASE demo TO {role_name}")
            self._exec(conn, f"GRANT USAGE ON SCHEMA {target_schema} TO {role_name}")
            self._exec(conn, f"GRANT SELECT ON ALL TABLES IN SCHEMA {target_schema} TO {role_name}")
        except Exception as e:
            logger.debug("PermissionError: grant baseline: %s", e)

        # Now revoke the specific permission to create the fault
        self._exec(conn, f"REVOKE SELECT ON {target_schema}.{target_table} FROM {role_name}")

        logger.info("PermissionError: revoked SELECT on %s.%s from %s",
                     target_schema, target_table, role_name)
        return {
            "role_name": role_name,
            "target_table": target_table,
            "target_schema": target_schema,
            "revoked_privilege": "SELECT",
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Check that the role has SELECT on the target table."""
        role = meta["role_name"]
        table = meta["target_table"]
        schema = meta["target_schema"]
        rows = self._exec(conn, f"""
            SELECT 1
            FROM information_schema.role_table_grants
            WHERE grantee = '{role}'
              AND table_schema = '{schema}'
              AND table_name = '{table}'
              AND privilege_type = 'SELECT'
        """, fetch=True)
        return bool(rows)

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Re-grant the permission."""
        role = meta["role_name"]
        table = meta["target_table"]
        schema = meta["target_schema"]
        try:
            self._exec(conn, f"GRANT SELECT ON {schema}.{table} TO {role}")
        except Exception as e:
            logger.warning("PermissionError cleanup: %s", e)


# ═══════════════════════════════════════════════════════════════════
# 16. Sequence Exhaustion / PK Conflict
# ═══════════════════════════════════════════════════════════════════

class SequenceExhaustionInjector(BaseFaultInjector):
    """Resets a sequence to 1 so INSERTs fail with duplicate key violations.

    Models sequence misconfiguration after a table restore or data import
    that did not update the sequence. Agent must query max(pk) and call
    setval() to re-synchronize the sequence with existing data.
    """

    @classmethod
    def get_prebake_sql(cls) -> Optional[dict]:
        return {
            "inject": [
                "SELECT setval('bookings.flights_flight_id_seq', 1, false)",
            ],
            "cleanup": [
                "SELECT setval('bookings.flights_flight_id_seq', (SELECT max(flight_id) FROM bookings.flights))",
            ],
            "meta": {
                "sequence_name": "bookings.flights_flight_id_seq",
                "target_table": "flights",
                "pk_column": "flight_id",
                "original_value": None,  # Will be set dynamically
            },
        }

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        sequence_name = params["sequence_name"]
        table = params["target_table"]
        pk_column = params["pk_column"]

        # Save original sequence value
        rows = self._exec(conn, f"SELECT last_value FROM {sequence_name}", fetch=True)
        original_value = rows[0][0] if rows else None

        # Reset sequence to 1
        self._exec(conn, f"SELECT setval('{sequence_name}', 1, false)")

        logger.info("SequenceExhaustion: reset %s to 1 (was %s)", sequence_name, original_value)
        return {
            "sequence_name": sequence_name,
            "target_table": table,
            "pk_column": pk_column,
            "original_value": original_value,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Check that sequence value >= max(pk_column)."""
        seq = meta["sequence_name"]
        table = meta["target_table"]
        pk = meta["pk_column"]

        rows = self._exec(conn, f"SELECT last_value FROM {seq}", fetch=True)
        seq_val = rows[0][0] if rows else 0

        rows = self._exec(conn, f"SELECT max({pk}) FROM bookings.{table}", fetch=True)
        max_pk = rows[0][0] if rows else 0

        return seq_val >= max_pk

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Reset sequence to correct value."""
        seq = meta["sequence_name"]
        table = meta["target_table"]
        pk = meta["pk_column"]
        try:
            self._exec(conn, f"SELECT setval('{seq}', (SELECT max({pk}) FROM bookings.{table}))")
        except Exception as e:
            logger.warning("SequenceExhaustion cleanup: %s", e)


# ═══════════════════════════════════════════════════════════════════
# 17. Compound: Connection Exhaustion + Deadlock
# ═══════════════════════════════════════════════════════════════════

class CompoundConnDeadlockInjector(BaseFaultInjector):
    """Combines connection exhaustion (idle-in-tx) with a concurrent deadlock.

    The hardest compound fault: agent must triage two simultaneous P1 issues
    with interacting symptoms. Connection exhaustion limits the ability to
    even diagnose the deadlock. Tests prioritization under resource pressure.
    """

    # Thread-only fault — not pre-bakeable
    # get_prebake_sql() returns None (inherited from base)

    def __init__(self):
        self._conn_injector = ConnectionExhaustionInjector()
        self._deadlock_injector = DeadlockChainInjector()

    def inject(self, conn, params: dict, bg_manager: BackgroundConnectionManager) -> dict:
        # Inject connection exhaustion (fewer connections to leave room for deadlock)
        conn_params = {
            "num_connections_base": params.get("num_connections_base", 80),
            "num_connections_range": params.get("num_connections_range", 5),
        }
        conn_meta = self._conn_injector.inject(conn, conn_params, bg_manager)

        # Inject deadlock
        deadlock_params = {
            "target_table": params.get("target_table", "bookings"),
            "book_ref_a": params.get("book_ref_a", "361A07"),
            "book_ref_b": params.get("book_ref_b", "363381"),
        }
        deadlock_meta = self._deadlock_injector.inject(conn, deadlock_params, bg_manager)

        logger.info("CompoundConnDeadlock: both faults injected")
        return {
            "conn_meta": conn_meta,
            "deadlock_meta": deadlock_meta,
        }

    def check_resolved(self, conn, meta: dict) -> bool:
        """Both idle connections cleared AND no deadlock locks remaining.
        Uses live DB state checks matching grader logic.
        """
        conn_ok = self._conn_injector.check_resolved(conn, meta.get("conn_meta", {}))
        # Check live lock state instead of static metadata (matches grader)
        deadlock_ok = self._deadlock_injector.check_resolved(conn, meta.get("deadlock_meta", {}))
        return conn_ok and deadlock_ok

    def cleanup(self, conn, meta: dict, bg_manager: BackgroundConnectionManager):
        """Clean up both faults."""
        self._conn_injector.cleanup(conn, meta.get("conn_meta", {}), bg_manager)
        self._deadlock_injector.cleanup(conn, meta.get("deadlock_meta", {}), bg_manager)


# ═══════════════════════════════════════════════════════════════════
# Registry — 17 fault types across 3 difficulty tiers
# Easy (single fault): missing_index, stale_statistics, bad_config, etc.
# Medium (multi-step): table_bloat, lock_contention, over_indexing
# Hard (compound): compound_stats_index, compound_lock_bloat, cascading_bloat
# ═══════════════════════════════════════════════════════════════════

INJECTOR_REGISTRY: Dict[str, BaseFaultInjector] = {
    "missing_index": MissingIndexInjector(),
    "stale_statistics": StaleStatsInjector(),
    "connection_exhaustion": ConnectionExhaustionInjector(),
    "lock_contention": LockContentionInjector(),
    "table_bloat": TableBloatInjector(),
    "over_indexing": OverIndexingInjector(),
    "compound_stats_index": CompoundStatsIndexInjector(),
    "compound_lock_bloat": CompoundLockBloatInjector(),
    # New deferred faults (tasks 9–17)
    "bad_config": BadConfigInjector(),
    "index_bloat": IndexBloatInjector(),
    "wrong_index_order": WrongIndexOrderInjector(),
    "deadlock_chain": DeadlockChainInjector(),
    "query_plan_flip": QueryPlanFlipInjector(),
    "cascading_bloat": CascadingBloatInjector(),
    "permission_error": PermissionErrorInjector(),
    "sequence_exhaustion": SequenceExhaustionInjector(),
    "compound_conn_deadlock": CompoundConnDeadlockInjector(),
}


def get_injector(fault_type: str) -> BaseFaultInjector:
    """Look up an injector by fault type. Raises KeyError if not found."""
    if fault_type not in INJECTOR_REGISTRY:
        raise KeyError(f"Unknown fault_type: {fault_type!r}")
    return INJECTOR_REGISTRY[fault_type]
