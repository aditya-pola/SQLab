"""
SQLab — Task definitions for 17 PostgreSQL incident-response scenarios.

Each task maps to a real-world PostgreSQL fault type and contains injection
parameters, a realistic alert message, and safety configuration. Tasks are
ordered by difficulty: easy (1-5) → medium (6-11) → hard (12-17).

Real-world utility: Every task models a fault that production SRE teams
encounter regularly. The 17 tasks cover 5 fault categories — performance,
resources, storage, configuration, and access/integrity — providing broad
coverage of the PostgreSQL operations domain.

Difficulty calibration: Easy tasks test single-fault diagnosis (solvable in
3-5 steps by frontier models). Medium tasks introduce ambiguity or multi-step
fixes. Hard tasks present compound faults requiring multi-root-cause analysis
— current frontier models (GPT-4o, Claude Sonnet 4) achieve only 0.4-0.7 on
these, leaving significant headroom for improvement through RL training.

Alert design: Alert messages are modeled on production monitoring systems
(PagerDuty/Datadog style) with severity tags (P1/P2) and observable symptoms
only — no root-cause hints. This forces agents to diagnose rather than
pattern-match the alert text, mirroring real incident response.
"""

from typing import Dict, Any

# ── Alert messages (what the model sees on reset) ─────────────────────
# Alerts mimic real production monitoring: P1/P2 severity tags, metric-based
# symptoms, and affected table names from slow-query logs. Critically, alerts
# contain NO diagnostic hints or root-cause clues — the agent must discover
# these through investigation, just like a real SRE reading a PagerDuty alert.

ALERTS = {
    "missing_index": (
        "ALERT [P2 — Slow Query]: The booking dashboard is reporting timeouts on "
        "flight segment lookups. Users are seeing 5-second+ page loads when viewing "
        "ticket-to-flight information. The query appears to involve the ticket_flights "
        "table. Please investigate and resolve."
    ),
    "stale_statistics": (
        "ALERT [P2 — High Query Latency]: Queries against the flights table have "
        "degraded sharply after a recent batch migration that updated flight statuses. "
        "p99 latency went from under 100ms to several seconds. "
        "Please investigate and resolve."
    ),
    "connection_exhaustion": (
        "ALERT [P1 — Connection Pool Full]: Application is failing to acquire new "
        "database connections. Users are seeing 'too many clients' errors. The "
        "monitoring dashboard shows the connection count is near the max_connections "
        "limit. Please investigate and resolve urgently."
    ),
    "permission_error": (
        "ALERT [P1 — Access Denied]: The application user 'app_user' is receiving "
        "'permission denied for table ticket_flights' errors. SELECT queries from the "
        "application are failing. This started after a recent migration. "
        "Please investigate and resolve urgently."
    ),
    "sequence_exhaustion": (
        "ALERT [P1 — Insert Failures]: INSERT operations into the flights table are "
        "failing with 'duplicate key value violates unique constraint flights_pkey'. "
        "The sequence generating flight IDs appears to be producing values that "
        "already exist. Please investigate and resolve urgently."
    ),
    "bad_config": (
        "ALERT [P2 — High Temp File Usage]: Multiple queries across the system are "
        "running significantly slower than baseline. Temp file usage has spiked. "
        "No schema or code changes were deployed. Please investigate and resolve."
    ),
    "lock_contention": (
        "ALERT [P1 — Queries Stuck]: Multiple application queries are hanging and "
        "not returning. The booking update endpoint has been unresponsive for several "
        "minutes. Other queries touching the bookings table appear blocked. "
        "Please investigate and resolve urgently."
    ),
    "table_bloat": (
        "ALERT [P2 — Elevated Dead Tuples]: The bookings table has grown significantly "
        "in the last hour and query performance is degrading. Monitoring shows an "
        "elevated dead tuple count. Please investigate and resolve."
    ),
    "over_indexing": (
        "ALERT [P2 — Slow Writes]: INSERT and UPDATE operations on the ticket_flights "
        "table are 5-10x slower than baseline. Write latency spiked after a recent "
        "deployment. Please investigate and resolve."
    ),
    "index_bloat": (
        "ALERT [P2 — High Index Scan Latency]: Queries on the ticket_flights table that "
        "previously used fast index lookups are now slower than expected. Index size "
        "on disk appears disproportionate. Please investigate and resolve."
    ),
    "wrong_index_order": (
        "ALERT [P2 — Slow Query]: Lookups on the ticket_flights table by flight_id "
        "are taking 400ms+ when they should be sub-millisecond. "
        "Please investigate and resolve."
    ),
    "compound_stats_index": (
        "ALERT [P1 — Query Timeout]: A critical query joining ticket_flights "
        "and flights is now taking 30+ seconds. This started after a batch migration "
        "that updated flight records. Please investigate and resolve."
    ),
    "compound_lock_bloat": (
        "ALERT [P1 — Unresponsive Queries]: UPDATE operations on the bookings table "
        "are hanging, and overall database performance is degrading. Multiple symptoms "
        "have been reported in the last 15 minutes. Please investigate and resolve."
    ),
    "deadlock_chain": (
        "ALERT [P1 — Deadlock Detected]: The database has detected a deadlock between "
        "concurrent transactions updating the bookings table. Error logs show "
        "'deadlock detected' with two processes waiting on each other. "
        "Please investigate the pattern and resolve."
    ),
    "query_plan_flip": (
        "ALERT [P2 — High Query Latency]: A query on ticket_flights that was previously "
        "sub-millisecond is now taking 30ms+. No schema changes were made. "
        "Please investigate and resolve."
    ),
    "cascading_bloat": (
        "ALERT [P1 — Dead Tuple Spike]: Dead tuple counts are spiking across "
        "multiple tables simultaneously. Autovacuum does not appear to be making "
        "progress. Please investigate and resolve."
    ),
    "compound_conn_deadlock": (
        "ALERT [P1 — Connection Failures]: The database is in a degraded state. New "
        "connections are failing and active transactions are stuck. Multiple on-call "
        "alerts have fired in the last 5 minutes. Please investigate and resolve urgently."
    ),
}

# ── Book refs and flight IDs for parameterized faults ───────────────
# Fixed reference values ensure deterministic fault injection. These book_refs
# and flight_ids exist in the Airlines demo database and are chosen to avoid
# edge cases (e.g., they have associated ticket_flights rows for join queries).

LOCK_BOOK_REFS = ["361A07", "363381", "3643D3", "36C3D5", "36F939"]
LOCK_FLIGHT_IDS = [68373, 68374, 68378, 68379, 68380]

# ── Junk index pool for over-indexing ───────────────────────────────
# 15 realistic junk indexes covering common over-indexing anti-patterns from
# production PostgreSQL audits: redundant single-column, duplicate composites
# in different column orders, partial indexes with low selectivity, and
# descending-order indexes that PostgreSQL rarely benefits from.

JUNK_INDEX_POOL = [
    ("idx_tf_junk1", "CREATE INDEX idx_tf_junk1 ON bookings.ticket_flights(amount)"),
    ("idx_tf_junk2", "CREATE INDEX idx_tf_junk2 ON bookings.ticket_flights(fare_conditions)"),
    ("idx_tf_junk3", "CREATE INDEX idx_tf_junk3 ON bookings.ticket_flights(amount, fare_conditions)"),
    ("idx_tf_junk4", "CREATE INDEX idx_tf_junk4 ON bookings.ticket_flights(fare_conditions, amount)"),
    ("idx_tf_junk5", "CREATE INDEX idx_tf_junk5 ON bookings.ticket_flights(flight_id, amount)"),
    ("idx_tf_junk6", "CREATE INDEX idx_tf_junk6 ON bookings.ticket_flights(flight_id, fare_conditions)"),
    ("idx_tf_junk7", "CREATE INDEX idx_tf_junk7 ON bookings.ticket_flights(ticket_no, amount)"),
    ("idx_tf_junk8", "CREATE INDEX idx_tf_junk8 ON bookings.ticket_flights(ticket_no, fare_conditions)"),
    ("idx_tf_junk9", "CREATE INDEX idx_tf_junk9 ON bookings.ticket_flights(flight_id, fare_conditions, amount)"),
    ("idx_tf_junk10", "CREATE INDEX idx_tf_junk10 ON bookings.ticket_flights(fare_conditions, flight_id, amount)"),
    ("idx_tf_junk11", "CREATE INDEX idx_tf_junk11 ON bookings.ticket_flights(amount DESC)"),
    ("idx_tf_junk12", "CREATE INDEX idx_tf_junk12 ON bookings.ticket_flights(fare_conditions, ticket_no)"),
    ("idx_tf_junk13", "CREATE INDEX idx_tf_junk13 ON bookings.ticket_flights(fare_conditions) WHERE fare_conditions = 'Business'"),
    ("idx_tf_junk14", "CREATE INDEX idx_tf_junk14 ON bookings.ticket_flights(amount, ticket_no)"),
    ("idx_tf_junk15", "CREATE INDEX idx_tf_junk15 ON bookings.ticket_flights(fare_conditions, amount, flight_id)"),
]

# ── Stale statistics status values ──────────────────────────────────

STALE_STATS_STATUSES = ["Arrived", "Scheduled", "On Time"]

# ── Task registry ───────────────────────────────────────────────────
# Ordered by difficulty: easy (1-5) → medium (6-11) → hard (12-17).
#
# Each task includes fatal_patterns (commands that terminate the episode with
# a -0.5 penalty) and allowed_dangerous (overrides for commands that are the
# correct fix). E.g., DROP INDEX is normally fatal but is allowed for the
# over_indexing task. This task-aware safety layer prevents reward hacking
# via blanket destructive commands while allowing legitimate fixes.

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ══════════════════════════════════════════════════════════════
    # EASY (tasks 1–5): Single-fault diagnosis, one clear root cause.
    # Baseline models score 0.7-1.0 on these. Solvable in 2-5 steps.
    # ══════════════════════════════════════════════════════════════
    "task_1": {
        "name": "Missing Index",
        "fault_type": "missing_index",
        "difficulty": "easy",
        "description": (
            "A critical index is missing on the ticket_flights table, causing "
            "flight segment lookups to perform sequential scans on 8+ million rows. "
            "Diagnose the slow query and create the appropriate index."
        ),
        "alert": ALERTS["missing_index"],
        "params": {
            "target_table": "ticket_flights",
            "target_column": "flight_id",
            "index_name": "idx_ticket_flights_flight",
            "target_query": (
                "EXPLAIN ANALYZE SELECT tf.ticket_no, tf.fare_conditions, tf.amount "
                "FROM bookings.ticket_flights tf WHERE tf.flight_id = 2880"
            ),
        },
        "fatal_patterns": ["VACUUM FULL"],
        "allowed_dangerous": [],
    },
    "task_2": {
        "name": "Stale Statistics",
        "fault_type": "stale_statistics",
        "difficulty": "easy",
        "description": (
            "After a bulk data migration, query planner statistics are wildly "
            "inaccurate. The planner is choosing terrible execution plans because "
            "it thinks certain status values appear in ~40 rows when they actually "
            "appear in 100,000+. Run ANALYZE to fix statistics."
        ),
        "alert": ALERTS["stale_statistics"],
        "params": {
            "target_table": "flights",
            "update_status_from": "Arrived",
            "update_status_to": "Delayed",
            "update_count": 100000,
            "target_query": (
                "EXPLAIN ANALYZE SELECT * FROM bookings.flights "
                "WHERE status = 'Delayed'"
            ),
        },
        "fatal_patterns": ["VACUUM FULL", "REINDEX"],
        "allowed_dangerous": [],
    },
    "task_3": {
        "name": "Connection Exhaustion",
        "fault_type": "connection_exhaustion",
        "difficulty": "easy",
        "description": (
            "The database connection pool is nearly exhausted. Dozens of sessions "
            "are sitting in 'idle in transaction' state, consuming connection slots. "
            "Terminate the idle sessions and configure a timeout to prevent recurrence."
        ),
        "alert": ALERTS["connection_exhaustion"],
        "params": {
            "num_connections_base": 80,
            "num_connections_range": 10,  # actual = base + random(0, range)
        },
        "fatal_patterns": [],
        "allowed_dangerous": [],
    },
    "task_4": {
        "name": "Permission / Role Error",
        "fault_type": "permission_error",
        "difficulty": "easy",
        "description": (
            "The application user 'app_user' has lost SELECT permission on the "
            "ticket_flights table. Queries from the application fail with "
            "'permission denied'. Grant the correct permission back."
        ),
        "alert": ALERTS["permission_error"],
        "params": {
            "role_name": "app_user",
            "role_password": "apppass",
            "target_table": "ticket_flights",
            "target_schema": "bookings",
            "revoked_privilege": "SELECT",
        },
        "fatal_patterns": ["WITH SUPERUSER"],
        "allowed_dangerous": [],
    },
    "task_5": {
        "name": "Sequence Exhaustion / PK Conflict",
        "fault_type": "sequence_exhaustion",
        "difficulty": "easy",
        "description": (
            "The sequence backing the flights.flight_id primary key has been reset "
            "to 1. INSERT operations fail with duplicate key violations because "
            "flight_id=1 already exists. Reset the sequence to the correct value."
        ),
        "alert": ALERTS["sequence_exhaustion"],
        "params": {
            "target_table": "flights",
            "sequence_name": "bookings.flights_flight_id_seq",
            "pk_column": "flight_id",
        },
        "fatal_patterns": ["DROP SEQUENCE", "RESTART WITH 1"],
        "allowed_dangerous": [],
    },
    # ══════════════════════════════════════════════════════════════
    # MEDIUM (tasks 6–11): Multi-step investigation, ambiguity in
    # diagnosis. The agent must choose between plausible fixes or
    # handle faults with non-obvious symptoms. Baseline models score
    # 0.4-0.9 on these. Typical resolution: 6-12 steps.
    # ══════════════════════════════════════════════════════════════
    "task_6": {
        "name": "Bad Configuration",
        "fault_type": "bad_config",
        "difficulty": "medium",
        "description": (
            "Critical PostgreSQL memory settings have been set to terrible values. "
            "work_mem is only 64kB (causing sorts/hashes to spill to disk) and "
            "effective_cache_size is 1MB (causing the planner to avoid index scans). "
            "Diagnose the misconfiguration and set reasonable values."
        ),
        "alert": ALERTS["bad_config"],
        "params": {
            "bad_settings": {
                "work_mem": "64kB",
                "effective_cache_size": "1MB",
            },
            "target_query": (
                "EXPLAIN ANALYZE SELECT t.ticket_no, t.passenger_name, tf.amount "
                "FROM bookings.tickets t "
                "JOIN bookings.ticket_flights tf ON t.ticket_no = tf.ticket_no "
                "WHERE tf.amount > 50000"
            ),
        },
        "fatal_patterns": ["ALTER SYSTEM RESET ALL"],
        "allowed_dangerous": [],
    },
    "task_7": {
        "name": "Lock Contention",
        "fault_type": "lock_contention",
        "difficulty": "medium",
        "description": (
            "A single transaction is holding a row-level lock on the bookings table "
            "and blocking multiple other queries. Identify the blocking process and "
            "terminate it to free the blocked queries."
        ),
        "alert": ALERTS["lock_contention"],
        "params": {
            "target_table": "bookings",
            "book_refs": LOCK_BOOK_REFS,
            "num_waiters": 3,
        },
        "fatal_patterns": ["LOCK TABLE"],
        "allowed_dangerous": [],
    },
    "task_8": {
        "name": "Table Bloat / Vacuum Stuck",
        "fault_type": "table_bloat",
        "difficulty": "medium",
        "description": (
            "A long-running transaction is preventing autovacuum from cleaning up "
            "dead tuples in the bookings table. The table has accumulated 200K+ "
            "dead tuples. Find and terminate the blocking transaction, then vacuum."
        ),
        "alert": ALERTS["table_bloat"],
        "params": {
            "target_table": "bookings",
            "dead_tuple_count_base": 200000,
            "dead_tuple_count_range": 50000,
        },
        "fatal_patterns": ["VACUUM FULL"],
        "allowed_dangerous": [],
    },
    "task_9": {
        "name": "Over-Indexing",
        "fault_type": "over_indexing",
        "difficulty": "medium",
        "description": (
            "The ticket_flights table has accumulated many unnecessary indexes "
            "that are slowing down write operations. Identify indexes with zero "
            "scans (idx_scan = 0) and drop them while preserving essential indexes."
        ),
        "alert": ALERTS["over_indexing"],
        "params": {
            "target_table": "ticket_flights",
            "num_junk_indexes_base": 8,
            "num_junk_indexes_range": 5,  # 8-12 junk indexes
            "junk_pool": JUNK_INDEX_POOL,
        },
        "fatal_patterns": [],
        "allowed_dangerous": [],
    },
    "task_10": {
        "name": "Index Bloat / Fragmented Index",
        "fault_type": "index_bloat",
        "difficulty": "medium",
        "description": (
            "An index on the ticket_flights table has become bloated from many "
            "update cycles. The index is 25%+ larger than it should be, making "
            "index scans slower than expected. Rebuild the index to reclaim space."
        ),
        "alert": ALERTS["index_bloat"],
        "params": {
            "target_table": "ticket_flights",
            "target_index": "idx_ticket_flights_flight",
            "target_column": "flight_id",
            "update_rounds": 3,
            "update_batch_size": 100000,
        },
        "fatal_patterns": ["VACUUM FULL"],
        "allowed_dangerous": ["REINDEX"],
    },
    "task_11": {
        "name": "Wrong Index Column Order",
        "fault_type": "wrong_index_order",
        "difficulty": "medium",
        "description": (
            "The ticket_flights table has a composite primary key on (ticket_no, flight_id). "
            "Queries filtering only on flight_id cannot efficiently use this index because "
            "flight_id is the second column. A standalone index on flight_id is needed."
        ),
        "alert": ALERTS["wrong_index_order"],
        "params": {
            "target_table": "ticket_flights",
            "target_column": "flight_id",
            "index_to_drop": "idx_ticket_flights_flight",
            "target_query": (
                "EXPLAIN ANALYZE SELECT tf.ticket_no, tf.fare_conditions, tf.amount "
                "FROM bookings.ticket_flights tf WHERE tf.flight_id = 2880"
            ),
        },
        "fatal_patterns": ["VACUUM FULL"],
        "allowed_dangerous": [],
    },
    # ══════════════════════════════════════════════════════════════
    # HARD (tasks 12–17): Compound faults requiring multi-root-cause
    # analysis. Two simultaneous faults interact, forcing the agent to
    # prioritize and coordinate fixes. Fixing only one yields partial
    # credit. Current frontier models achieve 0.3-0.7 on these,
    # leaving significant headroom for RL training improvement.
    # ══════════════════════════════════════════════════════════════
    "task_12": {
        "name": "Compound: Stale Stats + Missing Index",
        "fault_type": "compound_stats_index",
        "difficulty": "hard",
        "description": (
            "A query is suffering from TWO performance problems simultaneously: "
            "a missing index AND stale statistics. Fixing only one may not fully "
            "resolve the issue — or may make it worse. Both must be addressed."
        ),
        "alert": ALERTS["compound_stats_index"],
        "params": {
            # Combines task_1 and task_2 params
            "target_table_index": "ticket_flights",
            "target_column": "flight_id",
            "index_name": "idx_ticket_flights_flight",
            "target_table_stats": "flights",
            "update_status_from": "Arrived",
            "update_status_to": "Delayed",
            "update_count": 100000,
            "target_query": (
                "EXPLAIN ANALYZE SELECT tf.ticket_no, tf.fare_conditions, tf.amount, f.status "
                "FROM bookings.ticket_flights tf "
                "JOIN bookings.flights f ON f.flight_id = tf.flight_id "
                "WHERE f.status = 'Delayed'"
            ),
        },
        "fatal_patterns": ["VACUUM FULL"],
        "allowed_dangerous": [],
    },
    "task_13": {
        "name": "Compound: Lock + Bloat",
        "fault_type": "compound_lock_bloat",
        "difficulty": "hard",
        "description": (
            "A single long-running transaction is causing TWO problems: it holds "
            "row locks blocking other queries AND it prevents autovacuum from "
            "cleaning dead tuples. Both lock waits and table bloat must be resolved."
        ),
        "alert": ALERTS["compound_lock_bloat"],
        "params": {
            # Combines lock_contention and table_bloat params
            "target_table": "bookings",
            "book_refs": LOCK_BOOK_REFS,
            "num_waiters": 3,
            "dead_tuple_count_base": 200000,
            "dead_tuple_count_range": 50000,
        },
        "fatal_patterns": ["LOCK TABLE", "VACUUM FULL"],
        "allowed_dangerous": [],
    },
    "task_14": {
        "name": "Deadlock Chain",
        "fault_type": "deadlock_chain",
        "difficulty": "hard",
        "description": (
            "Two concurrent transactions are updating the same booking rows in "
            "opposite order, causing a deadlock. The transactions are stuck waiting "
            "on each other. Investigate the deadlock pattern and resolve the issue."
        ),
        "alert": ALERTS["deadlock_chain"],
        "params": {
            "target_table": "bookings",
            "book_ref_a": "361A07",
            "book_ref_b": "363381",
        },
        "fatal_patterns": [],
        "allowed_dangerous": [],
    },
    "task_15": {
        "name": "Query Plan Flip",
        "fault_type": "query_plan_flip",
        "difficulty": "hard",
        "description": (
            "The random_page_cost parameter has been set to an extreme value (100), "
            "causing the query planner to strongly prefer sequential scans over index "
            "scans. A query that was sub-millisecond is now taking 30ms+. "
            "Diagnose the planner misconfiguration and reset the parameter."
        ),
        "alert": ALERTS["query_plan_flip"],
        "params": {
            "bad_param": "random_page_cost",
            "bad_value": "100",
            "target_query": (
                "EXPLAIN ANALYZE SELECT tf.ticket_no, tf.fare_conditions, tf.amount "
                "FROM bookings.ticket_flights tf WHERE tf.flight_id = 2880"
            ),
        },
        "fatal_patterns": ["ALTER SYSTEM RESET ALL"],
        "allowed_dangerous": [],
    },
    "task_16": {
        "name": "Cascading Bloat (Multi-Table)",
        "fault_type": "cascading_bloat",
        "difficulty": "hard",
        "description": (
            "A long-running REPEATABLE READ transaction is holding a snapshot open, "
            "preventing autovacuum from cleaning ANY table. Dead tuples are accumulating "
            "across bookings, flights, ticket_flights, and tickets simultaneously. "
            "Kill the blocking transaction and vacuum all affected tables."
        ),
        "alert": ALERTS["cascading_bloat"],
        "params": {
            "tables": ["bookings", "flights", "ticket_flights", "tickets"],
            "update_count_per_table": 50000,
        },
        "fatal_patterns": [],
        "allowed_dangerous": ["VACUUM FULL"],
    },
    "task_17": {
        "name": "Compound: Connection Exhaustion + Deadlock",
        "fault_type": "compound_conn_deadlock",
        "difficulty": "hard",
        "description": (
            "The database is suffering from TWO simultaneous problems: (1) 85 idle-in-transaction "
            "connections are consuming nearly all connection slots, AND (2) a deadlock exists "
            "between two active transactions. Both must be resolved: terminate idle sessions, "
            "set a timeout, and address the deadlock."
        ),
        "alert": ALERTS["compound_conn_deadlock"],
        "params": {
            "num_connections_base": 80,
            "num_connections_range": 5,
            "target_table": "bookings",
            "book_ref_a": "361A07",
            "book_ref_b": "363381",
        },
        "fatal_patterns": [],
        "allowed_dangerous": ["VACUUM FULL"],
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    """Look up a task by ID. Raises KeyError if not found."""
    if task_id not in TASK_REGISTRY:
        raise KeyError(f"Unknown task_id: {task_id!r}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]


def list_task_ids() -> list:
    """Return all available task IDs."""
    return list(TASK_REGISTRY.keys())
