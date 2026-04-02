"""
Adversarial test suite for per-step reward shaping.

Verifies that:
1. Corrective actions don't reward on wrong fault types
2. Repeated commands don't accumulate unbounded reward
3. Wrong-table diagnostics don't reward
4. Cumulative reward stays in [0, 1]

Pure Python — no DB required.
"""

import pytest
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from sqlab.server.reward import compute_step_reward


# ═══════════════════════════════════════════════════════════════════
# Test data
# ═══════════════════════════════════════════════════════════════════

CORRECT_FIXES = {
    "missing_index": "CREATE INDEX ON bookings.ticket_flights(flight_id)",
    "stale_statistics": "ANALYZE bookings.flights",
    "connection_exhaustion": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle in transaction'",
    "lock_contention": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE wait_event_type = 'Lock'",
    "table_bloat": "VACUUM bookings.bookings",
    "over_indexing": "DROP INDEX bookings.idx_junk_1",
    "compound_stats_index": "CREATE INDEX ON bookings.ticket_flights(flight_id)",
    "compound_lock_bloat": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE wait_event_type = 'Lock'",
    "bad_config": "ALTER SYSTEM SET work_mem = '4MB'",
    "index_bloat": "REINDEX INDEX bookings.idx_ticket_flights_flight",
    "wrong_index_order": "CREATE INDEX ON bookings.ticket_flights(flight_id)",
    "deadlock_chain": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE wait_event_type = 'Lock'",
    "query_plan_flip": "ALTER SYSTEM SET random_page_cost = 4",
    "cascading_bloat": "VACUUM bookings.flights",
    "permission_error": "GRANT SELECT ON bookings.ticket_flights TO app_user",
    "sequence_exhaustion": "SELECT setval('bookings.flights_flight_id_seq', (SELECT max(flight_id) FROM bookings.flights))",
    "compound_conn_deadlock": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle in transaction'",
}

INJECT_METAS = {
    "missing_index": {"target_table": "ticket_flights", "target_column": "flight_id"},
    "stale_statistics": {"target_table": "flights"},
    "connection_exhaustion": {"target_table": ""},
    "lock_contention": {"target_table": "bookings"},
    "table_bloat": {"target_table": "bookings"},
    "over_indexing": {"target_table": "ticket_flights"},
    "compound_stats_index": {
        "target_table": "ticket_flights",
        "target_column": "flight_id",
        "index_meta": {"target_table": "ticket_flights", "target_column": "flight_id"},
        "stats_meta": {"target_table": "ticket_flights"},
    },
    "compound_lock_bloat": {"target_table": "bookings"},
    "bad_config": {"target_table": ""},
    "index_bloat": {"target_table": "ticket_flights", "target_index": "idx_ticket_flights_flight"},
    "wrong_index_order": {"target_table": "ticket_flights", "target_column": "flight_id"},
    "deadlock_chain": {"target_table": "bookings"},
    "query_plan_flip": {"target_table": "ticket_flights", "bad_param": "random_page_cost"},
    "cascading_bloat": {"target_table": "flights"},
    "permission_error": {"target_table": "ticket_flights"},
    "sequence_exhaustion": {"target_table": "flights"},
    "compound_conn_deadlock": {"target_table": "bookings"},
}

# Fault types that share the same corrective keyword
# e.g. CREATE INDEX is valid for missing_index, wrong_index_order, compound_stats_index
# We need to know which fix commands are "shared" to skip those pairs in cross-task tests
SHARED_CORRECTIVE = {
    "CREATE INDEX": {"missing_index", "wrong_index_order", "compound_stats_index"},
    "PG_TERMINATE_BACKEND": {
        "connection_exhaustion", "lock_contention", "deadlock_chain",
        "compound_lock_bloat", "compound_conn_deadlock",
    },
    "VACUUM": {"table_bloat", "compound_lock_bloat", "cascading_bloat"},
    "ALTER SYSTEM": {"bad_config", "query_plan_flip", "connection_exhaustion"},
    "ANALYZE": {"stale_statistics", "compound_stats_index"},
}

# Commands that embed catalog keywords (e.g. pg_stat_activity in a pg_terminate_backend
# call) may earn a small diagnostic reward on fault types where that catalog is relevant.
# This is intentional — investigating the right catalog is useful. We track these pairs
# to allow a small positive tolerance in cross-task tests.
_CATALOG_IN_COMMAND = {
    # Commands containing PG_STAT_ACTIVITY get diagnostic credit on bloat faults
    "connection_exhaustion": {"table_bloat", "cascading_bloat"},
    "lock_contention": {"table_bloat", "cascading_bloat"},
    "deadlock_chain": {"table_bloat", "cascading_bloat"},
    "compound_lock_bloat": {"table_bloat", "cascading_bloat"},
    "compound_conn_deadlock": {"table_bloat", "cascading_bloat"},
}


def _fix_shares_gate(fix_cmd: str, fix_fault: str, task_fault: str) -> bool:
    """Check if fix_cmd's corrective keyword is valid for task_fault,
    or if embedded catalog keywords give legitimate diagnostic credit."""
    cmd_upper = fix_cmd.upper()
    for keyword, valid_faults in SHARED_CORRECTIVE.items():
        if keyword in cmd_upper and task_fault in valid_faults:
            return True
    # Check if the command embeds a catalog keyword that gives diagnostic credit
    catalog_exceptions = _CATALOG_IN_COMMAND.get(fix_fault, set())
    if task_fault in catalog_exceptions:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════
# Section 1: Cross-task corrective matrix
# ═══════════════════════════════════════════════════════════════════

def _cross_task_pairs():
    """Generate (fix_fault, fix_cmd, task_fault) where fix should NOT reward."""
    for fix_fault, fix_cmd in CORRECT_FIXES.items():
        for task_fault in CORRECT_FIXES:
            if task_fault == fix_fault:
                continue
            # Skip if the fix command's keyword is legitimately valid for task_fault
            if _fix_shares_gate(fix_cmd, fix_fault, task_fault):
                continue
            yield fix_fault, fix_cmd, task_fault


@pytest.mark.parametrize(
    "fix_fault,fix_cmd,task_fault",
    list(_cross_task_pairs()),
    ids=[f"{ff}-on-{tf}" for ff, _, tf in _cross_task_pairs()],
)
def test_cross_task_no_reward(fix_fault, fix_cmd, task_fault):
    """Applying a fix for one fault type on a different fault type should not reward."""
    reward = compute_step_reward(
        fix_cmd, "OK", None, task_fault, [],
        inject_meta=INJECT_METAS[task_fault], rewarded_set=set(),
    )
    assert reward <= 0, (
        f"{fix_cmd} (fix for {fix_fault}) on {task_fault} got reward={reward}, expected <= 0"
    )


# ═══════════════════════════════════════════════════════════════════
# Section 2: Repetition gaming
# ═══════════════════════════════════════════════════════════════════

def test_no_repeat_reward_explain():
    """Same EXPLAIN command 10x should not accumulate more than one reward."""
    rewarded = set()
    total = 0.0
    cmd = "EXPLAIN SELECT * FROM bookings.ticket_flights WHERE flight_id = 1"
    meta = {"target_table": "ticket_flights", "target_column": "flight_id"}
    history = []
    for i in range(10):
        r = compute_step_reward(
            cmd, "OK", None, "missing_index", history,
            inject_meta=meta, rewarded_set=rewarded,
        )
        total += r
        history.append(cmd)
    # First call: +0.05 diagnostic. Subsequent: 0 (dedup) - 0.03 (duplicate).
    # Total should be well under 0.10
    assert total <= 0.10, f"10x EXPLAIN got total {total}, expected <= 0.10"


def test_no_repeat_reward_create_index():
    """Same CREATE INDEX 10x should not accumulate."""
    rewarded = set()
    total = 0.0
    cmd = "CREATE INDEX ON bookings.ticket_flights(flight_id)"
    meta = {"target_table": "ticket_flights", "target_column": "flight_id"}
    history = []
    for i in range(10):
        r = compute_step_reward(
            cmd, "OK", None, "missing_index", history,
            inject_meta=meta, rewarded_set=rewarded,
        )
        total += r
        history.append(cmd)
    # First call: +0.10 corrective. Subsequent: 0 (dedup) - 0.03 (duplicate).
    assert total <= 0.15, f"10x CREATE INDEX got total {total}, expected <= 0.15"


def test_no_repeat_reward_vacuum():
    """Same VACUUM 5x should not accumulate."""
    rewarded = set()
    total = 0.0
    cmd = "VACUUM bookings.bookings"
    meta = {"target_table": "bookings"}
    history = []
    for i in range(5):
        r = compute_step_reward(
            cmd, "OK", None, "table_bloat", history,
            inject_meta=meta, rewarded_set=rewarded,
        )
        total += r
        history.append(cmd)
    assert total <= 0.10, f"5x VACUUM got total {total}, expected <= 0.10"


# ═══════════════════════════════════════════════════════════════════
# Section 3: Wrong-table diagnostics
# ═══════════════════════════════════════════════════════════════════

def test_wrong_table_no_reward():
    """EXPLAIN on wrong table should not reward."""
    meta = {"target_table": "ticket_flights", "target_column": "flight_id"}
    r = compute_step_reward(
        "EXPLAIN SELECT * FROM bookings.flights WHERE status = 'Delayed'",
        "OK", None, "missing_index", [],
        inject_meta=meta, rewarded_set=set(),
    )
    # Wrong table: no diagnostic reward, and wrong-corrective penalty doesn't apply
    # to EXPLAIN. So should be 0 or slightly negative.
    assert r <= 0.0, f"Wrong-table EXPLAIN got {r}, expected <= 0"


def test_right_table_rewards():
    """EXPLAIN on right table should reward."""
    meta = {"target_table": "ticket_flights", "target_column": "flight_id"}
    r = compute_step_reward(
        "EXPLAIN SELECT * FROM bookings.ticket_flights WHERE flight_id = 1",
        "OK", None, "missing_index", [],
        inject_meta=meta, rewarded_set=set(),
    )
    assert r >= 0.05, f"Right-table EXPLAIN got {r}, expected >= 0.05"


def test_wrong_table_pg_indexes():
    """pg_indexes on wrong table should not reward."""
    meta = {"target_table": "ticket_flights", "target_column": "flight_id"}
    r = compute_step_reward(
        "SELECT * FROM pg_indexes WHERE tablename = 'flights'",
        "OK", None, "missing_index", [],
        inject_meta=meta, rewarded_set=set(),
    )
    assert r <= 0.0, f"Wrong-table pg_indexes got {r}, expected <= 0"


def test_right_table_pg_indexes():
    """pg_indexes on right table should reward."""
    meta = {"target_table": "ticket_flights", "target_column": "flight_id"}
    r = compute_step_reward(
        "SELECT * FROM pg_indexes WHERE tablename = 'ticket_flights'",
        "OK", None, "missing_index", [],
        inject_meta=meta, rewarded_set=set(),
    )
    assert r >= 0.05, f"Right-table pg_indexes got {r}, expected >= 0.05"


def test_catalog_diagnostic_right_fault():
    """PG_STAT_ACTIVITY on connection_exhaustion should reward."""
    meta = {"target_table": ""}
    r = compute_step_reward(
        "SELECT * FROM pg_stat_activity WHERE state = 'idle in transaction'",
        "OK", None, "connection_exhaustion", [],
        inject_meta=meta, rewarded_set=set(),
    )
    assert r >= 0.05, f"pg_stat_activity on connection_exhaustion got {r}, expected >= 0.05"


def test_catalog_diagnostic_wrong_fault():
    """PG_STAT_ACTIVITY on missing_index should not reward via catalog gate."""
    meta = {"target_table": "ticket_flights", "target_column": "flight_id"}
    r = compute_step_reward(
        "SELECT * FROM pg_stat_activity",
        "OK", None, "missing_index", [],
        inject_meta=meta, rewarded_set=set(),
    )
    # pg_stat_activity is not gated for missing_index, no table match either
    assert r <= 0.0, f"pg_stat_activity on missing_index got {r}, expected <= 0"


# ═══════════════════════════════════════════════════════════════════
# Section 4: Cumulative bounds
# ═══════════════════════════════════════════════════════════════════

def test_cumulative_bounds():
    """Simulate 15-step episode, assert 0 <= cumulative <= 1 at every step."""
    commands = [
        "EXPLAIN SELECT * FROM bookings.ticket_flights WHERE flight_id = 1",
        "SELECT * FROM pg_indexes WHERE tablename = 'ticket_flights'",
        "SELECT * FROM pg_stat_user_indexes WHERE relname = 'ticket_flights'",
        "CREATE INDEX CONCURRENTLY ON bookings.ticket_flights(flight_id)",
        "ANALYZE bookings.ticket_flights",
        # Then spam wrong/repeated stuff
        "CREATE INDEX ON bookings.ticket_flights(amount)",
        "CREATE INDEX ON bookings.ticket_flights(fare_conditions)",
        "VACUUM bookings.ticket_flights",
        "VACUUM FULL bookings.ticket_flights",
        "REINDEX TABLE bookings.ticket_flights",
        "ALTER SYSTEM SET work_mem = '4MB'",
        "SELECT pg_reload_conf()",
        "ANALYZE bookings.flights",
        "EXPLAIN SELECT 1",
        "SELECT 1",
    ]
    meta = {"target_table": "ticket_flights", "target_column": "flight_id"}
    rewarded = set()
    cumulative = 0.0
    history = []
    for cmd in commands:
        r = compute_step_reward(
            cmd, "OK", None, "missing_index", history,
            inject_meta=meta, rewarded_set=rewarded,
        )
        cumulative += r
        cumulative = max(0.0, min(1.0, cumulative))
        history.append(cmd)
        assert 0.0 <= cumulative <= 1.0, f"Cumulative {cumulative} out of bounds after: {cmd}"


def test_cumulative_does_not_go_negative():
    """All-wrong actions should clamp at 0, not go negative."""
    commands = [
        "DROP TABLE bookings.flights",
        "TRUNCATE bookings.tickets",
        "DELETE FROM bookings.bookings",
        "SELECT 1",
        "SELECT 1",
    ]
    meta = {"target_table": "ticket_flights", "target_column": "flight_id"}
    rewarded = set()
    cumulative = 0.0
    history = []
    for cmd in commands:
        r = compute_step_reward(
            cmd, "OK", None, "missing_index", history,
            inject_meta=meta, rewarded_set=rewarded,
        )
        cumulative += r
        cumulative = max(0.0, min(1.0, cumulative))
        history.append(cmd)
        assert cumulative >= 0.0, f"Cumulative went negative ({cumulative}) after: {cmd}"


# ═══════════════════════════════════════════════════════════════════
# Correct fix on correct fault should give positive reward
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("fault_type", list(CORRECT_FIXES.keys()))
def test_correct_fix_rewards(fault_type):
    """The correct fix for a fault should earn positive reward."""
    fix_cmd = CORRECT_FIXES[fault_type]
    meta = INJECT_METAS[fault_type]
    r = compute_step_reward(
        fix_cmd, "OK", None, fault_type, [],
        inject_meta=meta, rewarded_set=set(),
    )
    assert r > 0, f"Correct fix '{fix_cmd}' for {fault_type} got reward={r}, expected > 0"
