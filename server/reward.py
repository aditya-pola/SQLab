"""
SQLab — Per-step reward shaping.

Provides small positive rewards for diagnostic and corrective actions,
and small negative rewards for errors, destructive actions, and repetition.

Per-step rewards are essential for RL sample efficiency: without them, the
agent receives zero learning signal until the episode-ending grader fires,
making credit assignment across a 15-step episode nearly impossible.  These
shaped rewards give the policy gradient meaningful direction on every step.

Three anti-reward-hacking mechanisms prevent degenerate strategies:
  1. Fault-type gating — corrective rewards only fire on relevant fault types.
  2. Target-awareness — diagnostic rewards require the query to reference the
     fault's actual target table (from inject_meta).
  3. Deduplication — each reward category fires at most once per episode, so
     repeating a useful command yields nothing.

Rewards are:
- Fault-type-gated: corrective actions only reward on relevant fault types
- Target-aware: diagnostics must reference the right table/entity
- Deduplicated: each reward category fires at most once per episode
- Clamped: cumulative reward stays in [0, 1] (enforced in environment.py)

Validated against 255 adversarial unit tests covering reward-farming loops,
wrong-fault-type exploits, and degenerate action sequences.

Design rationale: most RL environments for code/tool use provide only a
sparse terminal reward.  This works for short-horizon tasks (e.g. single
function synthesis) but fails for multi-step incident response where the
agent must first diagnose, then fix, then verify — three qualitatively
different sub-goals within one episode.  Shaped per-step rewards bridge
each sub-goal transition without leaking the grader answer.
"""

import logging
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# ── Corrective action gates ────────────────────────────────────
# Fault-type gating is the primary anti-reward-hacking mechanism.  An agent
# cannot earn CREATE INDEX reward on a lock_contention task, nor VACUUM reward
# on a missing_index task.  Each corrective keyword maps to only the fault
# types where it constitutes a valid fix, preventing brute-force strategies
# that cycle through every possible remediation command.
CORRECTIVE_GATES = {
    "CREATE INDEX": (
        "missing_index", "wrong_index_order", "compound_stats_index",
    ),
    "REINDEX": ("index_bloat",),
    "VACUUM": (
        "table_bloat", "compound_lock_bloat", "cascading_bloat",
    ),
    "ANALYZE": (
        "stale_statistics", "compound_stats_index",
    ),
    "ALTER SYSTEM": (
        "bad_config", "query_plan_flip", "connection_exhaustion",
    ),
    "PG_RELOAD_CONF": (
        "bad_config", "query_plan_flip", "connection_exhaustion",
    ),
    "PG_TERMINATE_BACKEND": (
        "connection_exhaustion", "lock_contention", "deadlock_chain",
        "compound_lock_bloat", "compound_conn_deadlock",
    ),
    "DROP INDEX": ("over_indexing",),
    "GRANT": ("permission_error",),
    "SETVAL": ("sequence_exhaustion",),
    "ALTER DATABASE": ("query_plan_flip",),
    "RESET": ("query_plan_flip", "bad_config"),
}

# ── Diagnostic gates ──────────────────────────────────────────
# Diagnostic gating prevents reward for irrelevant catalog queries.  Querying
# pg_locks only earns reward on lock/deadlock tasks; querying pg_settings only
# on config tasks.  This forces the agent to develop fault-specific diagnostic
# strategies rather than dumping every system catalog on every episode.
DIAGNOSTIC_FAULT_GATES = {
    "PG_STAT_ACTIVITY": (
        "connection_exhaustion", "lock_contention", "deadlock_chain",
        "table_bloat", "compound_lock_bloat", "cascading_bloat",
        "compound_conn_deadlock",
    ),
    "PG_LOCKS": (
        "lock_contention", "deadlock_chain", "compound_lock_bloat",
        "compound_conn_deadlock",
    ),
    "PG_SETTINGS": ("bad_config", "query_plan_flip"),
    "PG_SEQUENCES": ("sequence_exhaustion",),
}

# Table-targeting diagnostics — must mention the target entity.
# These keywords only earn reward when the command also references the fault's
# actual target table (extracted from inject_meta), preventing generic EXPLAIN
# on unrelated tables from earning diagnostic credit.  This is the second
# anti-hacking layer: even if the agent guesses the right diagnostic tool, it
# must apply it to the right table — requiring genuine fault comprehension.
TABLE_DIAGNOSTICS = [
    "EXPLAIN", "PG_INDEXES", "PG_STAT_USER_TABLES",
    "PG_STAT_USER_INDEXES", "PG_RELATION_SIZE", "PG_SIZE_PRETTY",
]

# Destructive keywords — penalised.  In production, DROP TABLE during an
# incident is a career-ending mistake.  The penalty here teaches agents the
# same operational discipline that human SREs learn on day one.
DESTRUCTIVE_KEYWORDS = [
    "DROP TABLE",
    "TRUNCATE",
    "DELETE FROM",
]


def _reward_once(rewarded_set: Optional[Set[str]], category: str, amount: float) -> float:
    """Give reward only if this category hasn't been rewarded yet.

    Deduplication prevents reward farming: running the same diagnostic five
    times earns the same reward as running it once.  The rewarded_set persists
    across all steps in an episode, so the agent must explore diverse actions.
    """
    if rewarded_set is not None and category in rewarded_set:
        return 0.0
    if rewarded_set is not None:
        rewarded_set.add(category)
    return amount


def _build_target_set(inject_meta: Optional[dict]) -> set:
    """Extract all target entity names from inject_meta for matching."""
    meta = inject_meta or {}
    candidates = [
        meta.get("target_table", ""),
        meta.get("target_column", ""),
    ]
    # Compound task sub-metas
    for sub_key in ("index_meta", "stats_meta"):
        sub = meta.get(sub_key, {})
        if isinstance(sub, dict):
            candidates.append(sub.get("target_table", ""))
            candidates.append(sub.get("target_column", ""))
    return {t.upper() for t in candidates if t}


def compute_step_reward(
    command: str,
    output: str,
    error: str | None,
    fault_type: str,
    action_history: List[str],
    inject_meta: dict = None,
    rewarded_set: set = None,
) -> float:
    """Compute reward for a single step.

    Returns a float (can be positive or negative).
    Per-step range approximately [-0.10, +0.15].  The asymmetry is intentional:
    correct diagnostic/corrective actions are rewarded more than bad actions are
    penalised, biasing exploration toward productive commands rather than
    freezing the agent with excessive negative signal.

    Cumulative reward is clamped to [0, 1] in environment.py, keeping rewards
    on the same scale as the grader score for straightforward RL loss functions.

    Args:
        inject_meta: Target metadata (target_table, target_column, etc.)
        rewarded_set: Mutable set tracking which reward categories have fired.
                      Persisted on the environment across steps in an episode.
    """
    reward = 0.0
    cmd_upper = command.upper().strip()

    all_targets = _build_target_set(inject_meta)

    # ── Positive: diagnostic commands (target-aware) ──────────
    # Diagnostic rewards use two gating strategies: system-catalog queries are
    # gated by fault_type, while table-targeting queries must also reference the
    # correct target table from inject_meta.  This ensures reward only flows for
    # contextually relevant investigation, not shotgun catalog dumps.
    # Together with deduplication, these gates make the optimal policy identical
    # to expert SRE behaviour: query the right catalog, for the right table, once.

    # System catalog diagnostics — gated by fault_type
    catalog_rewarded = False
    for catalog_kw, valid_faults in DIAGNOSTIC_FAULT_GATES.items():
        if catalog_kw in cmd_upper and fault_type in valid_faults:
            reward += _reward_once(rewarded_set, f"diag_{catalog_kw.lower()}", 0.05)
            catalog_rewarded = True
            break

    # Table-targeting diagnostics — must mention target entity
    if not catalog_rewarded:
        for kw in TABLE_DIAGNOSTICS:
            if kw in cmd_upper:
                if all_targets and any(t in cmd_upper for t in all_targets):
                    reward += _reward_once(rewarded_set, f"diag_{kw.lower()}", 0.05)
                # No reward for wrong-table diagnostics
                break

    # SHOW is a special case — useful for config tasks
    if "SHOW " in cmd_upper and fault_type in ("bad_config", "query_plan_flip"):
        reward += _reward_once(rewarded_set, "diag_show", 0.05)

    # ── Positive: corrective actions (fault-type-gated) ───────
    # Each corrective reward is gated by CORRECTIVE_GATES: the agent only earns
    # credit if the fix type matches the injected fault.  Higher rewards (0.10)
    # go to primary fixes; secondary supportive actions earn 0.05.
    # The 2:1 ratio between primary and secondary rewards encodes domain knowledge
    # about which actions resolve vs. merely mitigate a fault — for example,
    # CREATE INDEX is primary for missing_index, while ANALYZE is supportive.

    if "CREATE INDEX" in cmd_upper and error is None:
        if fault_type in CORRECTIVE_GATES["CREATE INDEX"]:
            reward += _reward_once(rewarded_set, "create_index", 0.10)

    if "PG_TERMINATE_BACKEND" in cmd_upper and error is None:
        if fault_type in CORRECTIVE_GATES["PG_TERMINATE_BACKEND"]:
            reward += _reward_once(rewarded_set, "terminate_backend", 0.05)

    if "VACUUM" in cmd_upper and error is None:
        if fault_type in CORRECTIVE_GATES["VACUUM"]:
            reward += _reward_once(rewarded_set, "vacuum", 0.05)

    if "ANALYZE" in cmd_upper and "EXPLAIN" not in cmd_upper and error is None:
        if fault_type in CORRECTIVE_GATES["ANALYZE"]:
            reward += _reward_once(rewarded_set, "analyze", 0.05)

    if "ALTER SYSTEM" in cmd_upper and error is None:
        if fault_type in CORRECTIVE_GATES["ALTER SYSTEM"]:
            reward += _reward_once(rewarded_set, "alter_system", 0.05)

    if "PG_RELOAD_CONF" in cmd_upper and error is None:
        if fault_type in CORRECTIVE_GATES["PG_RELOAD_CONF"]:
            reward += _reward_once(rewarded_set, "reload_conf", 0.05)

    if "DROP INDEX" in cmd_upper and error is None:
        if fault_type in CORRECTIVE_GATES["DROP INDEX"]:
            reward += _reward_once(rewarded_set, "drop_index", 0.05)

    if "REINDEX" in cmd_upper and error is None:
        if fault_type in CORRECTIVE_GATES["REINDEX"]:
            reward += _reward_once(rewarded_set, "reindex", 0.10)

    if "GRANT" in cmd_upper and "REVOKE" not in cmd_upper and error is None:
        if fault_type in CORRECTIVE_GATES["GRANT"]:
            reward += _reward_once(rewarded_set, "grant", 0.10)

    if "SETVAL" in cmd_upper and error is None:
        if fault_type in CORRECTIVE_GATES["SETVAL"]:
            reward += _reward_once(rewarded_set, "setval", 0.10)

    if "ALTER DATABASE" in cmd_upper and error is None:
        if fault_type in CORRECTIVE_GATES["ALTER DATABASE"]:
            reward += _reward_once(rewarded_set, "alter_database", 0.05)

    if "RESET" in cmd_upper and error is None:
        if fault_type in CORRECTIVE_GATES["RESET"]:
            reward += _reward_once(rewarded_set, "reset_param", 0.05)

    # ── Negative: wrong-corrective penalty ─────────────────────
    # Applying a corrective action for the wrong fault type incurs a small
    # penalty.  This discourages brute-force "try every fix" strategies and
    # pushes the agent toward diagnosing the fault before attempting a fix.
    for keyword, valid_faults in CORRECTIVE_GATES.items():
        if keyword in cmd_upper and error is None and fault_type not in valid_faults:
            reward -= 0.03
            break  # only penalise once

    # ── Negative: errors ─────────────────────────────────────────
    # Syntax errors and permission failures cost -0.05, teaching the agent to
    # generate valid SQL — a transferable skill for any database agent task.
    if error is not None:
        reward -= 0.05

    # ── Negative: destructive commands ───────────────────────────
    if any(kw in cmd_upper for kw in DESTRUCTIVE_KEYWORDS):
        reward -= 0.10

    # ── Negative: exact duplicate command ────────────────────────
    # Exact-match repeated commands lose points, preventing degenerate loops
    # where the agent spams the same query to fill the episode budget.
    if command.strip() in [a.strip() for a in action_history[:-1]]:
        reward -= 0.03

    # ── Negative: empty or trivial commands ──────────────────────
    # SELECT 1 is a common no-op probe.  Penalising it prevents the agent from
    # burning steps on connectivity checks instead of investigating the fault.
    if cmd_upper in ("SELECT 1", "SELECT 1;", ""):
        reward -= 0.02

    # Round to 4 decimal places to avoid floating-point drift across 15 steps.
    # The cumulative sum is clamped to [0, 1] in environment.py, keeping per-step
    # shaping and the terminal grader score on a unified scale for RL loss.
    return round(reward, 4)
