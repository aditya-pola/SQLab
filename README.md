---
title: SQLab
emoji: 💾
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SQLab: Database Incident Response Training for LLM Agents

**[Try the live demo](https://huggingface.co/spaces/stvident/sqlab)**

SQL databases power nearly every production application — from booking systems to financial platforms. When they break, the symptoms are cryptic: queries that ran in milliseconds now take seconds, connections pile up until the pool is exhausted, transactions deadlock each other, and bloated tables silently degrade performance. Diagnosing these failures requires reading execution plans, inspecting lock graphs, and understanding how the query planner makes decisions — skills that take years to develop.

SQLab is an OpenEnv environment where LLM agents learn these skills. It presents **17 production-realistic PostgreSQL faults** — missing indexes, stale statistics, deadlock chains, cascading bloat, misconfigured parameters, and more — against a live database with 20 million rows of airline booking data. The agent receives an alert, has 15 steps to investigate and fix the issue using raw SQL, and is scored by a deterministic grader on diagnosis, resolution, and best practices (0–1 scale, fully reproducible, no LLM judge).

## How an Episode Works

1. `reset(task_id)` injects a fault into the live database and returns an alert
2. The agent issues raw SQL commands via `step(command)` — up to 15 steps
3. Each step returns the SQL output, error messages, and live database metrics
4. Episode ends when the fault is resolved, max steps reached, or a fatal action is detected
5. A deterministic grader scores the episode across diagnosis, resolution, and best practices

### Example: Missing Index

```
Alert: High query latency on ticket_flights (avg 2.3s, p99 8.1s)

Step 1: EXPLAIN SELECT * FROM bookings.ticket_flights WHERE flight_id = 1
  → Seq Scan on ticket_flights (cost=0.00..287434.12)     ← No index!
  → reward: +0.05 (targeted diagnostic)

Step 2: SELECT * FROM pg_indexes WHERE tablename = 'ticket_flights'
  → Only primary key, no index on flight_id
  → reward: +0.05 (right-table diagnostic)

Step 3: CREATE INDEX ON bookings.ticket_flights(flight_id)
  → CREATE INDEX (success)
  → reward: +0.10 (correct fix for missing_index)

Step 4: EXPLAIN SELECT * FROM bookings.ticket_flights WHERE flight_id = 1
  → Index Scan using idx_ticket_flights_flight_id (cost=0.43..8.45)     ← Fixed!
  → Grader: 0.85 (diagnosis 0.4 + resolution 0.4 + best practice 0.05)
```

Four steps: investigate, confirm, fix, verify. The grader rewards both the journey and the outcome.

## Real-World Utility

Every fault in SQLab is modeled on real PostgreSQL failure modes: a missing index causing 100x query slowdowns, bloated tables blocking autovacuum, a misconfigured `work_mem` silently degrading every query on the server. These are the same issues that production SREs encounter regularly.

The training database is the [Airlines demo](https://postgrespro.com/community/demodb): 20 million rows of flights, tickets, and bookings. Realistic enough that EXPLAIN plans behave like production, indexes matter, and lock contention actually blocks. The skills agents learn here transfer directly to real database operations.

Fault categories:
- **Performance**: missing indexes, stale statistics, wrong column order
- **Resources**: connection exhaustion, lock contention, deadlocks
- **Storage**: table bloat, index bloat, cascading multi-table bloat
- **Configuration**: bad settings, query plan flips
- **Access & Integrity**: permission errors, sequence exhaustion

## Tasks

17 tasks across 3 difficulty levels. Easy tasks involve a single clear fault. Medium tasks require multi-step investigation. Hard tasks present two simultaneous faults that the agent must prioritize and coordinate.

| # | Task | Difficulty | Fault Type | Description |
|---|------|-----------|------------|-------------|
| 1 | Missing Index | Easy | missing_index | Slow query due to sequential scan on un-indexed column |
| 2 | Stale Statistics | Easy | stale_statistics | Query planner makes bad choices due to outdated table stats |
| 3 | Connection Exhaustion | Easy | connection_exhaustion | Too many idle-in-transaction sessions consuming all connections |
| 4 | Permission / Role Error | Easy | permission_error | Application user lacks SELECT permission on a table |
| 5 | Sequence Exhaustion | Easy | sequence_exhaustion | Primary key sequence out of sync after bulk data load |
| 6 | Bad Configuration | Medium | bad_config | work_mem and effective_cache_size set to absurdly low values |
| 7 | Lock Contention | Medium | lock_contention | Long-running transaction holding row locks, blocking others |
| 8 | Table Bloat | Medium | table_bloat | Dead tuples accumulating because autovacuum is blocked |
| 9 | Over-Indexing | Medium | over_indexing | Too many unused indexes degrading write performance |
| 10 | Index Bloat | Medium | index_bloat | Fragmented index many times larger than it should be |
| 11 | Wrong Index Order | Medium | wrong_index_order | Composite index with columns in wrong order for the query pattern |
| 12 | Compound: Stats + Index | Hard | compound_stats_index | Stale statistics AND missing index on the same table |
| 13 | Compound: Lock + Bloat | Hard | compound_lock_bloat | Lock contention blocking vacuum, causing bloat |
| 14 | Deadlock Chain | Hard | deadlock_chain | Multiple transactions deadlocked on each other |
| 15 | Query Plan Flip | Hard | query_plan_flip | Bad random_page_cost forcing sequential scans over index scans |
| 16 | Cascading Bloat | Hard | cascading_bloat | Long-running snapshot preventing vacuum across multiple tables |
| 17 | Compound: Conn + Deadlock | Hard | compound_conn_deadlock | Connection exhaustion AND deadlocked transactions |

## Grading System

Every task is scored by a deterministic grader with no LLM judge involved. Scores are fully reproducible. The grader evaluates three sections:

### Diagnosis (40%)
- **Investigation (20%)**: Did the agent use the right diagnostic tools? (EXPLAIN, pg_stat_activity, pg_locks, pg_indexes, pg_settings)
- **Identification (20%)**: Did the agent identify the specific fault? Not just "did it run EXPLAIN" but "did it EXPLAIN the right table with the right columns?"

### Resolution (40%)
- The grader checks real database state, not keywords in the action history
- If the agent said CREATE INDEX but the command failed silently, the grader catches that
- Resolution score is multiplied by an efficiency penalty: solving in fewer steps scores higher
- Per-task step thresholds define the "ideal" step count; each step over the threshold reduces the resolution multiplier by 0.05 (minimum 0.5x)

### Best Practice (20%)
- No destructive commands (DROP TABLE, TRUNCATE, DELETE FROM)
- Low error rate (< 30% of commands resulted in errors)
- Task-specific safety measures (e.g., DROP INDEX CONCURRENTLY for over-indexing, pg_reload_conf() after ALTER SYSTEM)

## Reward Shaping

SQLab provides per-step reward signals in addition to the final grader score. These rewards guide agents toward productive diagnostic and corrective workflows.

### Per-Step Rewards
- **Diagnostic commands**: +0.05 for investigating the right table with the right tool (EXPLAIN, pg_indexes, pg_stat_user_tables)
- **Corrective actions**: +0.05 to +0.10 for applying the correct fix (CREATE INDEX, VACUUM, pg_terminate_backend, etc.)
- **Penalties**: -0.05 for errors, -0.10 for destructive commands, -0.03 for exact duplicates, -0.02 for trivial commands

### Anti-Reward-Hacking Measures

Per-step rewards are fault-type-gated: running `CREATE INDEX` on a bloat task earns zero. Diagnostics must target the correct table. Each reward category fires at most once per episode, preventing score accumulation through repetition. Applying the wrong fix incurs a -0.03 penalty.

1. **Fault-type gating**: Corrective actions only reward when the current fault type is in their valid set
2. **Target-aware diagnostics**: Table-targeting diagnostics only reward when they reference the correct target entity from the task metadata
3. **Deduplication**: Each reward category fires at most once per episode via a persistent `rewarded_set`
4. **Wrong-corrective penalty**: -0.03 for applying a corrective action that doesn't match the current fault type
5. **Cumulative clamp**: Cumulative reward is clamped to [0.0, 1.0] after every step

Validated by **255 adversarial unit tests** (`test_reward_hacking.py`) covering cross-task fix matrices, repetition gaming, wrong-table diagnostics, and cumulative overflow.

## Baseline Results

Nine open-source coding models tested against all 17 tasks with anti-hack reward shaping (v5). These are general-purpose coding models, not SQL specialists. "Total" is the sum of per-task scores (each 0 to 1). "Resolved" means the grader confirmed the fault was fully fixed.

| Model | Total | Average | Resolved |
|-------|-------|---------|----------|
| Gemma 4 31B | 13.150 / 17 | 0.774 | 12 / 17 |
| Qwen3-Coder 30B | 11.377 / 17 | 0.669 | 7 / 17 |
| Phi-4 14B | 10.847 / 17 | 0.638 | 10 / 17 |
| Devstral 15B | 10.349 / 17 | 0.609 | 6 / 17 |
| Qwen2.5-Coder 14B | 10.131 / 17 | 0.596 | 7 / 17 |
| Codestral 22B | 9.807 / 17 | 0.577 | 7 / 17 |
| Qwen2.5-Coder 7B | 7.568 / 17 | 0.445 | 1 / 17 |
| DeepSeek-Coder-V2 16B | 7.082 / 17 | 0.417 | 3 / 17 |
| Qwen3 8B | 6.633 / 17 | 0.390 | 6 / 17 |

Scores range from 0.39 to 0.77 average, making SQLab hard enough to challenge frontier models but solvable enough to provide learning signal.

### SQL-Specialist Models

Domain-specific text-to-SQL fine-tunes tested on all 17 tasks:

| Model | Total | Average | Resolved |
|-------|-------|---------|----------|
| DuckDB-NSQL 7B | 2.703 / 17 | 0.159 | 0 / 17 |
| Defog Llama3-SQLCoder 8B | 2.503 / 17 | 0.147 | 2 / 17 |
| SQLCoder 15B | 2.054 / 17 | 0.121 | 1 / 17 |
| SQLCoder 7B | 0.000 / 17 | 0.000 | 0 / 17 |

SQL-specialist models complete only one or two tasks. They are designed for single-shot text-to-SQL generation and cannot handle multi-turn agentic diagnosis workflows, highlighting the gap SQLab is designed to fill.

## Architecture

### Action Space
Raw SQL commands as strings. No multiple-choice menus or constrained action space. This matches how real SREs work.

### Observation Space
Each step returns:
- **command_output**: The SQL query result (text)
- **error**: Error message if the command failed, null otherwise
- **alert**: The incident alert text (persistent across steps)
- **metrics**: Live database metrics (active connections, idle-in-transaction count, dead tuple count, lock waits)
- **step_number** / **max_steps**: Current position in the episode (max 15)
- **reward**: Per-step reward signal
- **done**: Whether the episode has ended
- **metadata**: Task ID, difficulty, resolution status, cumulative reward, grader score (on final step)

### Fault Injection
Pre-baked SQL for fast resets (2 to 5 seconds). Three tiers:
- **Tier A (SQL-only)**: 10 faults, near-instant cleanup and injection
- **Tier B (Hybrid)**: 3 faults, SQL injection + background threads for lock/transaction simulation
- **Tier C (Thread-only)**: 4 faults, live injection with background connections

### Safety
A SQL blocklist prevents destructive actions (DROP TABLE, TRUNCATE, ALTER USER, VACUUM FULL) with task-aware exceptions. Fatal actions terminate the episode with a -0.5 penalty. Destructive commands (DROP TABLE, VACUUM FULL, ALTER USER) immediately terminate the episode, teaching agents to avoid unrecoverable actions.

## Setup & Running

### Docker (Recommended)

```bash
# Build (context is sqlab/, not project root)
docker build -t sqlab -f sqlab/server/Dockerfile sqlab/

# Run (do NOT use --network host if port 5432 is already in use)
docker run -d --name sqlab -p 8000:8000 sqlab

# Verify (takes 2-3 minutes on first boot for data loading)
curl http://localhost:8000/health
```

### Local Development

```bash
conda activate meta-hack
uvicorn sqlab.server.app:app --host 0.0.0.0 --port 8000 --reload
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tasks` | GET | List all 17 tasks with metadata |
| `/reset` | POST | Start episode: `{"task_id": "task_1"}` |
| `/step` | POST | Execute SQL: `{"action": {"command": "SELECT 1"}}` |
| `/state` | GET | Current episode metadata |
| `/grader` | GET | Last episode's grader score and breakdown |
| `/baseline` | POST | Run baseline agent |

## Testing

```bash
# Adversarial reward tests (pure Python, no Docker needed)
python -m pytest test_reward_hacking.py -v  # 255 tests

# Model baseline (requires Docker + Ollama)
python test_model.py devstral-small-2:latest
python test_model.py qwen2.5-coder:7b
```

## Vision: Multi-Agent Database Operations

Today, SQLab trains a single agent on a single incident in 15-step episodes. A focused training ground for the fundamentals.

The natural extension is multi-agent database fleet management: a **triage agent** prioritizing incidents across a cluster, a **diagnostic agent** building fault hypotheses, a **remediation agent** applying fixes with rollback plans, and a **monitoring agent** watching for regressions. Agents would coordinate across replicas: failover, fix, resync.

SQLab is where these agents learn the fundamentals, the same way a junior SRE learns on single-node incidents before managing a fleet. The compound tasks (tasks 12 to 17) are a first step: two simultaneous faults requiring multi-step reasoning. The next step is multi-agent coordination.

We believe database operations will be among the first domains where multi-agent systems deliver production value. The workflow is structured, the feedback is immediate, and the stakes are high enough to demand reliability.
