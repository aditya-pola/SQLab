"""
SQLab — Pydantic models for Action, Observation, and State.

These define the typed interface between the agent and the environment.

The action space is intentionally open-ended: agents submit arbitrary SQL
strings, mirroring how a real SRE interacts with a production PostgreSQL
instance via psql.  This contrasts with discrete-action environments — the
agent must compose valid SQL from scratch, making the problem closer to
real incident response than to a multiple-choice quiz.

The environment ships 17 fault-injection tasks across three difficulty tiers
(easy / medium / hard), each scored by a deterministic three-section grader
(diagnosis 30 % | resolution 50 % | best-practice 20 %).  Observations
surface the same signals a human SRE would see: an alert banner, live
health metrics, and verbatim psql-formatted output.

Why this matters for the RL/agent community: database incident response is
a high-value, under-served domain — no existing RL benchmark exercises
real SQL against a live database with production-grade fault injection.
SQLab fills that gap with a reproducible, Docker-containerised environment
that any researcher can spin up in minutes for agent evaluation or GRPO
fine-tuning.
"""

from typing import Optional, Dict, Any
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class DBSreAction(Action):
    """Agent submits a SQL command to diagnose or fix a database issue.

    The unbounded string action space is a deliberate design choice: frontier
    models like GPT-4o and Qwen-2.5 can generate syntactically valid SQL, so
    restricting them to a dropdown of pre-authored queries would trivialise
    the hard tasks and remove the compositional reasoning challenge.
    """
    # Open action space: any syntactically valid PostgreSQL command is accepted,
    # from SELECT on system catalogs to DDL fixes like CREATE INDEX or VACUUM.
    # This matches real SRE workflow — no artificial action discretisation.
    command: str = Field(
        ...,
        min_length=1,
        description="SQL command to execute against the PostgreSQL database"
    )


class DBSreObservation(Observation):
    """What the agent sees after each action.

    Inherits from Observation which provides:
        - done: bool (whether episode has terminated)
        - reward: Optional[float] (reward signal from last action)
        - metadata: Dict[str, Any]
    """
    # Formatted identically to psql terminal output so LLMs can leverage their
    # pre-training on PostgreSQL documentation and Stack Overflow examples.
    command_output: str = Field(
        default="",
        description="Raw output from the SQL command execution"
    )
    # SQL errors are surfaced verbatim so agents can learn from PostgreSQL's own
    # error codes — a skill that transfers directly to real-world SRE work.
    error: Optional[str] = Field(
        default=None,
        description="Error message if the SQL command failed"
    )
    # Persistent alert mirrors a PagerDuty/Opsgenie production alert — the agent
    # sees it on every step, just as a real SRE keeps the incident ticket open.
    alert: str = Field(
        default="",
        description="The incident alert text describing the database problem"
    )
    # Real-time health metrics matching production monitoring stacks (pganalyze,
    # pg_stat_monitor, Datadog).  Includes connection counts, lock counts, dead
    # tuple ratios, and cache hit rates — the same signals an SRE triages from.
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Database health metrics snapshot (connections, locks, dead tuples, etc.)"
    )
    # Step budget creates a tight episode horizon (15 steps), forcing efficient
    # triage.  Human SREs typically resolve incidents in 5-10 queries; 15 steps
    # gives enough room for exploration while penalising aimless wandering.
    step_number: int = Field(default=0, description="Current step in the episode")
    max_steps: int = Field(default=15, description="Maximum steps allowed per episode")


class DBSreState(State):
    """Episode metadata exposed to training harnesses and curriculum schedulers.

    Inherits from State which provides:
        - episode_id: Optional[str]
        - step_count: int

    cumulative_reward and grader_score are surfaced here so RL training loops
    (e.g. TRL's GRPO) can build curriculum strategies — for instance, promoting
    tasks where the agent consistently scores below 0.5 into more frequent
    sampling.
    """
    task_id: str = Field(default="", description="Identifier for the current task")
    task_name: str = Field(default="", description="Human-readable task name")
    # Three-tier difficulty enables curriculum learning: start on easy single-fault
    # tasks, graduate to hard compound faults (e.g. cascading_bloat) that require
    # multi-step remediation chains no frontier model has solved reliably.
    difficulty: str = Field(default="", description="Task difficulty: easy, medium, hard")
    fault_type: str = Field(default="", description="Type of fault injected")
    is_resolved: bool = Field(default=False, description="Whether the fault has been resolved")
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated this episode")
    grader_score: Optional[float] = Field(
        default=None,
        description="Final grader score (0.0-1.0), set at end of episode"
    )
