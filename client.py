"""
SQLab — Environment client.

Wraps WebSocket communication with the environment server.
Provides typed step/reset/state methods for the agent.
"""

from typing import Dict, Any
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from sqlab.models import DBSreAction, DBSreObservation, DBSreState


class DBSreEnv(EnvClient[DBSreAction, DBSreObservation, DBSreState]):
    """Client for the SQLab environment."""

    def _step_payload(self, action: DBSreAction) -> Dict[str, Any]:
        """Convert an Action to the JSON payload expected by the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DBSreObservation]:
        """Parse server response into a StepResult with typed observation."""
        obs_data = payload.get("observation", {})
        obs = DBSreObservation(
            **obs_data,
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DBSreState:
        """Parse server state response into typed State object."""
        return DBSreState(**payload)
