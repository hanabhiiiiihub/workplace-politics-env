"""
client.py — HTTP client for the Corporate Politics Simulator.

Mirrors the pattern in the reference GridClient but uses the politics
action/observation types.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath("OpenEnv/src/openenv"))

try:
    from core.http_env_client import HTTPEnvClient
    from core.types import StepResult
except ImportError:
    # Minimal shim so the file is importable outside the full OpenEnv install
    import httpx
    from pydantic import BaseModel

    class StepResult(BaseModel):
        observation: object
        reward: float
        done: bool

    class HTTPEnvClient:
        def __init__(self, base_url: str, prefix: str = ""):
            self._base = base_url.rstrip("/") + prefix

        async def reset(self):
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{self._base}/reset", json={}, timeout=30)
                r.raise_for_status()
                return r.json()

        async def step(self, action):
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{self._base}/step",
                                 json=action.model_dump(), timeout=30)
                r.raise_for_status()
                return r.json()

        async def state(self):
            async with httpx.AsyncClient() as c:
                r = await c.get(f"{self._base}/state", timeout=30)
                r.raise_for_status()
                return r.json()


from server.models import PoliticsAction, PoliticsObservation


class PoliticsClient(HTTPEnvClient):
    """HTTP client for one difficulty level of the Corporate Politics Simulator."""

    def __init__(self, base_url: str, level: int = 1):
        prefix = "" if level == 1 else f"/l{level}"
        super().__init__(base_url, prefix)

    def _step_payload(self, action: PoliticsAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult:
        obs = PoliticsObservation(**payload.get("observation", payload))
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )