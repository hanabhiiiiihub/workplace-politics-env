# server/models.py
from typing import List, Optional, Dict
from pydantic import BaseModel
import sys, os
sys.path.insert(0, os.path.abspath("OpenEnv/src/openenv"))
from core.env_server import Action, Observation, State


class PoliticsAction(Action):
    """One move the AI agent can take."""
    action_type: str          # "schedule_meeting", "share_info", "request_support",
                              # "observe", "make_proposal", "build_alliance"
    target: str               # who: "boss", "rival", "friend", "gatekeeper", "fence_sitter"
    content: Optional[str] = None   # what to say/share (for share_info)
    info_level: Optional[str] = None  # "full", "partial", "vague"


class CharacterState(BaseModel):
    name: str
    support_level: float      # -1.0 (opposed) to 1.0 (fully supportive)
    trust_level: float        # 0.0 to 1.0
    informed: bool            # does this character know about the proposal?
    suspicious: bool          # are they suspicious of the agent?


class PoliticsObservation(Observation):
    """What the AI sees after each action — always partial information."""
    turn: int
    max_turns: int
    task_level: int           # 1, 2, or 3
    observable_events: List[str]   # text clues like "Rival requested meeting with Boss"
    your_reputation: float    # 0.0 to 1.0
    political_capital: float  # 0.0 to 1.0
    proposal_submitted: bool
    proposal_outcome: Optional[str] = None   # None / "approved" / "rejected"
    known_support: Dict[str, str]   # what you think each person's stance is
    reward: float
    done: bool


class PoliticsState(State):
    """Full hidden state — only the environment sees this."""
    episode_id: str
    turn: int
    task_level: int
    characters: Dict[str, CharacterState]
    rival_moves: List[str]
    agent_actions_history: List[str]
    political_capital: float
    reputation: float
    proposal_submitted: bool
    coalition_strength: float