# server/environment.py
import random
import uuid
from typing import Optional, Dict, List
from server.models import (
    PoliticsAction, PoliticsObservation, PoliticsState, CharacterState
)
import sys, os
sys.path.insert(0, os.path.abspath("OpenEnv/src/openenv"))
from core.env_server import Environment
from core.env_server.interfaces import Transform


CHARACTERS = ["boss", "rival", "friend", "gatekeeper", "fence_sitter"]

# Clue templates for observable events
RIVAL_CLUES = [
    "The Rival requested a private meeting with the Boss — topic unknown.",
    "You overheard the Rival mention 'budget concerns' to the Gatekeeper.",
    "The Rival forwarded an email chain to the Fence-Sitter — contents unclear.",
    "The Rival seemed unusually cheerful after talking to the Boss.",
    "A colleague mentioned the Rival has been 'asking around' about your proposal.",
]

FRIEND_CLUES = [
    "Your Friend seemed hesitant when you brought up the proposal.",
    "Your Friend forwarded your draft to someone unexpected.",
    "Your Friend mentioned they had 'a chat' with the Rival.",
]

FENCE_SITTER_CLUES = [
    "The Fence-Sitter seemed distracted in today's all-hands meeting.",
    "The Fence-Sitter asked a colleague what they thought of project management changes.",
    "The Fence-Sitter was spotted having coffee with the Rival.",
]


class PoliticsEnv(Environment):

    def __init__(self, task_level: int = 1, transform=None, rubric=None):
        super().__init__(transform=transform, rubric=rubric)
        self.task_level = task_level
        self._episode_id = str(uuid.uuid4())[:8]
        self._characters: Dict[str, CharacterState] = {}
        self._rival_moves: List[str] = []
        self._agent_history: List[str] = []
        self._turn = 0
        self._political_capital = 1.0
        self._reputation = 0.7
        self._proposal_submitted = False
        self._proposal_outcome = None
        self._max_turns = {1: 15, 2: 12, 3: 10}[task_level]
        self._pending_events: List[str] = []
        self._reset_characters()

    def _reset_characters(self):
        """Initialize characters with hidden attitudes per level."""
        self._characters = {
            "boss": CharacterState(
                name="Boss", support_level=0.0, trust_level=0.5,
                informed=False, suspicious=False
            ),
            "rival": CharacterState(
                name="Rival",
                support_level=-0.8 if self.task_level > 1 else -0.3,
                trust_level=0.1, informed=False, suspicious=False
            ),
            "friend": CharacterState(
                name="Friend", support_level=0.6, trust_level=0.8,
                informed=False, suspicious=False
            ),
            "gatekeeper": CharacterState(
                name="Gatekeeper", support_level=0.0, trust_level=0.4,
                informed=False, suspicious=False
            ),
            "fence_sitter": CharacterState(
                name="Fence-Sitter", support_level=0.0, trust_level=0.5,
                informed=False, suspicious=False
            ),
        }

    def reset(self) -> PoliticsObservation:
        self._episode_id = str(uuid.uuid4())[:8]
        self._turn = 0
        self._political_capital = 1.0
        self._reputation = 0.7
        self._proposal_submitted = False
        self._proposal_outcome = None
        self._rival_moves = []
        self._agent_history = []
        self._pending_events = []
        self._reset_characters()

        return PoliticsObservation(
            turn=0,
            max_turns=self._max_turns,
            task_level=self.task_level,
            observable_events=[
                "You've just joined the company. The team uses outdated tools.",
                "You have an idea: switch to a better project management platform.",
                "You need enough support before making a formal proposal to the Boss.",
            ],
            your_reputation=self._reputation,
            political_capital=self._political_capital,
            proposal_submitted=False,
            proposal_outcome=None,
            known_support={c: "unknown" for c in CHARACTERS},
            reward=0.0,
            done=False,
        )

    def step(self, action: PoliticsAction) -> PoliticsObservation:
        self._turn += 1
        reward = -0.05   # small cost per turn (encourages efficiency)
        events: List[str] = list(self._pending_events)
        self._pending_events = []

        target = action.target.lower()
        atype = action.action_type.lower()

        # --- Process agent action ---
        if atype == "schedule_meeting":
            reward += self._handle_meeting(target, events)

        elif atype == "share_info":
            reward += self._handle_share_info(target, action.info_level or "partial", events)

        elif atype == "request_support":
            reward += self._handle_request_support(target, events)

        elif atype == "observe":
            reward += self._handle_observe(target, events)

        elif atype == "build_alliance":
            reward += self._handle_build_alliance(target, events)

        elif atype == "make_proposal":
            reward += self._handle_proposal(events)

        else:
            events.append("That action wasn't recognized. You lost some momentum.")
            reward -= 0.1

        self._agent_history.append(f"Turn {self._turn}: {atype} -> {target}")

        # --- Rival acts (levels 2+) ---
        if self.task_level >= 2 and not self._proposal_submitted:
            self._rival_acts(events)

        # --- Check terminal conditions ---
        done = False
        if self._proposal_submitted:
            done = True
        elif self._turn >= self._max_turns:
            done = True
            events.append("You ran out of time. The window for change has closed.")
            reward -= 0.3
        elif self._political_capital <= 0.0:
            done = True
            events.append("You've burned all your political capital. No one takes you seriously.")
            reward -= 0.5

        # --- Build known_support (agent's best guess) ---
        known_support = {}
        for name, char in self._characters.items():
            if char.informed:
                if char.support_level > 0.4:
                    known_support[name] = "supportive"
                elif char.support_level < -0.3:
                    known_support[name] = "opposed"
                else:
                    known_support[name] = "neutral"
            else:
                known_support[name] = "unknown"

        return PoliticsObservation(
            turn=self._turn,
            max_turns=self._max_turns,
            task_level=self.task_level,
            observable_events=events,
            your_reputation=round(self._reputation, 2),
            political_capital=round(self._political_capital, 2),
            proposal_submitted=self._proposal_submitted,
            proposal_outcome=self._proposal_outcome,
            known_support=known_support,
            reward=round(reward, 3),
            done=done,
        )

    # ------------------------------------------------------------------ #
    #  Action handlers                                                      #
    # ------------------------------------------------------------------ #

    def _handle_meeting(self, target: str, events: List[str]) -> float:
        reward = 0.0
        char = self._characters.get(target)
        if not char:
            events.append(f"You couldn't find {target}.")
            return -0.05

        if target == "gatekeeper":
            char.trust_level = min(1.0, char.trust_level + 0.15)
            char.support_level = min(1.0, char.support_level + 0.1)
            events.append("The Gatekeeper appreciated being consulted early. Trust increased.")
            reward += 0.1
        elif target == "boss":
            if self._characters["gatekeeper"].trust_level < 0.6:
                events.append("The Gatekeeper blocked your meeting. You need to build rapport first.")
                self._political_capital -= 0.1
                reward -= 0.1
            else:
                char.trust_level = min(1.0, char.trust_level + 0.1)
                events.append("The Boss gave you 15 minutes. They seemed cautious but open.")
                reward += 0.15
        elif target == "friend":
            char.trust_level = min(1.0, char.trust_level + 0.1)
            events.append("Your Friend is enthusiastic. Promises full support — for now.")
            reward += 0.1
        elif target == "fence_sitter":
            char.trust_level = min(1.0, char.trust_level + 0.08)
            events.append("The Fence-Sitter listened politely. Seemed to be waiting to see who else backs you.")
            reward += 0.05
        elif target == "rival":
            # Meeting rival: risky — might learn their plan but reveals yours
            if random.random() < 0.4:
                events.append("The Rival was cordial. You detected they're already aware of your idea.")
                self._characters["rival"].informed = True
                reward -= 0.05
            else:
                events.append("The Rival declined your meeting request. Ominous.")
                reward -= 0.1

        self._political_capital -= 0.05
        return reward

    def _handle_share_info(self, target: str, level: str, events: List[str]) -> float:
        reward = 0.0
        char = self._characters.get(target)
        if not char:
            return -0.05

        char.informed = True

        if level == "full":
            # Risk: rival might find out faster
            if target == "friend" and random.random() < 0.3:
                events.append("Your Friend accidentally mentioned your plan to the Rival. Confidentiality breach!")
                self._characters["rival"].informed = True
                reward -= 0.2
            else:
                char.support_level = min(1.0, char.support_level + 0.2)
                events.append(f"You shared full details with {char.name}. They now understand the vision.")
                reward += 0.15

        elif level == "partial":
            char.support_level = min(1.0, char.support_level + 0.1)
            events.append(f"You shared a high-level overview with {char.name}. They're intrigued.")
            reward += 0.08

        elif level == "vague":
            char.support_level = min(1.0, char.support_level + 0.03)
            events.append(f"You were vague with {char.name}. They're not sure what you're proposing.")
            reward += 0.02

        # Inconsistency check: if boss hears a different version
        if target == "boss" and "boss" in [a.split("->")[-1].strip() for a in self._agent_history]:
            events.append("The Boss mentioned something felt 'inconsistent' with what they heard before.")
            self._reputation -= 0.15
            reward -= 0.2

        self._political_capital -= 0.04
        return reward

    def _handle_request_support(self, target: str, events: List[str]) -> float:
        reward = 0.0
        char = self._characters.get(target)
        if not char:
            return -0.05

        if not char.informed:
            events.append(f"{char.name} doesn't know about your proposal yet. Build context first.")
            self._political_capital -= 0.05
            return -0.1

        if char.support_level > 0.5 and char.trust_level > 0.5:
            char.support_level = min(1.0, char.support_level + 0.1)
            events.append(f"{char.name} agreed to actively support your proposal. Solid ally secured.")
            reward += 0.2
        elif char.support_level > 0.0:
            events.append(f"{char.name} said they'd 'think about it.' Not a no.")
            reward += 0.05
        else:
            events.append(f"{char.name} pushed back. Pressing them hurt your reputation slightly.")
            self._reputation -= 0.05
            reward -= 0.1

        self._political_capital -= 0.06
        return reward

    def _handle_observe(self, target: str, events: List[str]) -> float:
        """Low-cost action: watch and gather clues."""
        if target == "rival" and self.task_level >= 2:
            events.append(random.choice(RIVAL_CLUES))
        elif target == "friend":
            events.append(random.choice(FRIEND_CLUES))
        elif target == "fence_sitter":
            events.append(random.choice(FENCE_SITTER_CLUES))
        elif target == "boss":
            events.append("The Boss seems busy. You note they value clear ROI data above all else.")
        elif target == "gatekeeper":
            events.append("The Gatekeeper keeps a tidy calendar. They prefer 48h advance notice for meetings.")
        return 0.02   # tiny reward for information gathering

    def _handle_build_alliance(self, target: str, events: List[str]) -> float:
        reward = 0.0
        char = self._characters.get(target)
        if not char:
            return -0.05

        if char.support_level > 0.3 and char.trust_level > 0.4:
            char.support_level = min(1.0, char.support_level + 0.15)
            self._reputation = min(1.0, self._reputation + 0.05)
            events.append(f"Alliance with {char.name} strengthened. Your coalition is growing.")
            reward += 0.15
        else:
            events.append(f"{char.name} is not ready to commit. You need more trust first.")
            reward -= 0.05

        self._political_capital -= 0.08
        return reward

    def _handle_proposal(self, events: List[str]) -> float:
        """Make the formal proposal to the Boss. This ends the episode."""
        self._proposal_submitted = True
        reward = 0.0

        coalition = self._coalition_strength()
        boss = self._characters["boss"]
        gatekeeper = self._characters["gatekeeper"]

        if not boss.informed:
            events.append("The Boss was blindsided. They hate surprises. Proposal rejected.")
            self._proposal_outcome = "rejected"
            reward -= 0.3
            return reward

        if gatekeeper.trust_level < 0.5:
            events.append("The Gatekeeper undermined you at the last moment. Proposal weakened.")
            coalition -= 0.2

        if coalition >= 0.7 and boss.trust_level >= 0.5:
            events.append("APPROVED! Strong coalition + boss trust = success. Team cheers.")
            self._proposal_outcome = "approved"
            reward += 1.0
            # Speed bonus
            turns_left = self._max_turns - self._turn
            reward += turns_left * 0.05
        elif coalition >= 0.4:
            events.append("Partially approved. The Boss wants a pilot first. Partial win.")
            self._proposal_outcome = "partial"
            reward += 0.4
        else:
            events.append("Proposal failed. Not enough support. Try building more coalition first.")
            self._proposal_outcome = "rejected"
            reward -= 0.1

        # Penalties
        if self._characters["rival"].informed and self.task_level >= 2:
            rival_damage = 0.3 * (1 - self._characters["rival"].trust_level)
            reward -= rival_damage
            events.append(f"The Rival had been working against you. Cost: -{rival_damage:.2f}")

        return reward

    def _coalition_strength(self) -> float:
        """Calculate how strong the agent's coalition is."""
        supporters = [
            c for name, c in self._characters.items()
            if name != "rival" and c.informed and c.support_level > 0.3
        ]
        if not supporters:
            return 0.0
        weights = {"boss": 0.5, "gatekeeper": 0.2, "friend": 0.15, "fence_sitter": 0.15}
        total = sum(
            weights.get(name, 0.1) * char.support_level
            for name, char in self._characters.items()
            if name != "rival" and char.informed and char.support_level > 0.3
        )
        return min(1.0, total)

    def _rival_acts(self, events: List[str]):
        """Rival takes hidden actions every 2-3 turns."""
        if self._turn % 3 == 0 and self._characters["rival"].informed:
            target = random.choice(["friend", "fence_sitter", "boss"])
            char = self._characters[target]
            damage = random.uniform(0.05, 0.15)
            char.support_level = max(-1.0, char.support_level - damage)
            events.append(random.choice(RIVAL_CLUES))

    def get_state(self) -> PoliticsState:
        return PoliticsState(
            episode_id=self._episode_id,
            turn=self._turn,
            task_level=self.task_level,
            characters=self._characters,
            rival_moves=self._rival_moves,
            agent_actions_history=self._agent_history,
            political_capital=self._political_capital,
            reputation=self._reputation,
            proposal_submitted=self._proposal_submitted,
            coalition_strength=self._coalition_strength(),
        )

    @property
    def state(self) -> PoliticsState:
        return self.get_state()