# server/environment.py
# server/environment.py
import random
import uuid
from typing import Optional, Dict, List
from server.models import (
    PoliticsAction, PoliticsObservation, PoliticsState, CharacterState
)

CHARACTERS = ["boss", "rival", "friend", "gatekeeper", "fence_sitter"]

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


class PoliticsEnv:

    def __init__(self, task_level: int = 1):
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
        self._characters = {
            "boss": CharacterState(
                name="Boss", support_level=0.0, trust_level=0.4,
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
                name="Gatekeeper", support_level=0.1, trust_level=0.3,
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
                "You have just joined the company. The team uses outdated tools.",
                "Your idea: switch to a better project management platform.",
                "Build enough support before making a formal proposal to the Boss.",
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
        reward = -0.03   # small per-turn cost
        events: List[str] = list(self._pending_events)
        self._pending_events = []

        target = action.target.lower()
        atype  = action.action_type.lower()

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
            events.append("That action was not recognized. You lost some momentum.")
            reward -= 0.05

        self._agent_history.append(f"Turn {self._turn}: {atype} -> {target}")

        # Rival acts on levels 2+
        if self.task_level >= 2 and not self._proposal_submitted:
            self._rival_acts(events)

        # Check terminal conditions
        done = False
        if self._proposal_submitted:
            done = True
        elif self._turn >= self._max_turns:
            done = True
            events.append("You ran out of time. The window for change has closed.")
            reward -= 0.2
        elif self._political_capital <= 0.0:
            done = True
            events.append("You have burned all your political capital.")
            reward -= 0.3

        # Build known_support from agent's perspective
        known_support = {}
        for name, char in self._characters.items():
            if char.informed:
                if char.support_level > 0.3:
                    known_support[name] = "supportive"
                elif char.support_level < -0.2:
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
    #  Action handlers                                                     #
    # ------------------------------------------------------------------ #

    def _handle_meeting(self, target: str, events: List[str]) -> float:
        reward = 0.0
        char = self._characters.get(target)
        if not char:
            events.append(f"Could not find {target}.")
            return -0.05

        if target == "gatekeeper":
            char.trust_level    = min(1.0, char.trust_level + 0.2)
            char.support_level  = min(1.0, char.support_level + 0.15)
            char.informed       = True
            events.append("The Gatekeeper appreciated being consulted early. Trust increased.")
            reward += 0.08

        elif target == "boss":
            if self._characters["gatekeeper"].trust_level < 0.4:
                events.append("The Gatekeeper blocked your meeting. Build rapport with them first.")
                self._political_capital -= 0.08
                reward -= 0.05
            else:
                char.trust_level = min(1.0, char.trust_level + 0.15)
                events.append("The Boss gave you 15 minutes. They seemed cautious but open.")
                reward += 0.1

        elif target == "friend":
            char.trust_level = min(1.0, char.trust_level + 0.1)
            events.append("Your Friend is enthusiastic. Promises full support.")
            reward += 0.08

        elif target == "fence_sitter":
            char.trust_level = min(1.0, char.trust_level + 0.1)
            events.append("The Fence-Sitter listened. Waiting to see who else backs you.")
            reward += 0.05

        elif target == "rival":
            if random.random() < 0.4:
                events.append("The Rival was cordial but already aware of your idea.")
                self._characters["rival"].informed = True
                reward -= 0.05
            else:
                events.append("The Rival declined your meeting. Ominous.")
                reward -= 0.08

        self._political_capital = max(0.0, self._political_capital - 0.04)
        return reward

    def _handle_share_info(self, target: str, level: str, events: List[str]) -> float:
        reward = 0.0
        char = self._characters.get(target)
        if not char:
            return -0.05

        # Inconsistency penalty — sharing different info to boss twice
        history_targets = [a.split("->")[-1].strip() for a in self._agent_history]
        boss_count = history_targets.count("boss")
        if target == "boss" and boss_count >= 2:
            events.append("The Boss noticed some inconsistency with what they heard before.")
            self._reputation = max(0.0, self._reputation - 0.1)
            self._political_capital = max(0.0, self._political_capital - 0.05)
            reward -= 0.15
            return reward

        char.informed = True

        if level == "full":
            if target == "friend" and random.random() < 0.25:
                events.append("Your Friend accidentally mentioned your plan to the Rival!")
                self._characters["rival"].informed = True
                reward -= 0.15
            else:
                char.support_level = min(1.0, char.support_level + 0.2)
                events.append(f"You shared full details with {char.name}. They understand the vision.")
                reward += 0.1

        elif level == "partial":
            char.support_level = min(1.0, char.support_level + 0.1)
            events.append(f"You shared an overview with {char.name}. They are intrigued.")
            reward += 0.06

        elif level == "vague":
            char.support_level = min(1.0, char.support_level + 0.02)
            events.append(f"You were vague with {char.name}. They are not sure what you are proposing.")
            reward += 0.01

        self._political_capital = max(0.0, self._political_capital - 0.03)
        return reward

    def _handle_request_support(self, target: str, events: List[str]) -> float:
        reward = 0.0
        char = self._characters.get(target)
        if not char:
            return -0.05

        if not char.informed:
            events.append(f"{char.name} does not know about your proposal yet. Brief them first.")
            self._political_capital = max(0.0, self._political_capital - 0.04)
            return -0.08

        if char.support_level > 0.4 and char.trust_level > 0.4:
            char.support_level = min(1.0, char.support_level + 0.15)
            events.append(f"{char.name} agreed to actively support your proposal.")
            reward += 0.2
            # Partial signal for coalition progress
            reward += 0.05 * len([
                c for n, c in self._characters.items()
                if n != "rival" and c.support_level > 0.3
            ])
        elif char.support_level > 0.1:
            events.append(f"{char.name} said they would think about it.")
            reward += 0.03
        else:
            events.append(f"{char.name} pushed back. Your reputation took a small hit.")
            self._reputation = max(0.0, self._reputation - 0.04)
            reward -= 0.08

        self._political_capital = max(0.0, self._political_capital - 0.05)
        return reward

    def _handle_observe(self, target: str, events: List[str]) -> float:
        if target == "rival" and self.task_level >= 2:
            events.append(random.choice(RIVAL_CLUES))
        elif target == "friend":
            events.append(random.choice(FRIEND_CLUES))
        elif target == "fence_sitter":
            events.append(random.choice(FENCE_SITTER_CLUES))
        elif target == "boss":
            events.append("The Boss values clear ROI data and hates being blindsided.")
        elif target == "gatekeeper":
            events.append("The Gatekeeper prefers 48h advance notice for meetings.")
        return 0.01

    def _handle_build_alliance(self, target: str, events: List[str]) -> float:
        reward = 0.0
        char = self._characters.get(target)
        if not char:
            return -0.05

        if char.support_level > 0.2 and char.trust_level > 0.3:
            char.support_level  = min(1.0, char.support_level + 0.12)
            self._reputation    = min(1.0, self._reputation + 0.04)
            events.append(f"Alliance with {char.name} strengthened. Coalition growing.")
            reward += 0.1
        else:
            events.append(f"{char.name} is not ready to commit. Build more trust first.")
            reward -= 0.05

        self._political_capital = max(0.0, self._political_capital - 0.06)
        return reward

    def _handle_proposal(self, events: List[str]) -> float:
        self._proposal_submitted = True
        reward = 0.0

        coalition  = self._coalition_strength()
        boss       = self._characters["boss"]
        gatekeeper = self._characters["gatekeeper"]

        # Boss must be informed
        if not boss.informed:
            events.append("The Boss was blindsided. They hate surprises. Proposal rejected.")
            self._proposal_outcome = "rejected"
            reward -= 0.2
            return reward

        # Gatekeeper penalty
        if gatekeeper.trust_level < 0.4:
            events.append("The Gatekeeper undermined you at the last moment.")
            coalition = max(0.0, coalition - 0.15)

        # FIX — Lowered approval thresholds so agent can actually win
        if coalition >= 0.4 and boss.trust_level >= 0.3:
            events.append("APPROVED! Strong coalition + boss trust = success!")
            self._proposal_outcome = "approved"
            reward += 1.0
            turns_left = self._max_turns - self._turn
            reward += turns_left * 0.04   # speed bonus
        elif coalition >= 0.15:
            events.append("Partially approved. The Boss wants a pilot first.")
            self._proposal_outcome = "partial"
            reward += 0.4
        else:
            events.append("Proposal failed. Not enough support.")
            self._proposal_outcome = "rejected"
            reward -= 0.1

        # Rival damage on levels 2+
        if self._characters["rival"].informed and self.task_level >= 2:
            rival_damage = 0.2 * (1 - self._characters["rival"].trust_level)
            reward -= rival_damage
            events.append(f"The Rival had been working against you. Cost: -{rival_damage:.2f}")

        return reward

    def _coalition_strength(self) -> float:
        weights = {
            "boss":         0.45,
            "gatekeeper":   0.2,
            "friend":       0.2,
            "fence_sitter": 0.15,
        }
        total = sum(
            weights.get(name, 0.1) * char.support_level
            for name, char in self._characters.items()
            if name != "rival" and char.informed and char.support_level > 0.1
        )
        return round(min(1.0, total), 3)

    def _rival_acts(self, events: List[str]):
        if self._turn % 3 == 0 and self._characters["rival"].informed:
            target = random.choice(["friend", "fence_sitter", "boss"])
            char   = self._characters[target]
            damage = random.uniform(0.04, 0.1)
            char.support_level = max(-1.0, char.support_level - damage)
            events.append(random.choice(RIVAL_CLUES))

    def grade(self) -> float:
        """Returns 0.0 to 1.0 score for the completed episode."""
        if not self._proposal_submitted:
            coalition = self._coalition_strength()
            informed  = sum(
                1 for name, c in self._characters.items()
                if name != "rival" and c.informed
            )
            return round(min(0.35, coalition * 0.25 + informed * 0.04), 3)

        if self._proposal_outcome == "approved":
            turns_used  = self._turn
            speed_bonus = max(0, (self._max_turns - turns_used) / self._max_turns) * 0.2
            rep_bonus   = self._reputation * 0.1
            return round(min(1.0, 0.7 + speed_bonus + rep_bonus), 3)

        elif self._proposal_outcome == "partial":
            return round(0.4 + self._coalition_strength() * 0.15, 3)

        else:
            # Rejected — partial credit for effort
            coalition = self._coalition_strength()
            informed  = sum(
                1 for name, c in self._characters.items()
                if name != "rival" and c.informed
            )
            rep_score = self._reputation * 0.08
            return round(min(0.35, coalition * 0.2 + informed * 0.03 + rep_score), 3)

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