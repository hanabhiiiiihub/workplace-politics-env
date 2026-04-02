# inference.py
import asyncio
import json
import os
import httpx
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
BASE_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a strategic AI agent playing a workplace politics simulation.
Your goal: get your proposal approved by the Boss by building a coalition.

Characters:
- boss: Says yes if enough people back you. Hates surprises. Needs gatekeeper access.
- rival: Actively works against you (level 2+). Don't let them find out too early.
- friend: Supports you but folds under pressure. Share info carefully.
- gatekeeper: Controls access to boss. Must be respected first.
- fence_sitter: Sides with whoever looks like they're winning.

Available actions (respond ONLY with valid JSON):
{
  "action_type": "schedule_meeting" | "share_info" | "request_support" | "observe" | "build_alliance" | "make_proposal",
  "target": "boss" | "rival" | "friend" | "gatekeeper" | "fence_sitter",
  "info_level": "full" | "partial" | "vague",   // only for share_info
  "content": "optional note"
}

Strategic tips:
1. Start by observing and meeting the gatekeeper (they block boss access)
2. Build friend trust before requesting support
3. Never tell the boss before you have coalition support
4. Watch for clues that the rival is moving against you
5. Make the proposal only when coalition strength is solid

Respond with ONLY the JSON action object. No explanation.
"""


async def get_ai_action(observation: dict, history: list) -> dict:
    """Ask Groq's LLM what action to take next."""
    
    obs_text = f"""
Turn {observation['turn']}/{observation['max_turns']}
Level: {observation['task_level']}
Political Capital: {observation['political_capital']}
Reputation: {observation['your_reputation']}
Known Support: {json.dumps(observation['known_support'], indent=2)}

Recent Events:
{chr(10).join('- ' + e for e in observation['observable_events'])}

Proposal submitted: {observation['proposal_submitted']}
"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history (last 6 turns to stay within context)
    for entry in history[-6:]:
        messages.append({"role": "user", "content": entry["obs"]})
        messages.append({"role": "assistant", "content": json.dumps(entry["action"])})
    
    messages.append({"role": "user", "content": obs_text})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3,
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()
    
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    return json.loads(raw.strip())


async def run_episode(task_level: int = 1) -> float:
    """Run one full episode and return the total score."""
    
    async with httpx.AsyncClient(timeout=30.0) as http:
        # Reset
        resp = await http.post(f"{BASE_URL}/reset", json={"task_level": task_level})
        obs = resp.json()
        
        print(f"\n{'='*60}")
        print(f"  WORKPLACE POLITICS — Level {task_level}")
        print(f"{'='*60}")
        print(f"Starting observation: {obs['observable_events']}")

        total_reward = 0.0
        history = []

        while not obs.get("done", False):
            # Get AI action
            action = await get_ai_action(obs, history)
            print(f"\nTurn {obs['turn']+1} | AI Action: {json.dumps(action)}")

            # Send to environment
            payload = {**action, "task_level": task_level}
            step_resp = await http.post(f"{BASE_URL}/step", json=payload)
            new_obs = step_resp.json()

            reward = new_obs.get("reward", 0.0)
            total_reward += reward

            print(f"  Events: {new_obs['observable_events']}")
            print(f"  Reward: {reward:+.3f} | Capital: {new_obs['political_capital']} | Rep: {new_obs['your_reputation']}")

            history.append({"obs": str(obs['observable_events']), "action": action})
            obs = new_obs

        outcome = obs.get("proposal_outcome", "none")
        print(f"\n{'='*60}")
        print(f"  EPISODE DONE | Outcome: {outcome.upper() if outcome else 'TIMEOUT'}")
        print(f"  Total Reward: {total_reward:.3f}")
        print(f"{'='*60}\n")

        return total_reward


async def main():
    scores = {}
    for level in [1, 2, 3]:
        score = await run_episode(task_level=level)
        scores[f"level_{level}"] = score

    print("\n=== FINAL SCORES ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")
    
    # Normalize to 0.0–1.0 for OpenEnv grader
    max_possible = 1.5 * 3   # roughly
    normalized = sum(scores.values()) / max_possible
    print(f"\n  Normalized Score: {min(1.0, normalized):.3f}")


if __name__ == "__main__":
    asyncio.run(main())