# inference.py
import os
import json
import httpx
from openai import OpenAI

# ── Required environment variables (checklist compliant) ──────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN         = os.getenv("HF_TOKEN")           # NO default — checklist requirement
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # optional

ENV_URL = os.getenv("ENV_URL", "https://hanabhi-workplace-politics-env.hf.space")

# ── OpenAI client — all LLM calls use this (checklist requirement) ────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-token",
)

TASKS = [
    {"task_id": "level_1_easy",   "level": 1, "max_steps": 15},
    {"task_id": "level_2_medium", "level": 2, "max_steps": 12},
    {"task_id": "level_3_hard",   "level": 3, "max_steps": 10},
]

SYSTEM_PROMPT = """\
You are a strategic AI agent in a workplace politics simulation.
Your FINAL GOAL is to call make_proposal -> boss. You MUST do this before turns run out.

Characters: boss, rival, friend, gatekeeper, fence_sitter
Actions: schedule_meeting, share_info, request_support, observe, build_alliance, make_proposal
info_level options (only for share_info): full, partial, vague

TURN-BY-TURN PLAN - follow this strictly:
  Turn 1: schedule_meeting -> gatekeeper
  Turn 2: share_info (partial) -> friend
  Turn 3: request_support -> friend
  Turn 4: share_info (partial) -> fence_sitter
  Turn 5: share_info (partial) -> boss
  Turn 6: request_support -> gatekeeper
  Turn 7: make_proposal -> boss   <-- YOU MUST DO THIS BY TURN 7
  Turn 8+: make_proposal -> boss  <-- ALWAYS PROPOSE IF BEHIND

CRITICAL RULES:
- If (max_turns - turn) <= 3, ALWAYS output make_proposal -> boss
- If you have 2+ supportive characters, ALWAYS output make_proposal -> boss
- Never repeat the same action+target twice in a row
- Output ONLY a JSON object. No explanation. No markdown. No code fences.

Examples:
{"action_type": "schedule_meeting", "target": "gatekeeper"}
{"action_type": "share_info", "target": "friend", "info_level": "partial"}
{"action_type": "make_proposal", "target": "boss"}
"""

VALID_ACTIONS = {"schedule_meeting", "share_info", "request_support",
                 "observe", "build_alliance", "make_proposal"}
VALID_TARGETS = {"boss", "rival", "friend", "gatekeeper", "fence_sitter"}
VALID_INFO    = {"full", "partial", "vague"}

HEURISTIC_SEQUENCE = [
    {"action_type": "schedule_meeting", "target": "gatekeeper"},
    {"action_type": "share_info",       "target": "friend",       "info_level": "partial"},
    {"action_type": "request_support",  "target": "friend"},
    {"action_type": "share_info",       "target": "fence_sitter", "info_level": "partial"},
    {"action_type": "share_info",       "target": "boss",         "info_level": "partial"},
    {"action_type": "request_support",  "target": "gatekeeper"},
    {"action_type": "make_proposal",    "target": "boss"},
    {"action_type": "make_proposal",    "target": "boss"},
    {"action_type": "make_proposal",    "target": "boss"},
    {"action_type": "make_proposal",    "target": "boss"},
]


def get_heuristic_action(obs: dict, step: int) -> dict:
    turn  = obs.get("turn", step)
    max_t = obs.get("max_turns", 15)
    if max_t - turn <= 2:
        return {"action_type": "make_proposal", "target": "boss"}
    idx = min(step - 1, len(HEURISTIC_SEQUENCE) - 1)
    return HEURISTIC_SEQUENCE[idx]


def parse_action(raw: str):
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break
    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    action_type = obj.get("action_type", "").strip().lower()
    target      = obj.get("target", "").strip().lower()
    if action_type not in VALID_ACTIONS:
        return None
    if target not in VALID_TARGETS:
        return None
    result = {"action_type": action_type, "target": target}
    if action_type == "share_info":
        info = obj.get("info_level", "partial").strip().lower()
        result["info_level"] = info if info in VALID_INFO else "partial"
    return result


def get_llm_action(obs: dict, history: list, step: int) -> dict:
    turn    = obs.get("turn", step)
    max_t   = obs.get("max_turns", 15)
    support = obs.get("known_support", {})
    capital = obs.get("political_capital", 1.0)

    # Force proposal when running out of turns
    if max_t - turn <= 2:
        return {"action_type": "make_proposal", "target": "boss"}

    # Force proposal when coalition is strong enough
    supportive = [k for k, v in support.items() if v == "supportive"]
    if len(supportive) >= 2 and capital > 0.3:
        return {"action_type": "make_proposal", "target": "boss"}

    events_str  = "\n".join(f"  - {e}" for e in obs.get("observable_events", []))
    support_str = json.dumps(support)
    obs_text = (
        f"Turn {turn} of {max_t} — moves remaining: {max_t - turn}\n"
        f"Political capital: {capital:.2f}\n"
        f"Reputation: {obs.get('your_reputation', 0.7):.2f}\n"
        f"Known support: {support_str}\n"
        f"Supportive characters: {supportive}\n"
        f"Recent events:\n{events_str}\n\n"
        f"REMINDER: If moves remaining <= 3, output make_proposal -> boss.\n"
        f"What is your next action? Reply with JSON only."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-3:]:
        messages.append({"role": "user",      "content": h["obs"]})
        messages.append({"role": "assistant", "content": json.dumps(h["action"])})
    messages.append({"role": "user", "content": obs_text})

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=80,
            )
            raw    = resp.choices[0].message.content
            action = parse_action(raw)
            if action:
                return action
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    "Invalid JSON. Reply with ONLY a JSON object, for example:\n"
                    '{"action_type": "make_proposal", "target": "boss"}'
                )
            })
        except Exception as e:
            print(f"  [LLM error attempt {attempt + 1}]: {e}")

    print(f"  [WARN] LLM failed after 3 attempts, using heuristic for step {step}")
    return get_heuristic_action(obs, step)


def run_task(task: dict) -> dict:
    task_id   = task["task_id"]
    level     = task["level"]
    max_steps = task["max_steps"]
    use_llm   = bool(HF_TOKEN)
    model_tag = MODEL_NAME if use_llm else "heuristic"

    with httpx.Client(timeout=30.0) as http:

        obs = http.post(
            f"{ENV_URL}/reset",
            json={"task_level": level}
        ).json()

        total_reward  = 0.0
        history       = []
        step          = 0
        obs_text      = ""
        inline_grade  = None   # grade returned directly in final step response

        print(f'[START] task_id="{task_id}" level={level} '
              f'model="{model_tag}" env_url="{ENV_URL}"')

        while not obs.get("done", False) and step < max_steps:
            step += 1

            if use_llm:
                action = get_llm_action(obs, history, step)
            else:
                action = get_heuristic_action(obs, step)

            payload = {**action, "task_level": level}
            try:
                new_obs = http.post(f"{ENV_URL}/step", json=payload).json()
            except Exception as e:
                print(f"  [ENV ERROR] step {step}: {e}")
                break

            reward        = new_obs.get("reward", 0.0)
            total_reward += reward

            # Grab inline grade if episode just ended
            if new_obs.get("done", False) and "grade" in new_obs:
                inline_grade = new_obs["grade"]

            n_supportive  = sum(
                1 for v in new_obs.get("known_support", {}).values()
                if v == "supportive"
            )
            support_label = (
                "high"   if n_supportive >= 3 else
                "medium" if n_supportive >= 1 else
                "low"
            )

            print(f'[STEP]  task_id="{task_id}" '
                  f'step={step} '
                  f'action="{action.get("action_type")}" '
                  f'target="{action.get("target")}" '
                  f'reward={reward:.4f} '
                  f'total_reward={total_reward:.4f} '
                  f'done={new_obs.get("done", False)} '
                  f'support="{support_label}" '
                  f'moves_left={max_steps - step}')

            history.append({"obs": obs_text, "action": action})
            obs = new_obs

        # ── Score: prefer inline grade, then /grade endpoint, then fallback ──
        score = None

        if inline_grade is not None:
            score = inline_grade

        if score is None:
            try:
                grade_resp = http.get(
                    f"{ENV_URL}/grade",
                    params={"task_level": level}
                )
                score = grade_resp.json().get("score", None)
            except Exception:
                pass

        if score is None:
            outcome = obs.get("proposal_outcome")
            if outcome == "approved":
                score = 1.0
            elif outcome == "partial":
                score = 0.4
            elif outcome == "rejected":
                score = 0.15
            else:
                score = max(0.0, min(0.35, (total_reward + 1.0) / 5.0))

        success = score >= 0.5

        print(f'[END]   task_id="{task_id}" '
              f'total_reward={total_reward:.4f} '
              f'score={score:.4f} '
              f'steps={step} '
              f'success={str(success).lower()}')

        return {
            "task_id":      task_id,
            "level":        level,
            "score":        round(score, 4),
            "total_reward": round(total_reward, 4),
            "steps":        step,
            "success":      success,
            "outcome":      obs.get("proposal_outcome"),
        }


def main():
    print(f"\n{'='*60}")
    print(f"  WORKPLACE POLITICS OPENENV — BASELINE EVALUATION")
    print(f"  Model : {MODEL_NAME if HF_TOKEN else 'heuristic (no HF_TOKEN set)'}")
    print(f"  Env   : {ENV_URL}")
    print(f"{'='*60}\n")

    results = []
    for task in TASKS:
        result = run_task(task)
        results.append(result)
        print()

    avg_score = sum(r["score"] for r in results) / len(results)

    print(f'[SUMMARY] avg_score={avg_score:.4f} '
          f'tasks={len(results)} '
          f'model="{MODEL_NAME if HF_TOKEN else "heuristic"}"')

    with open("baseline_scores.json", "w") as f:
        json.dump({
            "model":     MODEL_NAME if HF_TOKEN else "heuristic",
            "env_url":   ENV_URL,
            "results":   results,
            "avg_score": round(avg_score, 4),
        }, f, indent=2)

    print("[SAVED] baseline_scores.json")
    print(f"\n{'='*60}")
    print(f"  Level 1 (easy)  : {results[0]['score']:.4f} — {results[0]['outcome']}")
    print(f"  Level 2 (medium): {results[1]['score']:.4f} — {results[1]['outcome']}")
    print(f"  Level 3 (hard)  : {results[2]['score']:.4f} — {results[2]['outcome']}")
    print(f"  Average         : {avg_score:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()