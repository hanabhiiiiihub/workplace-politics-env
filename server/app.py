# server/app.py
from fastapi import FastAPI, Request
from server.environment import PoliticsEnv
from server.models import PoliticsAction

app = FastAPI(title="Workplace Politics OpenEnv")

envs = {
    1: PoliticsEnv(task_level=1),
    2: PoliticsEnv(task_level=2),
    3: PoliticsEnv(task_level=3),
}

@app.get("/")
async def root():
    return {"status": "ok", "name": "Workplace Politics OpenEnv", "levels": [1, 2, 3]}

@app.post("/reset")
async def reset(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    level = int(body.get("task_level", 1))
    if level not in envs:
        level = 1
    obs = envs[level].reset()
    return obs.dict()

@app.post("/step")
async def step(request: Request):
    body   = await request.json()
    level  = int(body.get("task_level", 1))
    if level not in envs:
        level = 1
    action = PoliticsAction(**{k: v for k, v in body.items() if k != "task_level"})
    obs    = envs[level].step(action)
    result = obs.dict()
    if obs.done:
        result["grade"] = envs[level].grade()
    return result

@app.get("/state")
async def state(task_level: int = 1):
    if task_level not in envs:
        task_level = 1
    return envs[task_level].get_state().dict()

@app.get("/grade")
async def grade(task_level: int = 1):
    if task_level not in envs:
        task_level = 1
    score = envs[task_level].grade()
    env   = envs[task_level]
    return {
        "score":              score,
        "task_level":         task_level,
        "proposal_outcome":   env._proposal_outcome,
        "turns_used":         env._turn,
        "coalition_strength": env._coalition_strength(),
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
