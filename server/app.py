# server/app.py
import traceback
from server.environment import PoliticsEnv
from server.models import PoliticsAction, PoliticsObservation
import sys, os
sys.path.insert(0, os.path.abspath("OpenEnv/src/openenv"))
from core.env_server import create_fastapi_app


def env_factory_level1():
    try:
        return PoliticsEnv(task_level=1)
    except Exception:
        traceback.print_exc()
        raise

def env_factory_level2():
    return PoliticsEnv(task_level=2)

def env_factory_level3():
    return PoliticsEnv(task_level=3)


# Default app uses level 1; level is set via query param
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Workplace Politics OpenEnv")

envs = {
    1: PoliticsEnv(task_level=1),
    2: PoliticsEnv(task_level=2),
    3: PoliticsEnv(task_level=3),
}

@app.post("/reset")
async def reset(request: Request):
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    level = int(body.get("task_level", 1))
    obs = envs[level].reset()
    return obs.dict()

@app.post("/step")
async def step(request: Request):
    body = await request.json()
    level = int(body.get("task_level", 1))
    action = PoliticsAction(**{k: v for k, v in body.items() if k != "task_level"})
    obs = envs[level].step(action)
    return obs.dict()

@app.get("/state")
async def state(task_level: int = 1):
    return envs[task_level].get_state().dict()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()