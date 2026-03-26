from fastapi import FastAPI, HTTPException
from app.models import *
from app.environment import SupportTriageEnv
import json

app = FastAPI(title="Support Ticket Triage OpenEnv", version="1.0.0")
env = SupportTriageEnv()

@app.post("/reset", response_model=Observation)
def reset(task_id: str = "task_easy"):
    return env.reset(task_id)

@app.post("/step", response_model=StepResult)
def step(action: Action):
    try:
        return env.step(action)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/state", response_model=State)
def state():
    try:
        return env.state()
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "id": "task_easy",
                "name": "Ticket Classification",
                "difficulty": "easy",
                "description": "Classify ticket into correct category",
                "action_schema": {
                    "required": ["category"],
                    "optional": ["tags"]
                }
            },
            {
                "id": "task_medium",
                "name": "Triage and Routing",
                "difficulty": "medium",
                "description": "Classify, set priority, and route to team",
                "action_schema": {
                    "required": ["category", "priority", "assigned_team"],
                    "optional": ["tags", "escalate"]
                }
            },
            {
                "id": "task_hard",
                "name": "Full Resolution",
                "difficulty": "hard",
                "description": "Full triage + draft resolution response",
                "action_schema": {
                    "required": ["category", "priority", "assigned_team", "response_draft"],
                    "optional": ["escalate", "close_ticket", "tags"]
                }
            }
        ]
    }

@app.get("/grader")
def grader():
    if env._state is None:
        raise HTTPException(400, "No active episode")
    state = env.state()
    return {
        "task_id": state.task_id,
        "episode_id": state.episode_id,
        "cumulative_reward": state.cumulative_reward,
        "steps_taken": state.step,
        "done": state.done,
        "actions_taken": state.actions_taken,
    }

@app.post("/baseline")
def baseline():
    """Run baseline inference and return scores for all 3 tasks"""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "baseline.py", "--return-json"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise HTTPException(500, result.stderr)
    return json.loads(result.stdout)

@app.get("/health")
def health():
    return {"status": "ok"}