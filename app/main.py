from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from typing import Optional
from pydantic import BaseModel
from app.models import *
from app.environment import SupportTriageEnv
import json

app = FastAPI(title="Support Ticket Triage OpenEnv", version="1.0.0")

# HF Spaces uses HEAD requests to detect if the space is "Running".
# Without explicit HEAD handlers FastAPI returns 405, keeping the badge stuck on "Building".
@app.head("/")
def head_root():
    return HTMLResponse(content="", status_code=200)

@app.head("/health")
def head_health():
    return HTMLResponse(content="", status_code=200)

@app.get("/", response_class=HTMLResponse)
def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>OpenEnv Support Triage</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
    }
    .card {
      background: rgba(255,255,255,0.07);
      backdrop-filter: blur(16px);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 20px;
      padding: 48px 56px;
      max-width: 600px;
      text-align: center;
      box-shadow: 0 8px 48px rgba(0,0,0,0.4);
    }
    .badge {
      display: inline-block;
      background: #22c55e;
      color: #fff;
      font-size: 13px;
      font-weight: 600;
      padding: 4px 14px;
      border-radius: 999px;
      margin-bottom: 24px;
      letter-spacing: 0.5px;
    }
    h1 { font-size: 2rem; font-weight: 700; margin-bottom: 12px; }
    p  { color: rgba(255,255,255,0.65); font-size: 1rem; line-height: 1.6; margin-bottom: 32px; }
    .endpoints { text-align: left; }
    .endpoints h2 { font-size: 1rem; font-weight: 600; margin-bottom: 12px; color: rgba(255,255,255,0.8); }
    .endpoint {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      background: rgba(255,255,255,0.05);
      border-radius: 8px;
      margin-bottom: 8px;
      font-size: 0.875rem;
    }
    .method {
      font-weight: 700;
      font-size: 0.75rem;
      padding: 2px 8px;
      border-radius: 4px;
      min-width: 46px;
      text-align: center;
    }
    .get  { background: #3b82f6; }
    .post { background: #10b981; }
    code { color: rgba(255,255,255,0.85); }
    .docs-link {
      display: inline-block;
      margin-top: 24px;
      background: linear-gradient(90deg, #6366f1, #8b5cf6);
      color: #fff;
      text-decoration: none;
      padding: 12px 28px;
      border-radius: 10px;
      font-weight: 600;
      font-size: 0.95rem;
      transition: opacity 0.2s;
    }
    .docs-link:hover { opacity: 0.85; }
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">🟢 Running</div>
    <h1>OpenEnv Support Triage</h1>
    <p>A reinforcement-learning environment for AI-powered support ticket triage. Reset, step, and grade your agent's performance.</p>
    <div class="endpoints">
      <h2>Available Endpoints</h2>
      <div class="endpoint"><span class="method post">POST</span><code>/reset</code></div>
      <div class="endpoint"><span class="method post">POST</span><code>/step</code></div>
      <div class="endpoint"><span class="method get">GET</span><code>/state</code></div>
      <div class="endpoint"><span class="method get">GET</span><code>/tasks</code></div>
      <div class="endpoint"><span class="method get">GET</span><code>/grader</code></div>
      <div class="endpoint"><span class="method get">GET</span><code>/health</code></div>
    </div>
    <a href="/docs" class="docs-link">📖 Open API Docs</a>
  </div>
</body>
</html>"""

env = SupportTriageEnv()

class ResetRequest(BaseModel):
    task_id: str = "task_easy"

@app.post("/reset", response_model=Observation)
def reset(body: Optional[ResetRequest] = Body(default=None)):
    task_id = body.task_id if body else "task_easy"
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
                "has_grader": True,
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
                "has_grader": True,
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
                "has_grader": True,
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
