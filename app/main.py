import json
import subprocess
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import Observation, Action, StepResult, State, ResetRequest
from app.environment import SupportTriageEnv

app = FastAPI(
    title="Support Ticket Triage — OpenEnv",
    description="An OpenEnv environment where AI agents learn to triage and resolve customer support tickets.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SupportTriageEnv()


# ─────────────────────────────────────────
#  CORE OPENENV ENDPOINTS
# ─────────────────────────────────────────

@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest):
    """Reset the environment and return the first observation."""
    try:
        return env.reset(request.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """Take an action in the environment."""
    try:
        return env.step(action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=State)
def state():
    """Return the current environment state."""
    try:
        return env.state()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─────────────────────────────────────────
#  REQUIRED EXTRA ENDPOINTS
# ─────────────────────────────────────────

@app.get("/tasks")
def tasks():
    """Return all tasks and their action schemas."""
    return {
        "tasks": [
            {
                "id": "task_easy",
                "name": "Ticket Classification",
                "difficulty": "easy",
                "description": "Classify a support ticket into the correct category.",
                "max_steps": 3,
                "reward_range": [0.0, 1.0],
                "action_schema": {
                    "required": ["category"],
                    "optional": ["tags", "close_ticket"],
                    "fields": {
                        "category": "billing | technical | shipping | account | general",
                        "tags": "list of strings",
                        "close_ticket": "boolean"
                    }
                }
            },
            {
                "id": "task_medium",
                "name": "Triage and Routing",
                "difficulty": "medium",
                "description": "Classify the ticket, assign priority, and route to the correct team.",
                "max_steps": 5,
                "reward_range": [0.0, 1.0],
                "action_schema": {
                    "required": ["category", "priority", "assigned_team"],
                    "optional": ["tags", "escalate", "close_ticket"],
                    "fields": {
                        "category": "billing | technical | shipping | account | general",
                        "priority": "low | medium | high | critical",
                        "assigned_team": "tech_support | billing_team | shipping_team | account_team | general_support",
                        "escalate": "boolean",
                        "tags": "list of strings",
                        "close_ticket": "boolean"
                    }
                }
            },
            {
                "id": "task_hard",
                "name": "Full Resolution",
                "difficulty": "hard",
                "description": "Full triage plus write a helpful, professional response to the customer.",
                "max_steps": 8,
                "reward_range": [-1.0, 1.0],
                "action_schema": {
                    "required": ["category", "priority", "assigned_team", "response_draft"],
                    "optional": ["tags", "escalate", "close_ticket"],
                    "fields": {
                        "category": "billing | technical | shipping | account | general",
                        "priority": "low | medium | high | critical",
                        "assigned_team": "tech_support | billing_team | shipping_team | account_team | general_support",
                        "response_draft": "string — your reply to the customer",
                        "escalate": "boolean",
                        "tags": "list of strings",
                        "close_ticket": "boolean"
                    }
                }
            }
        ]
    }


@app.get("/grader")
def grader():
    """Return grader score for the current episode."""
    try:
        s = env.state()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "task_id": s.task_id,
        "episode_id": s.episode_id,
        "cumulative_reward": s.cumulative_reward,
        "steps_taken": s.step,
        "done": s.done,
        "actions_taken": s.actions_taken,
        "final_score": round(
            max(0.0, min(1.0, s.cumulative_reward / max(s.step, 1))), 3
        ),
    }


@app.post("/baseline")
def baseline():
    """Run the baseline inference script and return scores for all 3 tasks."""
    try:
        result = subprocess.run(
            [sys.executable, "baseline.py", "--return-json"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Baseline script failed: {result.stderr}"
            )
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Baseline script timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "name": "Support Ticket Triage OpenEnv",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
    }