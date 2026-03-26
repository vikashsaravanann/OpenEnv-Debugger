from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SHIPPING = "shipping"
    ACCOUNT = "account"
    GENERAL = "general"


class Observation(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: str
    previous_contacts: int
    created_at: str
    attachments: List[str] = []
    customer_sentiment: str = "neutral"
    sla_breach_risk: bool = False
    task_id: str
    step_number: int
    max_steps: int


class Action(BaseModel):
    priority: Optional[Priority] = None
    category: Optional[Category] = None
    assigned_team: Optional[str] = None
    response_draft: Optional[str] = None
    escalate: bool = False
    close_ticket: bool = False
    tags: List[str] = []


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    breakdown: Dict[str, float] = {}
    reason: str = ""


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = {}


class State(BaseModel):
    task_id: str
    episode_id: str
    step: int
    current_ticket: Dict[str, Any]
    actions_taken: List[Dict[str, Any]]
    cumulative_reward: float
    done: bool


class ResetRequest(BaseModel):
    task_id: str = "task_easy"