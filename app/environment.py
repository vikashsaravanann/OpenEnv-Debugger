import uuid
import json
import random
from pathlib import Path
from typing import Optional
from app.models import Observation, Action, Reward, StepResult, State
from app.graders import grade_easy, grade_medium, grade_hard

GRADERS = {
    "task_easy": grade_easy,
    "task_medium": grade_medium,
    "task_hard": grade_hard,
}

MAX_STEPS = {
    "task_easy": 3,
    "task_medium": 5,
    "task_hard": 8,
}

TICKET_POOLS = {
    "task_easy": (0, 10),
    "task_medium": (10, 20),
    "task_hard": (20, 30),
}


class SupportTriageEnv:
    def __init__(self):
        self.tickets = self._load_tickets()
        self._state: Optional[State] = None
        self._current_ticket = None

    def _load_tickets(self):
        path = Path(__file__).parent / "data" / "tickets.json"
        return json.loads(path.read_text())

    def reset(self, task_id: str = "task_easy") -> Observation:
        if task_id not in GRADERS:
            raise ValueError(f"Unknown task_id: {task_id}. Choose from: {list(GRADERS.keys())}")

        ticket = self._pick_ticket(task_id)
        self._current_ticket = ticket

        self._state = State(
            task_id=task_id,
            episode_id=str(uuid.uuid4()),
            step=0,
            current_ticket=ticket,
            actions_taken=[],
            cumulative_reward=0.01,
            done=False,
        )
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        if self._state is None:
            raise ValueError("No active episode. Call reset() first.")
        if self._state.done:
            raise ValueError("Episode is done. Call reset() to start a new one.")

        self._state.step += 1
        self._state.actions_taken.append(action.model_dump())

        # Check for tool usage in tags to populate system_context
        system_context = ""
        if "query_system_logs" in action.tags:
            system_context += "[SYSTEM LOGS]: Intermittent 500 Errors across gateways since 08:00:00Z. "
        if "fetch_billing_history" in action.tags:
            system_context += "[BILLING HISTORY]: Account in good standing but recent payment flagged by fraud detection. "
        
        self._state.system_context = system_context

        # Simulate customer reply if ticket is not closed but response is drafted
        if not action.close_ticket and action.response_draft:
            # Append interaction to the ticket body
            self._current_ticket["body"] += f"\n\n[Agent]: {action.response_draft}"
            self._current_ticket["body"] += "\n[Customer]: Please hurry and resolve this, I have provided the details."
            
            # Degrade sentiment if dragged out
            if self._state.step >= 3:
                self._current_ticket["customer_sentiment"] = "angry"

        # Get grader for current task
        grader = GRADERS[self._state.task_id]
        reward_val, breakdown, reason = grader(
            action, self._current_ticket, self._state.step
        )

        # Partial progress reward: small bonus each step for trying
        if self._state.step > 1:
            reward_val = reward_val + 0.01  # tiny progress signal

        reward_val = round(max(0.01, min(0.99, reward_val)), 3)
        reward = Reward(value=reward_val, breakdown=breakdown, reason=reason)
        # The grader returns the TOTAL episode score (including step penalties).
        # We simply assign it to cumulative_reward, rather than accumulating it.
        self._state.cumulative_reward = reward_val

        # Check if episode should end
        max_steps = MAX_STEPS[self._state.task_id]
        done = (
            self._state.step >= max_steps
            or action.close_ticket
        )

        if done:
            self._state.done = True

        obs = self._make_observation()

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "cumulative_reward": self._state.cumulative_reward,
                "steps_remaining": max_steps - self._state.step,
                "episode_id": self._state.episode_id,
                "ground_truth": self._current_ticket["ground_truth"],
            },
        )

    def state(self) -> State:
        if self._state is None:
            raise ValueError("No active episode. Call reset() first.")
        return self._state

    def _make_observation(self) -> Observation:
        t = self._current_ticket
        return Observation(
            ticket_id=t["ticket_id"],
            subject=t["subject"],
            body=t["body"],
            customer_tier=t["customer_tier"],
            previous_contacts=t["previous_contacts"],
            created_at=t["created_at"],
            attachments=t.get("attachments", []),
            customer_sentiment=t.get("customer_sentiment", "neutral"),
            sla_breach_risk=t.get("sla_breach_risk", False),
            task_id=self._state.task_id,
            step_number=self._state.step,
            max_steps=MAX_STEPS[self._state.task_id],
            system_context=self._state.system_context if getattr(self, '_state', None) else "",
        )

    def _pick_ticket(self, task_id: str):
        start, end = TICKET_POOLS[task_id]
        pool = self.tickets[start:end]
        if not pool:
            pool = self.tickets
        return random.choice(pool)