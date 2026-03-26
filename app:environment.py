import uuid
import json
from pathlib import Path
from app.models import *
from app.graders import grade_easy, grade_medium, grade_hard

GRADERS = {
    "task_easy": grade_easy,
    "task_medium": grade_medium,
    "task_hard": grade_hard,
}

MAX_STEPS = {"task_easy": 3, "task_medium": 5, "task_hard": 8}

class SupportTriageEnv:
    def __init__(self):
        self.tickets = self._load_tickets()
        self._state: Optional[State] = None
        self._current_ticket = None

    def _load_tickets(self):
        path = Path(__file__).parent / "data" / "tickets.json"
        return json.loads(path.read_text())

    def reset(self, task_id: str = "task_easy") -> Observation:
        ticket = self._pick_ticket(task_id)
        self._current_ticket = ticket
        self._state = State(
            task_id=task_id,
            episode_id=str(uuid.uuid4()),
            step=0,
            current_ticket=ticket,
            actions_taken=[],
            cumulative_reward=0.0,
            done=False,
        )
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        if self._state is None or self._state.done:
            raise ValueError("Call reset() first")

        self._state.step += 1
        self._state.actions_taken.append(action.dict())

        grader = GRADERS[self._state.task_id]
        reward_val, breakdown, reason = grader(
            action, self._current_ticket, self._state.step
        )
        reward = Reward(value=reward_val, breakdown=breakdown, reason=reason)
        self._state.cumulative_reward += reward_val

        max_steps = MAX_STEPS[self._state.task_id]
        done = self._state.step >= max_steps or action.close_ticket

        if done:
            self._state.done = True

        obs = self._make_observation()
        return StepResult(observation=obs, reward=reward, done=done, info={
            "cumulative_reward": self._state.cumulative_reward,
            "ground_truth": self._current_ticket["ground_truth"],
        })

    def state(self) -> State:
        if self._state is None:
            raise ValueError("No active episode")
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
            task_id=self._state.task_id,
            step_number=self._state.step,
            max_steps=MAX_STEPS[self._state.task_id],
        )

    def _pick_ticket(self, task_id: str):
        import random
        pool = {
            "task_easy": self.tickets[:10],
            "task_medium": self.tickets[10:20],
            "task_hard": self.tickets[20:],
        }
        return random.choice(pool.get(task_id, self.tickets))