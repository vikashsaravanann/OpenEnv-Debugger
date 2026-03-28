import json
import random
import os

class HardTask:
    def __init__(self):
        self.task_name = "hard_resolution"
        self.goal = "Perform full routing (category, priority, team) AND draft a professional response to the customer."
        self.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.json")

    def get_initial_state(self):
        try:
            with open(self.data_path, "r") as f:
                tickets = json.load(f)
            ticket = random.choice(tickets)
        except Exception:
            # Safe fallback if tickets.json is missing
            ticket = {"id": "H-303", "issue": "My server crashed and I lost data.", "category": "technical", "priority": "critical", "team": "devops"}

        observation = {
            "ticket_id": ticket.get("id", "UNKNOWN"),
            "customer_message": ticket.get("issue", "No message provided."),
            "instructions": "Output action string: resolve(category='...', priority='...', team='...', response='...')"
        }
        return observation, ticket