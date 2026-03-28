import json
import random
import os

class EasyTask:
    def __init__(self):
        self.task_name = "easy_classification"
        self.goal = "Read the customer ticket and output ONLY the correct category (e.g., billing, technical, shipping)."
        self.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.json")

    def get_initial_state(self):
        try:
            with open(self.data_path, "r") as f:
                tickets = json.load(f)
            ticket = random.choice(tickets)
        except Exception:
            # Safe fallback if tickets.json is missing
            ticket = {"id": "E-101", "issue": "I can't log into my account.", "category": "technical"}

        observation = {
            "ticket_id": ticket.get("id", "UNKNOWN"),
            "customer_message": ticket.get("issue", "No message provided."),
            "instructions": "Output action string: classify(category='...')"
        }
        return observation, ticket