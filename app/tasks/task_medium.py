import json
import random
import os

class MediumTask:
    def __init__(self):
        self.task_name = "medium_triage"
        self.goal = "Read the customer issue. Categorize the ticket, assign a priority level (low, medium, high, critical), and route it to the correct team."
        # Dynamically find your tickets.json file in the app/data folder
        self.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.json")

    def get_initial_state(self):
        """Loads a random ticket and formats the observation for the AI agent."""
        try:
            with open(self.data_path, "r") as f:
                tickets = json.load(f)
            ticket = random.choice(tickets)
        except Exception as e:
            # Fallback ticket just in case the JSON file doesn't load
            ticket = {
                "id": "ERR-999",
                "issue": "I was double-charged on my credit card for the Pro subscription.",
                "category": "billing",
                "priority": "high",
                "team": "finance"
            }

        # The observation is what the AI agent actually sees
        observation = {
            "ticket_id": ticket.get("id", "UNKNOWN"),
            "customer_message": ticket.get("issue", ticket.get("text", "No message provided.")),
            "instructions": "Output action string in format: route(category='...', priority='...', team='...')"
        }

        # Return the observation for the agent, and the hidden 'true' ticket data for your grader
        return observation, ticket