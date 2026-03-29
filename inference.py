"""
Inference Script Example: Support Ticket Triage (Fresh Tensors)
===============================================================
MANDATORY VARIABLES CONFIGURED:
    API_BASE_URL    The API endpoint for the LLM.
    MODEL_NAME      The model identifier to use for inference.
    HF_TOKEN        Hugging Face / API key.
"""

import os
import requests
import json
from openai import OpenAI

# --- 1. MANDATORY LLM CONFIGURATION ---
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# --- 2. YOUR ENVIRONMENT ENDPOINT ---
# This points directly to your deployed Hugging Face FastAPI backend
ENV_API_URL = "https://vikashsaravanan-openenv-support-triage.hf.space"
MAX_STEPS = 5

SYSTEM_PROMPT = """
You are an AI Customer Support Triage Agent.
You will be given a JSON observation of a customer support ticket.
Your job is to read the ticket and output a valid action string to categorize, prioritize, and route the ticket to the correct team.
Do not include any extra text or explanations, just the action.
"""

def main():
    # Initialize the mandatory OpenAI Client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"Connecting to Support Triage Environment at: {ENV_API_URL}...")

    # Step 1: Reset the environment to get the first support ticket
    try:
        response = requests.post(f"{ENV_API_URL}/reset")
        response.raise_for_status()
        env_state = response.json()
        print("Successfully connected and fetched the first ticket!")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to connect to environment: {e}")
        return

    # Step 2: The Agent Loop
    for step in range(1, MAX_STEPS + 1):
        observation = env_state.get("observation", {})
        is_done = env_state.get("done", False)

        if is_done:
            print(f"\nEpisode complete! Final Reward: {env_state.get('reward', 0)}")
            break

        print(f"\n--- Step {step} ---")
        print(f"Ticket Observation: {json.dumps(observation, indent=2)}")

        # Step 3: Ask the LLM what action to take
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current ticket: {json.dumps(observation)}\nWhat is your action?"}
        ]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=150,
            )
            action_text = completion.choices[0].message.content.strip()
            print(f"LLM Decided Action: {action_text}")
        except Exception as e:
            print(f"LLM request failed: {e}")
            action_text = "noop()" # Fallback action

        # Step 4: Send the LLM's action back to your FastAPI environment
        try:
            step_payload = {"action": action_text}
            step_response = requests.post(f"{ENV_API_URL}/step", json=step_payload)
            step_response.raise_for_status()
            env_state = step_response.json()

            reward = env_state.get("reward", 0)
            print(f"Environment Output -> Reward: {reward} | Done: {env_state.get('done', False)}")

        except Exception as e:
            print(f"Failed to send step to environment: {e}")
            break
    else:
        print("\nReached max steps.")

if __name__ == "__main__":
    main()