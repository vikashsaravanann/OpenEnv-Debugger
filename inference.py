"""
Inference Script: Support Ticket Triage (OpenEnv)
==================================================
MANDATORY VARIABLES CONFIGURED:
    API_BASE_URL    The API endpoint for the LLM.
    MODEL_NAME      The model identifier to use for inference.
    HF_TOKEN        Hugging Face / API key.
"""

import os
import json
import requests
from openai import OpenAI

# --- 1. MANDATORY LLM CONFIGURATION ---
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# --- 2. YOUR ENVIRONMENT ENDPOINT ---
ENV_API_URL = "https://vikashsaravanan-openenv-support-triage.hf.space"
MAX_STEPS = 5

# --- 3. SYSTEM PROMPT (instructs LLM to return valid Action JSON) ---
SYSTEM_PROMPT = """\
You are an AI Customer Support Triage Agent.

You will be given a JSON observation for a support ticket.
Your job is to output ONLY a valid JSON object that matches this schema — no extra text, no markdown:

{
  "category": "<billing|technical|shipping|account|general>",
  "priority": "<low|medium|high|critical>",
  "assigned_team": "<e.g. billing-team, tech-support, shipping-ops, account-management>",
  "escalate": <true|false>,
  "close_ticket": <true|false>,
  "tags": ["<tag1>", "<tag2>"],
  "response_draft": "<Optional short professional reply to the customer>"
}

Rules:
- "category" and "priority" MUST always be included and use the exact enum values.
- "assigned_team" should be a sensible lowercase kebab-case team name.
- Output ONLY the JSON object. Do NOT include markdown fences or explanations.
"""


def call_llm(client, observation: dict, task_id: str) -> dict:
    """Call the LLM and return a parsed Action dict."""
    # Tailor the user message based on task difficulty
    user_msg = (
        f"Task: {task_id}\n"
        f"Support ticket observation:\n{json.dumps(observation, indent=2)}\n\n"
        "Respond with ONLY the JSON action object."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},  # enforce JSON mode where supported
        )
        raw = completion.choices[0].message.content.strip()
    except Exception:
        # Some providers don't support response_format — retry without it
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=400,
            )
            raw = completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [LLM] Request failed: {e}")
            return _fallback_action(observation)

    # Strip markdown fences if the model added them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [LLM] Could not parse JSON: {raw[:200]}")
        action = _fallback_action(observation)

    # Validate / default required enum fields
    valid_categories = {"billing", "technical", "shipping", "account", "general"}
    valid_priorities = {"low", "medium", "high", "critical"}

    if action.get("category") not in valid_categories:
        action["category"] = "general"
    if action.get("priority") not in valid_priorities:
        action["priority"] = "medium"
    if not action.get("assigned_team"):
        action["assigned_team"] = f"{action['category']}-team"
    if "escalate" not in action:
        action["escalate"] = False
    if "close_ticket" not in action:
        action["close_ticket"] = False
    if "tags" not in action:
        action["tags"] = []

    return action


def _fallback_action(observation: dict) -> dict:
    """Minimal safe fallback when the LLM fails."""
    subject = observation.get("subject", "").lower()
    if "bill" in subject or "charge" in subject or "invoice" in subject:
        category = "billing"
    elif "ship" in subject or "deliver" in subject or "track" in subject:
        category = "shipping"
    elif "account" in subject or "login" in subject or "password" in subject:
        category = "account"
    elif "bug" in subject or "error" in subject or "crash" in subject or "technical" in subject:
        category = "technical"
    else:
        category = "general"

    return {
        "category": category,
        "priority": "medium",
        "assigned_team": f"{category}-team",
        "escalate": False,
        "close_ticket": False,
        "tags": [],
        "response_draft": (
            "Thank you for contacting support. We have received your request "
            "and a specialist will assist you shortly."
        ),
    }


def main():
    # Initialize the OpenAI-compatible client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"Connecting to OpenEnv at: {ENV_API_URL}")
    print(f"LLM endpoint: {API_BASE_URL} | Model: {MODEL_NAME}")

    # --- Step 1: Reset the environment ---
    try:
        response = requests.post(
            f"{ENV_API_URL}/reset",
            json={"task_id": "task_easy"},
            timeout=30,
        )
        response.raise_for_status()
        env_state = response.json()
        task_id = env_state.get("task_id", "task_easy")
        print(f"Environment reset successful. Task: {task_id}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to connect to environment: {e}")
        return

    # --- Step 2: Agent Loop ---
    for step in range(1, MAX_STEPS + 1):
        observation = {
            k: v
            for k, v in env_state.items()
            if k not in ("done", "reward", "task_id", "step_number", "max_steps")
        }
        is_done = env_state.get("done", False)

        if is_done:
            print(f"\nEpisode complete!")
            break

        print(f"\n--- Step {step} ---")
        print(f"Ticket: {observation.get('ticket_id', '?')} | {observation.get('subject', '?')[:60]}")

        # --- LLM decides the action ---
        action_dict = call_llm(client, env_state, task_id)
        print(f"Action: category={action_dict.get('category')} | priority={action_dict.get('priority')} | team={action_dict.get('assigned_team')}")

        # --- Send action directly to /step (Action schema, NOT wrapped in "action" key) ---
        try:
            step_resp = requests.post(
                f"{ENV_API_URL}/step",
                json=action_dict,
                timeout=30,
            )
            step_resp.raise_for_status()
            env_state = step_resp.json()

            reward = env_state.get("reward", {})
            reward_val = reward.get("value", 0) if isinstance(reward, dict) else reward
            done = env_state.get("done", False)
            cumulative = env_state.get("info", {}).get("cumulative_reward", "?")

            print(f"Reward: {reward_val} | Cumulative: {cumulative} | Done: {done}")

            if done:
                print(f"\nEpisode finished at step {step}.")
                break

        except Exception as e:
            print(f"Failed to send step to environment: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response body: {e.response.text[:500]}")
            break
    else:
        print("\nReached max steps without episode ending.")


if __name__ == "__main__":
    main()