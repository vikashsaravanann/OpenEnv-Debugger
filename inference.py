import os
import sys
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

ENV_API_URL = os.getenv("ENV_URL", "http://localhost:7860")
EPISODES_PER_TASK = 3

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert customer support triage agent.
Respond with ONLY valid JSON. Fields:
- category: billing|technical|shipping|account|general
- priority: low|medium|high|critical
- assigned_team: tech_support|billing_team|shipping_team|account_team|general_support
- response_draft: string
- escalate: true|false
- close_ticket: true|false
- tags: list of strings
Always set category, priority, and assigned_team. Output ONLY JSON."""


def build_prompt(obs):
    return (
        f"Ticket ID: {obs.get('ticket_id', '')}\n"
        f"Subject: {obs.get('subject', '')}\n"
        f"Customer Tier: {obs.get('customer_tier', '')}\n"
        f"Sentiment: {obs.get('customer_sentiment', 'neutral')}\n"
        f"Message: {obs.get('body', '')}\n"
        f"Task: {obs.get('task_id', '')}\n"
        f"Step: {obs.get('step_number', 1)} of {obs.get('max_steps', 1)}\n"
        "Respond with JSON only."
    )


def call_llm(obs):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(obs)},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        raw = completion.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[LLM Error] {e}", file=sys.stderr)
        return {"category": "general", "priority": "medium", "assigned_team": "general_support", "escalate": False, "close_ticket": False, "tags": []}


def run_episode(task_id, max_steps):
    try:
        resp = requests.post(f"{ENV_API_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
        cumulative = 0.0
        for _ in range(max_steps):
            action = call_llm(obs)
            step_resp = requests.post(f"{ENV_API_URL}/step", json=action, timeout=30)
            step_resp.raise_for_status()
            result = step_resp.json()
            reward = result.get("reward", {})
            
            # Extract reward value robustly
            if isinstance(reward, dict):
                val = reward.get("value", 0)
            else:
                val = float(reward or 0)
            
            cumulative += val
            
            if result.get("done", False):
                break
            obs = result.get("observation", obs)
        return round(cumulative, 4)
    except Exception as e:
        print(f"[Episode Error] task={task_id}: {e}", file=sys.stderr)
        return 0.0


def main():
    print("[START]")
    tasks = [("task_easy", 3), ("task_medium", 5), ("task_hard", 8)]
    all_scores = {}

    try:
        health = requests.get(f"{ENV_API_URL}/health", timeout=10)
        health.raise_for_status()
        print("[INFO] Environment reachable")
    except Exception as e:
        print(f"[ERROR] Cannot reach environment: {e}")
        sys.exit(1)

    for task_id, max_steps in tasks:
        scores = []
        for ep in range(EPISODES_PER_TASK):
            score = run_episode(task_id, max_steps)
            scores.append(score)
            print(f"[STEP] task={task_id} episode={ep+1} score={score}")
        avg = round(sum(scores) / len(scores), 4)
        all_scores[task_id] = avg
        print(f"[STEP] task={task_id} average={avg}")

    overall = round(sum(all_scores.values()) / len(all_scores), 4)
    print(f"[END] scores={json.dumps(all_scores)} overall={overall}")


if __name__ == "__main__":
    main()
