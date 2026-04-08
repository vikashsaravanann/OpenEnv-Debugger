import os
import sys
import json
import requests
from openai import OpenAI

# --- MANDATORY VARIABLES ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

ENV_API_URL = "https://vikashsaravanan-openenv-support-triage.hf.space"
EPISODES_PER_TASK = 3

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert customer support triage agent.
Given a support ticket, respond with ONLY a valid JSON object — no markdown, no explanation.
Fields:
- category: billing|technical|shipping|account|general
- priority: low|medium|high|critical
- assigned_team: tech_support|billing_team|shipping_team|account_team|general_support
- response_draft: string (professional reply, 50-150 words for hard tasks)
- escalate: true|false
- close_ticket: true|false
- tags: list of strings
Rules: Always set category, priority, and assigned_team. Output ONLY JSON."""


def build_prompt(obs: dict) -> str:
    return (
        f"Support Ticket:
"
        f"Ticket ID: {obs.get(chr(39)+'ticket_id'+chr(39), chr(39)+chr(39))}
"
        f"Subject: {obs.get(chr(39)+'subject'+chr(39), chr(39)+chr(39))}
"
        f"Customer Tier: {obs.get(chr(39)+'customer_tier'+chr(39), chr(39)+chr(39))}
"
        f"Sentiment: {obs.get(chr(39)+'customer_sentiment'+chr(39), chr(39)+'neutral'+chr(39))}
"
        f"Message: {obs.get(chr(39)+'body'+chr(39), chr(39)+chr(39))}
"
        f"Task: {obs.get(chr(39)+'task_id'+chr(39), chr(39)+chr(39))}
"
        f"Step: {obs.get(chr(39)+'step_number'+chr(39), 1)} of {obs.get(chr(39)+'max_steps'+chr(39), 1)}
"
        "Respond with JSON only."
    )


def call_llm(obs: dict) -> dict:
    prompt = build_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        raw = completion.choices[0].message.content.strip()
        if raw.startswith("")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[LLM Error] {e}", file=sys.stderr)
        return {
            "category": "general",
            "priority": "medium",
            "assigned_team": "general_support",
            "escalate": False,
            "close_ticket": False,
            "tags": [],
        }


def run_episode(task_id: str, max_steps: int) -> float:
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
        cumulative += reward.get("value", 0) if isinstance(reward, dict) else reward
        if result.get("done", False):
            break
        obs = result.get("observation", obs)
    return round(cumulative, 4)


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
