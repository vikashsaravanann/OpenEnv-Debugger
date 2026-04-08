import os
import sys
import json
import argparse
import httpx
from openai import OpenAI

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────

BASE_URL = os.getenv("ENV_URL", "http://localhost:7860")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EPISODES_PER_TASK = 3

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are an expert customer support triage agent.

Given a support ticket, you must respond with a JSON action object.
Always respond with ONLY valid JSON — no markdown, no explanation, no code blocks.

Available fields:
- category: "billing" | "technical" | "shipping" | "account" | "general"
- priority: "low" | "medium" | "high" | "critical"
- assigned_team: "tech_support" | "billing_team" | "shipping_team" | "account_team" | "general_support"
- response_draft: string (your professional reply to the customer)
- escalate: true | false
- close_ticket: true | false
- tags: list of strings

Rules:
- Always set category
- For medium/hard tasks also set priority and assigned_team
- For hard tasks also write a helpful response_draft of 50-150 words
- Be professional, empathetic, and solution-focused
- Only escalate if the issue is truly critical and unresolvable at first level
"""


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def build_prompt(obs: dict) -> str:
    return f"""Support Ticket:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ticket ID     : {obs['ticket_id']}
Subject       : {obs['subject']}
Customer Tier : {obs['customer_tier']}
Sentiment     : {obs.get('customer_sentiment', 'neutral')}
SLA Risk      : {obs.get('sla_breach_risk', False)}
Prior Contacts: {obs['previous_contacts']}
Created At    : {obs['created_at']}
Attachments   : {', '.join(obs.get('attachments', [])) or 'None'}

Message:
{obs['body']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task   : {obs['task_id']}
Step   : {obs['step_number']} of {obs['max_steps']}

Respond with a JSON action object only."""


def call_llm(prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError:
        # Fallback safe action
        return {"close_ticket": True}
    except Exception as e:
        print(f"  [LLM Error] {e}", file=sys.stderr)
        return {"close_ticket": True}


# ─────────────────────────────────────────
#  EPISODE RUNNER
# ─────────────────────────────────────────

def run_episode(http: httpx.Client, task_id: str, max_steps: int) -> float:
    # Reset environment
    reset_resp = http.post("/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    cumulative_reward = 0.0

    for step_num in range(max_steps):
        prompt = build_prompt(obs)
        action = call_llm(prompt)

        step_resp = http.post("/step", json=action)
        step_resp.raise_for_status()
        result = step_resp.json()

        info = result.get("info", {})
        if "cumulative_reward" in info:
            cumulative_reward = info["cumulative_reward"]
        else:
            reward = result["reward"]["value"]
            cumulative_reward += reward
            
        done = result["done"]
        obs = result["observation"]

        if done:
            break
            
    # Guarantee bounds (0, 1) safely
    final_score = float(cumulative_reward)
    if final_score <= 0.0:
        final_score = 0.01
    elif final_score >= 1.0:
        final_score = 0.99

    return round(final_score, 4)


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────

def main(return_json: bool = False):
    tasks = [
        ("task_easy",   3),
        ("task_medium", 5),
        ("task_hard",   8),
    ]

    all_scores = {}

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as http:
        # Health check
        try:
            health = http.get("/health")
            health.raise_for_status()
        except Exception as e:
            print(f"ERROR: Environment not reachable at {BASE_URL}", file=sys.stderr)
            print(f"Make sure the server is running: uvicorn app.main:app --port 7860", file=sys.stderr)
            sys.exit(1)

        for task_id, max_steps in tasks:
            episode_scores = []

            if not return_json:
                print(f"\nRunning {task_id} ({EPISODES_PER_TASK} episodes)...")

            for ep in range(EPISODES_PER_TASK):
                score = run_episode(http, task_id, max_steps)
                episode_scores.append(score)

                if not return_json:
                    print(f"  Episode {ep + 1}: {score:.4f}")

            avg = round(sum(episode_scores) / len(episode_scores), 4)
            all_scores[task_id] = {
                "average_score": avg,
                "episode_scores": episode_scores,
                "episodes_run": EPISODES_PER_TASK,
            }

            if not return_json:
                print(f"  → Average: {avg:.4f}")

    if return_json:
        print(json.dumps(all_scores))
    else:
        overall = round(
            sum(v["average_score"] for v in all_scores.values()) / len(all_scores), 4
        )
        print(f"\n{'━'*40}")
        print(f"BASELINE RESULTS")
        print(f"{'━'*40}")
        for task_id, data in all_scores.items():
            print(f"  {task_id:<15} : {data['average_score']:.4f}")
        print(f"{'━'*40}")
        print(f"  {'OVERALL':<15} : {overall:.4f}")
        print(f"{'━'*40}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv Baseline Inference Script")
    parser.add_argument(
        "--return-json",
        action="store_true",
        help="Output results as JSON (used by /baseline endpoint)"
    )
    args = parser.parse_args()
    main(args.return_json)