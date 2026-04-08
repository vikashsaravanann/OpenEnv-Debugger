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
- response_draft: string (draft a response if interacting or closing)
- escalate: true|false
- close_ticket: true|false (set to false if you are waiting for a customer reply)
- tags: list of strings (SPECIAL TOOLS: use "query_system_logs" to fetch server logs, or "fetch_billing_history" to fetch payment history. You will see the results in the next step's System Context).
Always set category, priority, and assigned_team. Output ONLY JSON."""

def build_prompt(obs):
    system_ctx = obs.get('system_context', '')
    sys_block = f"System Context: {system_ctx}\n" if system_ctx else ""
    return (
        f"Ticket ID: {obs.get('ticket_id', '')}\n"
        f"Subject: {obs.get('subject', '')}\n"
        f"Customer Tier: {obs.get('customer_tier', '')}\n"
        f"Sentiment: {obs.get('customer_sentiment', 'neutral')}\n"
        f"Message: {obs.get('body', '')}\n"
        f"{sys_block}"
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
    print(f"[START] task={task_id} env=OpenEnv-Debugger model={MODEL_NAME}", flush=True)
    
    rewards = []
    try:
        resp = requests.post(f"{ENV_API_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
        cumulative = 0.01  # Safe minimum
        done = False
        error_msg = "null"
        
        for step_n in range(1, max_steps + 1):
            if done:
                break
                
            action = call_llm(obs)
            action_str = json.dumps(action).replace(" ", "")
            
            try:
                step_resp = requests.post(f"{ENV_API_URL}/step", json=action, timeout=30)
                step_resp.raise_for_status()
                result = step_resp.json()
                
                info = result.get("info", {})
                if "cumulative_reward" in info:
                    cumulative = info["cumulative_reward"]
                else:
                    reward_dict = result.get("reward", {})
                    if isinstance(reward_dict, dict):
                        val = reward_dict.get("value", 0)
                    else:
                        val = float(reward_dict or 0)
                    cumulative = val
                
                # Fetch step reward for logging
                reward_dict = result.get("reward", {})
                if isinstance(reward_dict, dict):
                    step_reward = reward_dict.get("value", 0)
                else:
                    step_reward = float(reward_dict or 0)
                rewards.append(f"{step_reward:.2f}")

                done = result.get("done", False)
                obs = result.get("observation", obs)
                
                print(f"[STEP] step={step_n} action={action_str} reward={step_reward:.2f} done={str(done).lower()} error=null", flush=True)

            except Exception as e:
                error_msg = str(e).replace('\n', ' ')
                print(f"[STEP] step={step_n} action={action_str} reward=0.00 done=true error={error_msg}", flush=True)
                done = True
                rewards.append("0.00")
                break
            
        final = float(cumulative)
        if final <= 0.0:
            final = 0.01
        elif final >= 1.0:
            final = 0.99
            
        success = final >= 0.5
        rewards_joined = ",".join(rewards)
        print(f"[END] success={str(success).lower()} steps={len(rewards)} score={final:.2f} rewards={rewards_joined}", flush=True)
        return round(final, 4)
    except Exception as e:
        error_msg = str(e).replace('\n', ' ')
        print(f"[STEP] step=1 action={{}} reward=0.00 done=true error={error_msg}", flush=True)
        print(f"[END] success=false steps=1 score=0.01 rewards=0.00", flush=True)
        return 0.01

def main():
    tasks = [("task_easy", 3), ("task_medium", 5), ("task_hard", 8)]
    
    try:
        health = requests.get(f"{ENV_API_URL}/health", timeout=10)
        health.raise_for_status()
    except Exception as e:
        print(f"Cannot reach environment: {e}", file=sys.stderr)
        sys.exit(1)

    # We only run ONE episode per task to match the Hackathon parser logic which expects one [START]/[END] block per task execution.
    for task_id, max_steps in tasks:
        run_episode(task_id, max_steps)

if __name__ == "__main__":
    main()
