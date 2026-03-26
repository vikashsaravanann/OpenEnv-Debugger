# Task: Full triage + quality response draft
def grade_hard(action, ticket, step):
    gt = ticket["ground_truth"]
    breakdown = {}

    # Triage score (same as medium)
    cat_score = 1.0 if action.category == gt["category"] else 0.0
    pri_score = 1.0 if action.priority == gt["priority"] else 0.0
    team_score = 1.0 if action.assigned_team == gt["assigned_team"] else 0.0
    triage_score = (cat_score * 0.15 + pri_score * 0.15 + team_score * 0.10)
    breakdown["triage"] = triage_score

    # Response quality scoring
    response_score = 0.0
    if action.response_draft:
        draft_lower = action.response_draft.lower()
        key_elements = gt.get("key_response_elements", [])
        hits = sum(1 for kw in key_elements if kw.lower() in draft_lower)
        response_score = hits / max(len(key_elements), 1)

        # Length check: penalize too short or form-letter-ish
        word_count = len(action.response_draft.split())
        if word_count < 20:
            response_score *= 0.5
        elif word_count > 300:
            response_score *= 0.8  # slight penalty for bloat

    breakdown["response_quality"] = response_score * 0.60

    # Escalation check
    escalate_correct = action.escalate == gt["should_escalate"]
    breakdown["escalation"] = 0.0 if not escalate_correct else 0.0  # bonus

    # Penalty: closing without a response
    if action.close_ticket and not action.response_draft:
        breakdown["close_penalty"] = -0.3

    total = max(-1.0, min(1.0, sum(breakdown.values())))
    reason = f"Triage={triage_score:.2f}, Response={response_score:.2f}"
    return round(total, 3), breakdown, reason