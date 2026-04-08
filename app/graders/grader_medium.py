def grade_medium(action, ticket, step):
    gt = ticket["ground_truth"]
    breakdown = {}

    PRIORITY_ORDER = ["low", "medium", "high", "critical"]

    # Category score (30%)
    cat_score = 1.0 if action.category == gt["category"] else 0.0
    if cat_score == 0.0 and action.category is not None:
        RELATED = {
            "billing": ["account"],
            "account": ["billing"],
            "technical": ["general"],
            "general": ["technical"],
            "shipping": [],
        }
        if action.category in RELATED.get(gt["category"], []):
            cat_score = 0.3
    breakdown["category"] = cat_score * 0.30

    # Priority score (40%) with partial credit
    pri_score = 0.0
    if action.priority == gt["priority"]:
        pri_score = 1.0
    elif action.priority is not None and gt["priority"] is not None:
        diff = abs(
            PRIORITY_ORDER.index(action.priority) -
            PRIORITY_ORDER.index(gt["priority"])
        )
        if diff == 1:
            pri_score = 0.5   # adjacent priority = partial credit
        elif diff == 2:
            pri_score = 0.2
    breakdown["priority"] = pri_score * 0.40

    # Team routing score (30%)
    team_score = 1.0 if action.assigned_team == gt["assigned_team"] else 0.0
    breakdown["team_routing"] = team_score * 0.30

    # Escalation bonus
    escalate_correct = action.escalate == gt["should_escalate"]
    breakdown["escalation_bonus"] = 0.05 if escalate_correct else -0.05

    # Step penalty
    step_penalty = max(0.0, (step - 1) * 0.03)
    breakdown["step_penalty"] = -step_penalty

    total = max(0.01, min(0.99, sum(breakdown.values())))
    reason = (
        f"Cat={cat_score:.1f}, Pri={pri_score:.1f}, "
        f"Team={team_score:.1f}, Escalate={'ok' if escalate_correct else 'wrong'}"
    )
    return round(total, 3), breakdown, reason