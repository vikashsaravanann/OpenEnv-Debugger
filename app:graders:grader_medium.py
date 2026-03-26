# Task: Category + Priority + Team routing
def grade_medium(action, ticket, step):
    gt = ticket["ground_truth"]
    breakdown = {}

    cat_score = 1.0 if action.category == gt["category"] else 0.0
    pri_score = 1.0 if action.priority == gt["priority"] else 0.0
    team_score = 1.0 if action.assigned_team == gt["assigned_team"] else 0.0

    breakdown["category"] = cat_score * 0.3
    breakdown["priority"] = pri_score * 0.4
    breakdown["team_routing"] = team_score * 0.3

    # Partial credit: adjacent priority (e.g. high vs critical)
    PRIORITY_ORDER = ["low", "medium", "high", "critical"]
    if not pri_score and action.priority and gt["priority"]:
        diff = abs(PRIORITY_ORDER.index(action.priority) - 
                   PRIORITY_ORDER.index(gt["priority"]))
        if diff == 1:
            breakdown["priority"] = 0.2  # partial credit

    total = sum(breakdown.values())
    reason = f"Cat={cat_score}, Pri={pri_score}, Team={team_score}"
    return round(total, 3), breakdown, reason