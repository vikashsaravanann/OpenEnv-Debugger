def grade_easy(action, ticket, step):
    gt = ticket["ground_truth"]
    breakdown = {}

    # Category match — full score
    cat_score = 1.0 if action.category == gt["category"] else 0.0
    breakdown["category"] = cat_score

    # Partial credit: only 1 step away in logic
    if cat_score == 0.0 and action.category is not None:
        RELATED = {
            "billing": ["account"],
            "account": ["billing"],
            "technical": ["general"],
            "general": ["technical"],
            "shipping": [],
        }
        if action.category in RELATED.get(gt["category"], []):
            breakdown["category"] = 0.3

    # Small step penalty for taking too long
    step_penalty = max(0.0, (step - 1) * 0.05)
    breakdown["step_penalty"] = -step_penalty

    # Bonus for adding correct tags
    tag_bonus = 0.0
    if action.tags:
        tag_bonus = 0.1
    breakdown["tag_bonus"] = tag_bonus

    total = max(0.0, min(1.0, sum(breakdown.values())))
    reason = f"Category: {'correct' if cat_score == 1.0 else 'wrong'}, Step penalty: {step_penalty}"
    return round(total, 3), breakdown, reason