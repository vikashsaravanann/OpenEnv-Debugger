# Task: Correct category classification only
def grade_easy(action, ticket, step):
    gt = ticket["ground_truth"]
    breakdown = {}

    # Category match
    cat_score = 1.0 if action.category == gt["category"] else 0.0
    breakdown["category"] = cat_score

    # Small penalty for taking too many steps
    step_penalty = max(0, (step - 1) * 0.1)
    breakdown["step_penalty"] = -step_penalty

    total = max(0.0, cat_score - step_penalty)
    reason = f"Category: {'correct' if cat_score else 'wrong'}"
    return round(total, 3), breakdown, reason