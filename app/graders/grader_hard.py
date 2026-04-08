def grade_hard(action, ticket, step):
    gt = ticket["ground_truth"]
    breakdown = {}

    PRIORITY_ORDER = ["low", "medium", "high", "critical"]

    # --- TRIAGE SECTION (40% total) ---

    # Category (15%)
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
    breakdown["category"] = cat_score * 0.15

    # Priority (15%)
    pri_score = 0.0
    if action.priority == gt["priority"]:
        pri_score = 1.0
    elif action.priority is not None and gt["priority"] is not None:
        diff = abs(
            PRIORITY_ORDER.index(action.priority) -
            PRIORITY_ORDER.index(gt["priority"])
        )
        if diff == 1:
            pri_score = 0.5
        elif diff == 2:
            pri_score = 0.2
    breakdown["priority"] = pri_score * 0.15

    # Team routing (10%)
    team_score = 1.0 if action.assigned_team == gt["assigned_team"] else 0.0
    breakdown["team_routing"] = team_score * 0.10

    # --- RESPONSE QUALITY SECTION (60% total) ---
    response_score = 0.0

    if action.response_draft:
        draft_lower = action.response_draft.lower()
        key_elements = gt.get("key_response_elements", [])

        # Keyword coverage
        hits = sum(1 for kw in key_elements if kw.lower() in draft_lower)
        keyword_score = hits / max(len(key_elements), 1)

        # Length scoring
        word_count = len(action.response_draft.split())
        if word_count < 20:
            length_multiplier = 0.4   # too short
        elif word_count < 50:
            length_multiplier = 0.7   # acceptable
        elif word_count <= 200:
            length_multiplier = 1.0   # ideal
        else:
            length_multiplier = 0.85  # slight penalty for bloat

        # Professionalism check
        polite_words = ["thank", "sorry", "apolog", "understand", "help", "assist"]
        professionalism = 0.1 if any(w in draft_lower for w in polite_words) else 0.0

        response_score = (keyword_score * 0.8 + professionalism) * length_multiplier
    else:
        # No response draft = heavy penalty
        response_score = 0.0

    breakdown["response_quality"] = min(0.60, response_score * 0.60)

    # --- ESCALATION CHECK ---
    escalate_correct = action.escalate == gt["should_escalate"]
    breakdown["escalation"] = 0.05 if escalate_correct else -0.05

    # --- ADVANCED AGENT MECHANICS ---
    tool_bonus = 0.0
    if action.tags and any("fetch" in t or "query" in t for t in action.tags):
        tool_bonus = 0.10
    breakdown["tool_bonus"] = tool_bonus

    # --- PENALTIES ---
    # Closing without a response draft
    if action.close_ticket and not action.response_draft:
        breakdown["close_penalty"] = -0.30

    # Step penalty
    step_penalty = max(0.0, (step - 1) * 0.02)
    breakdown["step_penalty"] = -step_penalty

    total = max(0.01, min(0.99, sum(breakdown.values())))
    reason = (
        f"Triage={cat_score:.1f}/{pri_score:.1f}/{team_score:.1f}, "
        f"Response={response_score:.2f}, "
        f"Escalate={'ok' if escalate_correct else 'wrong'}"
    )
    return round(total, 3), breakdown, reason