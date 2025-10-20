# Simple rule-based recommender (placeholder).
# Later, you can implement color harmony / category pairing rules or learn embeddings.
def suggest_outfit(predicted_category: str, predicted_color: str) -> str:
    if predicted_category.lower() in {"shirt", "top"}:
        return "Suggestion: Pair with neutral pants and white sneakers."
    if predicted_category.lower() in {"pants", "jeans"}:
        return "Suggestion: Pair with a plain tee and denim jacket."
    return "Suggestion: Add a complementary item and simple accessories."
