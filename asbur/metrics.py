import math

def ranking_metrics(ranked_ids, relevant_ids, max_k=10):
    ranked = list(ranked_ids)[:max_k]
    relevant = set(relevant_ids)
    if not relevant:
        return {
            "precision_at_5": 0.0,
            "precision_at_10": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "mrr": 0.0,
        }

    hits_5 = [int(u in relevant) for u in ranked[:5]]
    hits_10 = [int(u in relevant) for u in ranked[:10]]

    p_5 = sum(hits_5) / 5.0
    p_10 = sum(hits_10) / 10.0

    r_5 = sum(hits_5) / len(relevant) if len(relevant) > 0 else 0.0
    r_10 = sum(hits_10) / len(relevant) if len(relevant) > 0 else 0.0

    dcg_10 = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(hits_10))
    ideal_10 = [1] * min(len(relevant), 10) + [0] * max(0, 10 - len(relevant))
    idcg_10 = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_10))
    ndcg_10 = dcg_10 / idcg_10 if idcg_10 > 0 else 0.0

    first = next((i + 1 for i, hit in enumerate(hits_10) if hit), None)
    mrr = 1.0 / first if first else 0.0

    return {
        "precision_at_5": p_5,
        "precision_at_10": p_10,
        "recall_at_5": r_5,
        "recall_at_10": r_10,
        "ndcg_at_10": ndcg_10,
        "mrr": mrr,
    }
