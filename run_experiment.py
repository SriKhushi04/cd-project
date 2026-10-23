import argparse, json, os
import pandas as pd
from asbur.data import load_records, temporal_split
from asbur.model import ASBUR
from asbur.metrics import ranking_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--train-before", required=True)
    ap.add_argument("--test-from", required=True)
    ap.add_argument("--test-until", required=True)
    ap.add_argument("--max-queries", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    df = load_records(args.csv)
    train, test = temporal_split(df, args.train_before, args.test_from, args.test_until)
    os.makedirs(args.out, exist_ok=True)
    print(f"Records={len(df)} train={len(train)} test={len(test)}")
    model = ASBUR().fit(train)
    queries = test.sample(n=min(args.max_queries, len(test)), random_state=42)
    metrics, saved = [], []
    for row in queries.itertuples(index=False):
        relevant = set(test.loc[test.community.astype(str) == str(row.community), "user_id"].astype(str))
        relevant.discard(str(row.user_id))
        if not relevant:
            continue
        ranked = model.rank(str(row.text), top_k=args.top_k, exclude_user=str(row.user_id))
        ids = [r["user_id"] for r in ranked]
        m = ranking_metrics(ids, relevant, args.top_k)
        metrics.append(m)
        saved.append({"query_text": row.text, "query_user": str(row.user_id),
                      "target_community_for_evaluation_only": str(row.community),
                      "ranked_user_ids": ids, **m})
    result = pd.DataFrame(metrics)
    if result.empty:
        raise RuntimeError("No evaluable queries with relevant users; inspect split and labels.")
    summary = {c: float(result[c].mean()) for c in result.columns}
    summary.update({"records": len(df), "train_records": len(train), "test_records": len(test),
                    "evaluated_queries": len(result),
                    "label_note": "same-community participation proxy, not ground-truth friendship"})
    with open(os.path.join(args.out, "asbur_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(saved).to_csv(os.path.join(args.out, "asbur_rankings.csv"), index=False)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
