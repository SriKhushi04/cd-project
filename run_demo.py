import argparse
import os
import pandas as pd
from asbur.data import load_records
from asbur.model import ASBUR


def get_demo_df():
    """Lightweight synthetic dataset for smoke testing only; not research data."""
    rows = [
        ("u1", "I build neural networks for image classification.", "machine-learning"),
        ("u1", "Computer vision can support medical diagnosis.", "machine-learning"),
        ("u2", "I play guitar and listen to jazz music.", "music"),
        ("u2", "Jazz improvisation is fascinating.", "music"),
        ("u3", "I study robotics and autonomous navigation.", "robotics"),
        ("u3", "Computer vision is useful for mobile robots.", "robotics"),
        ("u4", "I read about healthcare and medical imaging.", "machine-learning"),
    ]
    df = pd.DataFrame(rows, columns=["user_id", "text", "community"])
    df["timestamp"] = pd.date_range("2019-01-01", periods=len(df), tz="UTC")
    return df


def main():
    parser = argparse.ArgumentParser(description="ASBUR Sentence-Driven User Recommendation Demo")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to a custom CSV dataset (columns: user_id, text, timestamp, community).",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="data/processed/train_embeddings.npy",
        help="Optional path to cached embeddings .npy file for real datasets.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Encoder name or 'gemini' for Google Gemini API.",
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.35,
        help="Minimum community cosine similarity threshold for community prediction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit on dataset for fast live demonstration (e.g. 5000).",
    )
    args = parser.parse_args()

    precomputed_vecs = None
    if args.csv:
        print(f"Loading user dataset from: {args.csv}")
        df = load_records(args.csv)
        print(f"Loaded {len(df):,} records with {df['user_id'].nunique():,} unique users across {df['community'].nunique()} communities.")
        if args.limit and args.limit < len(df):
            print(f"Limiting to first {args.limit} records for fast live demo...")
            df = df.head(args.limit).copy().reset_index(drop=True)
            if args.cache and os.path.exists(args.cache):
                import numpy as np
                try:
                    mmap_v = np.load(args.cache, mmap_mode="r")
                    if mmap_v.ndim == 2 and len(mmap_v) >= args.limit:
                        precomputed_vecs = np.array(mmap_v[:args.limit], dtype=np.float32)
                        print(f"Successfully sliced {args.limit} cached embeddings from {args.cache}.")
                    else:
                        print(f"Warning: Cache shape {mmap_v.shape} invalid or smaller than limit {args.limit}. Recomputing on-the-fly.")
                        precomputed_vecs = None
                except Exception as e:
                    print(f"Warning: Failed to load cached embeddings ({e}). Falling back to re-encoding.")
                    precomputed_vecs = None
            cache_path = None
        else:
            cache_path = args.cache if (args.cache and os.path.exists(args.cache)) else None
    else:
        print("=" * 60)
        print("NOTICE: No --csv supplied. Running in SYNTHETIC SMOKE-TEST MODE (7 posts, 4 users).")
        print("To run on real Reddit data, specify: --csv data/processed/reddit_train.csv")
        print("=" * 60)
        df = get_demo_df()
        cache_path = None

    model = ASBUR(
        encoder=args.encoder,
        max_prototypes=2,
        min_community_similarity=args.min_sim,
    ).fit(df, cache_path=cache_path, precomputed_vectors=precomputed_vecs)

    print("\n" + "=" * 60)
    print("ASBUR Interactive Recommendation Demo")
    print("Enter interest sentences to receive user recommendations.")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        try:
            query = input("\nEnter an interest sentence: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting ASBUR demo.")
            break

        if query.lower() == "exit":
            print("ASBUR demo closed.")
            break

        if not query:
            print("Please enter a non-empty sentence.")
            continue

        results = model.rank(query, top_k=5, debug=True)

        print("Recommended users:")
        for i, item in enumerate(results, 1):
            comm_str = item['inferred_community'] if item['inferred_community'] is not None else "None (Out-of-Domain)"
            print(
                f"{i}. {item['user_id']} | "
                f"score={item['score']:.3f} | "
                f"semantic={item['semantic_score']:.3f} | "
                f"behavior={item['behavior_score']:.3f} | "
                f"community={comm_str}"
            )


if __name__ == "__main__":
    main()
