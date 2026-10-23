import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone


def prepare_reddit_data(
    raw_dir="data/reddit-corpus-small",
    out_dir="data/processed",
    split_date_str="2018-09-15T00:00:00Z",
):
    """
    Load ConvoKit Reddit Corpus, clean records, create chronological train/test split,
    validate for data leakage, and save output CSVs to data/processed/.
    """
    print(f"=== Starting Reddit Data Preparation ===")
    print(f"Input directory: {raw_dir}")
    print(f"Output directory: {out_dir}")

    utterances_path = os.path.join(raw_dir, "utterances.json")
    conversations_path = os.path.join(raw_dir, "conversations.json")

    if not os.path.exists(utterances_path) or not os.path.exists(conversations_path):
        raise FileNotFoundError(
            f"Missing dataset files in {raw_dir}. Required: utterances.json, conversations.json"
        )

    print("Loading utterances.json and conversations.json...")
    with open(utterances_path, "r", encoding="utf-8") as f:
        utterances = json.load(f)

    with open(conversations_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    total_raw = len(utterances)
    print(f"Total raw utterances loaded: {total_raw:,}")

    # Track filter counts for transparent validation
    cnt_empty_text = 0
    cnt_deleted_text = 0
    cnt_bot_user = 0
    cnt_missing_community = 0
    cnt_invalid_ts = 0

    valid_records = []
    known_bots = {"automoderator", "[deleted]", "[removed]", "none", "null", ""}

    for u in utterances:
        user_id = str(u.get("user") or "").strip()
        text = str(u.get("text") or "").strip()
        ts_val = u.get("timestamp")
        root_id = str(u.get("root") or "").strip()

        # Extract community from utterance meta or conversation root fallback
        meta = u.get("meta") or {}
        community = str(meta.get("subreddit") or "").strip()
        if not community and root_id in conversations:
            community = str(conversations[root_id].get("subreddit") or "").strip()

        # Filter 1: Bot / deleted user
        if not user_id or user_id.lower() in known_bots:
            cnt_bot_user += 1
            continue

        # Filter 2: Empty / deleted text
        if not text:
            cnt_empty_text += 1
            continue
        if text.lower() in ["[deleted]", "[removed]"]:
            cnt_deleted_text += 1
            continue

        # Filter 3: Missing community
        if not community:
            cnt_missing_community += 1
            continue

        # Filter 4: Invalid timestamp
        if ts_val is None or not isinstance(ts_val, (int, float)) or ts_val <= 0:
            cnt_invalid_ts += 1
            continue

        # Convert timestamp to ISO 8601 UTC string
        try:
            dt = datetime.fromtimestamp(ts_val, tz=timezone.utc)
            ts_iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            cnt_invalid_ts += 1
            continue

        # Normalize whitespace in text
        clean_text = " ".join(text.split())

        valid_records.append({
            "user_id": user_id,
            "text": clean_text,
            "timestamp": ts_iso,
            "community": community,
        })

    print("\n--- Filtering Summary ---")
    print(f"Raw utterances:                {total_raw:,}")
    print(f"Excluded (Bot / deleted user): {cnt_bot_user:,}")
    print(f"Excluded (Empty text):         {cnt_empty_text:,}")
    print(f"Excluded (Deleted/removed):    {cnt_deleted_text:,}")
    print(f"Excluded (Missing community):  {cnt_missing_community:,}")
    print(f"Excluded (Invalid timestamp):  {cnt_invalid_ts:,}")
    print(f"Valid clean records:           {len(valid_records):,}")

    # Build DataFrame
    df = pd.DataFrame(valid_records)

    # Exact deduplication of identical (user_id, text, timestamp, community) rows
    initial_len = len(df)
    df = df.drop_duplicates(subset=["user_id", "text", "timestamp", "community"]).reset_index(drop=True)
    dedup_removed = initial_len - len(df)
    print(f"Exact duplicates removed:      {dedup_removed:,}")
    print(f"Final clean dataset records:   {len(df):,}")

    # Parse timestamps to pd.DatetimeIndex for temporal splitting
    df["dt"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("dt").reset_index(drop=True)

    # Chronological Split
    split_boundary = pd.Timestamp(split_date_str)
    train_df = df[df["dt"] < split_boundary].copy()
    test_df = df[df["dt"] >= split_boundary].copy()

    # Drop temporary datetime column before saving
    train_df = train_df.drop(columns=["dt"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["dt"]).reset_index(drop=True)
    df_save = df.drop(columns=["dt"]).reset_index(drop=True)

    # Data Leakage & Validation Checks
    print("\n--- Data Validation & Leakage Checks ---")

    # 1. Missing values check
    assert df_save["user_id"].isnull().sum() == 0, "Validation Error: Null user_id found!"
    assert df_save["text"].isnull().sum() == 0, "Validation Error: Null text found!"
    assert df_save["timestamp"].isnull().sum() == 0, "Validation Error: Null timestamp found!"
    assert df_save["community"].isnull().sum() == 0, "Validation Error: Null community found!"
    print("PASS: Zero null values across all required columns.")

    # 2. Chronological separation check
    max_train_ts = train_df["timestamp"].max()
    min_test_ts = test_df["timestamp"].min()
    print(f"Train max timestamp: {max_train_ts}")
    print(f"Test min timestamp:  {min_test_ts}")
    assert max_train_ts < min_test_ts, (
        f"Data Leakage Error: Train max ({max_train_ts}) >= Test min ({min_test_ts})"
    )
    print("PASS: Strict temporal boundary maintained (max_train < min_test).")

    # 3. Exact row overlap check between train and test
    train_tuples = set(zip(train_df["user_id"], train_df["text"], train_df["timestamp"]))
    test_tuples = set(zip(test_df["user_id"], test_df["text"], test_df["timestamp"]))
    overlap_rows = train_tuples.intersection(test_tuples)
    assert len(overlap_rows) == 0, f"Data Leakage Error: {len(overlap_rows)} exact records found in both train and test!"
    print("PASS: Zero exact record overlap between train and test splits.")

    # Calculate User Overlap & Statistics
    train_users = set(train_df["user_id"])
    test_users = set(test_df["user_id"])
    overlapping_users = train_users.intersection(test_users)

    train_communities = set(train_df["community"])
    test_communities = set(test_df["community"])

    print("\n--- Dataset Statistics ---")
    print(f"All Records:   {len(df_save):>8,} rows | {len(set(df_save['user_id'])):>6,} users | {len(set(df_save['community'])):>3} communities | {df_save['timestamp'].min()} to {df_save['timestamp'].max()}")
    print(f"Train Split:   {len(train_df):>8,} rows | {len(train_users):>6,} users | {len(train_communities):>3} communities | {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Test Split:    {len(test_df):>8,} rows | {len(test_users):>6,} users | {len(test_communities):>3} communities | {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    print(f"User Overlap:  {len(overlapping_users):>8,} users active in BOTH train and test splits.")

    # Save Output Files
    os.makedirs(out_dir, exist_ok=True)

    records_file = os.path.join(out_dir, "reddit_records.csv")
    train_file = os.path.join(out_dir, "reddit_train.csv")
    test_file = os.path.join(out_dir, "reddit_test.csv")

    df_save.to_csv(records_file, index=False)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"\nSaved CSV files:")
    print(f"  - Full dataset: {records_file}")
    print(f"  - Train split:  {train_file}")
    print(f"  - Test split:   {test_file}")

    # Write documentation README under data/processed/
    readme_path = os.path.join(out_dir, "README.md")
    readme_content = f"""# ASBUR Processed Reddit Dataset

Generated on: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
Input Source: `data/reddit-corpus-small/`

## Files
- `reddit_records.csv`: Full cleaned dataset ({len(df_save):,} rows).
- `reddit_train.csv`: Training split ({len(train_df):,} rows, 2018-09-01 to 2018-09-14).
- `reddit_test.csv`: Testing split ({len(test_df):,} rows, 2018-09-15 to 2018-09-30).

## Schema
All files contain the exact columns required by `asbur.data.load_records`:
- `user_id` (str): Anonymized Reddit username.
- `text` (str): Body text of the utterance (cleaned and normalized whitespace).
- `timestamp` (str): ISO 8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`).
- `community` (str): Subreddit label.

## Filtering Rules
1. Excluded empty text and whitespace-only utterances ({cnt_empty_text:,} records).
2. Excluded deleted/removed posts (`[deleted]`, `[removed]`) ({cnt_deleted_text:,} records).
3. Excluded bot and system accounts (`AutoModerator`, `[deleted]`, etc.) ({cnt_bot_user:,} records).
4. Deduplicated exact duplicate `(user_id, text, timestamp, community)` records ({dedup_removed:,} records).

## Temporal Split Details
- **Cutoff Date:** `2018-09-15T00:00:00Z`
- **Training Period:** 2018-09-01 00:01:18Z to 2018-09-14 23:59:57Z ({len(train_users):,} users, {len(train_communities):,} communities).
- **Testing Period:** 2018-09-15 00:00:01Z to 2018-09-30 23:59:29Z ({len(test_users):,} users, {len(test_communities):,} communities).
- **Overlapping Users:** {len(overlapping_users):,} users active in both splits (suitable for temporal evaluation).
"""

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"  - Documentation: {readme_path}")
    print("=== Reddit Data Preparation Completed Successfully ===")


if __name__ == "__main__":
    prepare_reddit_data()

