# ASBUR Processed Reddit Dataset

Generated on: 2026-10-10 12:49:40 UTC
Input Source: `data/reddit-corpus-small/`

## Files
- `reddit_records.csv`: Full cleaned dataset (264,134 rows).
- `reddit_train.csv`: Training split (124,747 rows, 2018-09-01 to 2018-09-14).
- `reddit_test.csv`: Testing split (139,387 rows, 2018-09-15 to 2018-09-30).

## Schema
All files contain the exact columns required by `asbur.data.load_records`:
- `user_id` (str): Anonymized Reddit username.
- `text` (str): Body text of the utterance (cleaned and normalized whitespace).
- `timestamp` (str): ISO 8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`).
- `community` (str): Subreddit label.

## Filtering Rules
1. Excluded empty text and whitespace-only utterances (4,482 records).
2. Excluded deleted/removed posts (`[deleted]`, `[removed]`) (257 records).
3. Excluded bot and system accounts (`AutoModerator`, `[deleted]`, etc.) (28,251 records).
4. Deduplicated exact duplicate `(user_id, text, timestamp, community)` records (8 records).

## Temporal Split Details
- **Cutoff Date:** `2018-09-15T00:00:00Z`
- **Training Period:** 2018-09-01 00:01:18Z to 2018-09-14 23:59:57Z (62,509 users, 100 communities).
- **Testing Period:** 2018-09-15 00:00:01Z to 2018-09-30 23:59:29Z (67,818 users, 100 communities).
- **Overlapping Users:** 12,815 users active in both splits (suitable for temporal evaluation).
