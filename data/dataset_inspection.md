# Dataset Inspection Report: Reddit Corpus Small

**Inspection Date:** 2026-10-10  
**Dataset Path:** `data/reddit-corpus-small/`  
**Dataset Source Format:** Cornell ConvoKit Reddit Corpus  

---

## 1. Overview & File Contents

The `data/reddit-corpus-small/` directory contains 5 JSON files:

| File Name | File Size | Description |
|---|---|---|
| `corpus.json` | 99 bytes | High-level corpus metadata summary |
| `index.json` | 836 bytes | Field and metadata type specifications |
| `conversations.json` | 2.56 MB | Dictionary mapping thread root IDs to conversation metadata |
| `users.json` | 6.26 MB | Dictionary mapping usernames to aggregate post/comment counts |
| `utterances.json` | 187.24 MB | List of all post and comment utterances |

*Note: No standalone `README.md` was found in the dataset folder; `corpus.json` and `index.json` specify schema metadata.*

---

## 2. JSON Structure and Field Definitions

### `corpus.json`
Corpus-level counts:
- `subreddit`: `"reddit-corpus-small"`
- `num_posts`: `8286`
- `num_comments`: `288846`
- `num_user`: `119889`

### `conversations.json`
Dictionary keyed by thread root ID (e.g., `"9c716m"`):
- `title` (str): Thread title
- `num_comments` (int): Total comments in thread
- `domain` (str): Submission domain (e.g. `"self.singapore"`)
- `timestamp` (int): Unix epoch timestamp in seconds
- `subreddit` (str): Subreddit name (e.g. `"singapore"`)
- `gilded` (int): Total gildings
- `gildings` (dict): Detailed gilding counts
- `stickied` (bool): Sticky post flag
- `author_flair_text` (str/null): Author flair string

### `users.json`
Dictionary keyed by username (e.g., `"AutoModerator"`):
- `num_posts` (int): Submission count by user
- `num_comments` (int): Comment count by user

### `utterances.json`
Array of 297,132 utterance objects. Each object contains:
- `id` (str): Unique identifier for submission or comment
- `user` (str): Author username
- `root` (str): Thread root ID (foreign key to `conversations.json`)
- `reply_to` (str/null): Parent utterance ID (null for top-level submission)
- `timestamp` (int): UTC Unix epoch timestamp in seconds
- `text` (str): Body text of the utterance
- `meta` (dict): Additional attributes including `score`, `top_level_comment`, `retrieved_on`, `gilded`, `gildings`, `subreddit`, `stickied`, `permalink`, `author_flair_text`

---

## 3. Relationships and Structural Connections

- **Utterances $\rightarrow$ Users:** `utterance["user"]` maps directly to username keys in `users.json`.
- **Utterances $\rightarrow$ Conversations:** `utterance["root"]` maps to thread root keys in `conversations.json`.
- **Comment Tree Hierarchy:** `utterance["reply_to"]` points to parent utterance IDs (`null` for root submission).

---

## 4. Community / Subreddit Mapping

- **Location:** Subreddit information is present in `conversations.json[root]["subreddit"]` and `utterance["meta"]["subreddit"]`.
- **Consistency Verification:** Cross-verification confirmed **0 mismatches** across all 297,132 utterances. Community applies at the thread level, and every utterance inherits its conversation's subreddit.
- **Diversity:** The dataset contains **100 unique subreddits** (e.g., `pics`, `todayilearned`, `science`, `explainlikeimfive`, `worldnews`, `videos`, `askscience`, `technology`, `Music`, `news`, etc.).

---

## 5. Dataset Statistics

| Metric | Value |
|---|---|
| **Total Utterances** | 297,132 |
| **Valid Text Utterances** (non-empty, non-deleted, non-bot) | 264,142 |
| **Total Raw Users** | 119,889 |
| **Total Valid Human Users** | 117,512 |
| **Total Conversations (Threads)** | 8,286 |
| **Unique Subreddits / Communities** | 100 |
| **Timestamp Range** | 2018-09-01 00:01:18 UTC to 2018-09-30 23:59:29 UTC (30 days) |
| **Empty / Whitespace Text** | 4,526 utterances |
| **Deleted / Removed Text** (`[deleted]`, `[removed]`) | 25,868 utterances |
| **Total Invalid Utterances** | 30,394 (10.2%) |

### User Post Count Distribution (Valid Utterances)
- **Min:** 1 post
- **Max:** 188 posts
- **Mean:** 2.25 posts
- **Median:** 1.0 post
- **Users with $\ge 2$ valid posts:** 46,824 (39.8% of valid users)
- **Users with $\ge 5$ valid posts:** 11,755 (10.0%)
- **Users with $\ge 10$ valid posts:** 2,871 (2.4%)
- **Users with $\ge 20$ valid posts:** 580 (0.5%)

---

## 6. Temporal Train/Test Split Viability

- **Observation Period:** 30 days (September 2018).
- **Split Point Evaluated:** September 15, 2018 00:00:00 UTC.
  - Active users before Sept 15: 62,509
  - Active users after Sept 15: 67,818
  - **Users active in BOTH periods:** **12,815 users**.
- **Conclusion:** **Highly Viable.** Over 12,800 repeat users span both temporal windows, providing ample sample size for temporal recommendation evaluations.

---

## 7. Sample Anonymized Utterance Records

```json
{
  "id": "e58h...",
  "user": "user_anon_1",
  "root": "9c716m",
  "reply_to": "9c716m",
  "timestamp": 1535840100,
  "text": "What are your favorite local coffee spots near downtown?",
  "meta": {
    "subreddit": "singapore",
    "score": 5
  }
}
```

---

## 8. Limitations & Data Cleaning Requirements

1. **Noise Filtering:** 10.2% of utterances contain missing text, `[deleted]`, or `[removed]`. These must be filtered out.
2. **Bot Filtering:** Accounts such as `AutoModerator` and `[deleted]` must be excluded from user profiling.
3. **User Minimum Post Threshold:** 60.2% of users have only 1 post. For multi-interest prototype evaluation, users must be filtered to those with $\ge 2$ (or $\ge 3$) posts.

---

## 9. Recommended Next Steps

1. **Create Dataset Converter Script (`convert_reddit.py`):**
   Parse `utterances.json` and convert valid records to ASBUR standard CSV format: `user_id,text,timestamp,community`.
2. **Apply User Activity Filtering:**
   Filter to users with at least 2 or 3 valid posts to support interest clustering and evaluation.
3. **Run Temporal Experiment:**
   Execute `run_experiment.py` using a temporal split (e.g. Train: Sept 1–15, Test: Sept 15–30).

