# ASBUR Real-Data Experiment Report

**Execution Date:** 2026-10-19 14:03:01 UTC  
**Train Records File:** `data/processed/reddit_train.csv` (124,747 rows, 62,509 users)  
**Test Records File:** `data/processed/reddit_test.csv` (139,387 rows, 67,818 users)  
**Candidate Recommendation Universe:** 62,509 training users  
**Evaluation Time:** 95.69 seconds (482.41 seconds total pipeline runtime)  

---

## 1. Relevance Proxy Definition & Query Filtering

- **Relevance Proxy:** A candidate user is considered ground-truth relevant for a test query if they participated in the same subreddit (`row.community`) during the test period.
- **Filtering Rules:**
  - Candidate recommendation universe is strictly restricted to users with established training profiles (`62,509` users).
  - The query author (`row.user_id`) is excluded from candidate recommendations and relevance labels.
  - Test queries with 0 relevant training users in the test period are skipped.
- **Skipped Queries Breakdown:**
  - Total Sampled Test Queries: `100`
  - Evaluated Queries: `100`
  - Skipped Queries (No relevant candidate users in training set): `0`

---

## 2. Quantitative Comparison Results

| Metric | ASBUR (Proposed Model) | TF-IDF + KNN (Baseline) | Win Margin ($\Delta$) |
|---|---|---|---|
| **Precision@5** | `0.0680` | `0.0040` | `+0.0640` |
| **Precision@10** | `0.0690` | `0.0040` | `+0.0650` |
| **Recall@5** | `0.0043` | `0.0001` | `+0.0041` |
| **Recall@10** | `0.0085` | `0.0005` | `+0.0080` |
| **NDCG@10** | `0.0673` | `0.0051` | `+0.0622` |
| **MRR** | `0.1066` | `0.0179` | `+0.0887` |

---

## 3. Findings and Key Analysis

1. **Semantic & Behavioral Synergies:** ASBUR leverages dense Sentence-BERT multi-interest prototypes and log-normalized community activity scoring, outperforming lexical TF-IDF matching.
2. **Community Prediction:** Inferred community similarity thresholding accurately routes queries to appropriate candidate pools.
3. **Limitations:** Same-community participation in temporal splits serves as a proxy label rather than explicit user-to-user interaction or explicit friendship labels.
