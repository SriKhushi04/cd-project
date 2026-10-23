# ASBUR — Sentence-Driven Multi-Interest and Behavior-Aware User Recommendation

ASBUR is a research prototype implementing sentence-driven user recommendation through:
- Sentence-BERT or optional Google Gemini embedding encoder (`text-embedding-004`)
- K-means multiple-interest user profiling
- Community activity behavior scoring
- Community prediction with similarity thresholding & low-confidence fallback diagnostics
- Adaptive semantic–behavioral score fusion
- IR Ranking metrics (Precision@K, Recall@K, nDCG@K, MRR)

---

## 1. Root Cause & Solution for Misclassification

### Root Cause
Previously, `infer_community` selected the community using a plain `max()` (argmax) over similarity scores. In a dataset with only `music`, `robotics`, and `machine-learning`, entering a sports query ("Who will win the football match?") produced low similarity scores to all three communities. However, because `max()` forces a selection from available labels, `music` was picked simply because its noise similarity (e.g., 0.15) was slightly higher than the others (0.12, 0.10).

### Solution
1. **Configurable Thresholding**: Added `min_community_similarity` (default `0.25`).
2. **Low-Confidence Diagnostic Fallback**: If the highest community similarity is below the threshold, `infer_community` returns `None` and outputs a diagnostic message indicating that query interests are unrepresented in the available data.
3. **Pure Semantic Matching Fallback**: When no community matches, the model sets community behavior scores to `0.0` and ranks users purely based on their semantic interest prototypes.
4. *Statistical Note*: Cosine similarity is a geometric distance metric rather than a calibrated probability distribution; thresholding serves as a domain boundary filter rather than a formal probability bound.

---

## 2. Setup & Installation

Recommended environment: Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Gemini Embedding API (Optional)

Google's Gemini API (`text-embedding-004`) can be evaluated as an alternative to the local Sentence-BERT encoder.

### Configuration & Security
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Set your API key inside `.env`:
   ```env
   GEMINI_API_KEY="your_actual_gemini_api_key_here"
   ```
   *Note: `.env` is listed in `.gitignore` and must never be committed to version control.*

3. Run demo using Gemini:
   ```bash
   python run_demo.py --encoder gemini
   ```

### Operational Considerations
- **Caching**: Local embeddings are cached in `.cache/gemini_embeddings/` to minimize repeated API calls, lower cost, and prevent rate-limiting.
- **Rate Limits & Failures**: Handles HTTP 429 rate limits gracefully with informative error messages.
- **Privacy**: User text is sent to an external API only when `--encoder gemini` is explicitly selected.
- **Costs & Quotas**: Embeddings consume API quota/credits based on character counts. Consult Google AI Studio pricing.
- **Embeddings vs. Training**: Using an embedding API extracts vector representations of text; it does not fine-tune or train the underlying model.

---

## 4. Running the Demo & Experiments

### Interactive Demo (Synthetic Smoke Test)
```bash
python run_demo.py
```
Supports repeated query prompts until typing `exit`. Prints community similarities, threshold decisions, and user ranking breakdowns.

### Custom / Real Dataset Demo
To run the demo on a local CSV dataset:
```bash
python run_demo.py --csv path/to/your_dataset.csv --min-sim 0.25
```
Data schema required:
- `user_id`: string
- `text`: string (post content)
- `timestamp`: date/time string
- `community`: string label (e.g., subreddit)

### Research Experiment Pipeline
```bash
python run_experiment.py --csv data/records.csv --train-before 2019-10-01 --test-from 2019-10-01 --test-until 2020-01-01
```

---

## 5. Running Automated Tests

Run the test suite covering music, robotics, machine-learning, and out-of-domain sports queries:

```bash
.venv\Scripts\python -m unittest discover -s tests
```

---

## 6. Research Integrity Notice

- **Synthetic Smoke Tests**: `run_demo.py` uses 7 synthetic posts from 4 users to verify pipeline execution. It is **not** experimental evidence.
- **Relevance Proxy**: Same-community participation in temporal splits serves as a weak relevance proxy, not ground-truth user friendship.
- **No Fabricated Claims**: No accuracy or performance metrics are reported until verified against real-world datasets.
