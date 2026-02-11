# Phase 2 Report — Improvements (Weakly Supervised Phrase Glossary Extraction)

## 1. Goal
Phase 2 focuses on improving Phase 1 baseline (TF-IDF) by introducing **phrase-level candidates** and **weak supervision** via small seed lists, aiming to better capture multi-word economic terminology.

## 2. Motivation
The Phase 1 gold glossary contains many **multi-word terms** (e.g., "absolute advantage", "adaptive expectations").  
A token/unigram-only approach tends to surface frequent but non-terminological words and fails to align with glossary-style phrases.  
Therefore, we move from **token-level extraction** to **phrase-level extraction**.

## 3. Improvements Implemented

### 3.1 Phrase Candidate Extraction (TF-IDF n-grams)
We generate candidate glossary phrases from the corpus using TF-IDF over **bigrams and trigrams**:
- Extract top TF-IDF phrases as candidates (`data/processed/phrase_candidates.json`)
- This restricts the search space to glossary-like phrases and reduces noise from single-token function words.

**Implementation:** `src/glossex/phrases.py`

### 3.2 Seed-guided Semantic Ranking (Weak Supervision)
We apply weak supervision using two small seed sets:
- `data/seeds/economics.txt` (domain seeds)
- `data/seeds/general.txt` (non-domain/general seeds)

For each phrase, we compute an embedding by averaging token embeddings and rank phrases by:
\[
score = sim(phrase, econ\_seeds) - sim(phrase, general\_seeds)
\]
This encourages phrases that are semantically closer to economic seeds than general seeds.

**Implementation:** `src/glossex/rank_phrases.py`

### 3.3 Hybrid Scoring (TF-IDF + Embedding)
We combine the strengths of frequency-based ranking and semantic ranking using a hybrid score:
- TF-IDF captures frequent domain terms
- Embedding similarity captures semantic relatedness and improves coverage

The hybrid ranker outputs the final ranked list used for evaluation:
- `demo/outputs/final_terms.csv`

**Implementation:** `src/glossex/hybrid_rank.py`  
**Output generation:** `scripts/make_phrase_outputs.py`

## 4. Evaluation Setup
We evaluate extracted terms against the gold glossary (`data/gold_glossary.csv`) using Top-K metrics:
- Precision@K
- Recall@K
- F1@K  
for **K ∈ {50, 100, 200}**.

Baseline:
- TF-IDF ranking in `demo/outputs/tfidf_baseline_top_terms.csv`

Model:
- Phase-2 ranked phrases in `demo/outputs/final_terms.csv`

**Implementation:** `evaluation/eval_topk.py`

## 5. Results (Model vs Baseline)

| K | Model Precision | Model Recall | Model F1 | Baseline Precision | Baseline Recall | Baseline F1 |
|---|------------------|-------------|----------|--------------------|----------------|------------|
| 50  | 0.060 | 0.030 | 0.040 | 0.040 | 0.020 | 0.027 |
| 100 | 0.060 | 0.060 | 0.060 | 0.020 | 0.020 | 0.020 |
| 200 | 0.040 | 0.080 | 0.053 | 0.010 | 0.020 | 0.013 |

### Key observations
- The Phase-2 model consistently improves **recall and F1** over the TF-IDF baseline.
- Improvements become more visible at higher K, suggesting better **semantic coverage** of economic terminology.
- Precision remains relatively low due to small corpus size, limited seed lists, and strict matching against a 100-term gold glossary.

## 6. Discussion & Limitations
- **Gold glossary size (100)** limits maximum attainable recall and makes evaluation sensitive to exact string matching.
- The corpus is relatively small and may not contain many gold terms verbatim.
- The phrase embedding is computed as an average of token embeddings, which is simple but may miss phrase compositionality.

## 7. Future Work
- Expand corpus size using more economics sources (e.g., textbooks, Wikipedia sections, articles).
- Improve phrase representation (e.g., contextual embedding for full phrases).
- Add morphological/lemmatization consistency between corpus phrases and gold terms.
- Add manual evaluation or annotation to complement exact-match scoring.
