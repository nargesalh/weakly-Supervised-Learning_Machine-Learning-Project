# Phase 1 Report — Glossary Extraction (Weakly Supervised)

## 1. Problem Definition
The goal of this project is **domain glossary extraction**: given an unlabeled domain corpus (economics text), we aim to extract a ranked list of **domain-relevant terms and phrases**.  
This is approached as a **weakly supervised** task using only small seed lists (economics vs. general) and an external reference glossary for evaluation.

**Output:** a ranked list of candidate glossary terms saved as `demo/outputs/final_terms.csv`.

## 2. Dataset
- **Corpus:** `data/raw/economics_sample.txt` (economics domain text; expanded to a larger file during development).
- **Gold glossary (reference):** `data/gold_glossary.csv` (100 economics terms/phrases collected from Wikipedia).

## 3. Preprocessing
The pipeline normalizes and tokenizes the corpus to create processed tokens used by downstream models:
- Lowercasing
- Basic token cleaning
- Saving processed tokens to `data/processed/processed_tokens.json`

## 4. Exploratory Data Analysis (EDA)
We generated EDA plots to understand corpus statistics, including:
- Word length distribution
- Sentence length distribution
- Top frequent words
- Zipf plot (rank-frequency)
- Vocabulary growth
- Token type distribution

All figures are saved in: `results/figures/`.

## 5. Baseline Model (TF-IDF)
As a baseline, we rank candidate terms using TF-IDF scoring.

### Metric
We evaluate extracted terms against the gold glossary using Top-K overlap metrics:
- **Precision@K**
- **Recall@K**
- **F1@K**

### Baseline Result (@K=50)
| K | Precision | Recall | F1 |
|---|-----------|--------|----|
| 50 | 0.368 | 0.081 | 0.133 |

The TF-IDF baseline performs strongly in this small domain corpus due to frequency-based ranking.  
However, the gold glossary contains many **multi-word** phrases, which motivates phrase-level modeling in Phase 2.

## 6. Phase 2 Experiment Plan (Preview)
In Phase 2, we will implement and compare:
1. **Phrase extraction (n-grams)** using TF-IDF with bigrams/trigrams  
2. **Seed-guided semantic ranking** using embedding similarity (economics vs general seeds)  
3. **Hybrid scoring** combining TF-IDF and embedding scores  
4. Compare results at **K = 50, 100, 200** and analyze precision/recall trade-offs

## 7. Reproducibility
Key scripts:
- EDA: `src/baselines/eda_phase1.py`
- Baseline: `src/baselines/tfidf_baseline.py`
- Evaluation: `evaluation/eval_topk.py`

To run evaluation:
```bash
python evaluation/eval_topk.py
