# Experiments

This document describes the experiments conducted for the course project on automatic glossary extraction.

---

## Experimental Setup

- **Task**: Automatic extraction of domain-specific terms (glossary extraction)
- **Language**: English
- **Domain**: Economics
- **Approach**: Saliency-based candidate extraction + BERT embeddings + clustering + weak supervision

The goal of the experiments is **qualitative evaluation** of the proposed pipeline, since no labeled ground-truth glossary is available for the selected corpus.

---

## Dataset Used in Experiments

- Source: *Principles of Economics 3e* (OpenStax)
- License: CC BY 4.0
- Data handling:
  - A small English sample text was extracted for demonstration purposes.
  - Full dataset is referenced via download links (not included in the repository).

This setup allows fast, reproducible execution while preserving correctness of the pipeline.

---

## Experiment 1: Main Pipeline

### Configuration
- Tokenization: NLTK (English)
- Stopword removal: Enabled
- Embedding model: `bert-base-uncased`
- Clustering algorithm: Agglomerative Clustering
- Number of clusters: Heuristic based on number of candidate terms
- Supervision: Seed-list–based weak supervision

### Output
The pipeline produces:
- ranked salient candidate terms
- semantic clusters of terms
- filtered clusters corresponding to economics-related terminology

Results include terms such as:
*economics, markets, supply, demand, inflation, consumption, production*

---

## Experiment 2: Baseline Comparison

### Baseline Method
A simple TF-IDF–style baseline was implemented to provide a point of comparison:
- Term frequency computed on the domain corpus
- Penalization using general-language frequency (wordfreq Zipf scale)

### Comparison
- The baseline successfully identifies frequent domain terms.
- However, it does not group semantically related terms.
- It also lacks a mechanism for filtering irrelevant semantic clusters.

This highlights the advantage of the proposed pipeline over a frequency-based baseline.

---

## Evaluation Strategy

- **Type of evaluation**: Qualitative
- **Reason**: No gold-standard glossary or labeled dataset is available.
- **Criteria**:
  - semantic relevance of extracted terms
  - coherence of term clusters
  - comparison with baseline outputs

---

## Reproducibility

All experiments are reproducible using the provided scripts:

```bash
python src/glossex/preprocess.py
python src/glossex/embeddings.py
python src/glossex/clustering.py
python src/glossex/filtering.py
python src/baselines/tfidf_baseline.py
```

Demo-level outputs can be regenerated using:
```bash
python scripts/make_demo_outputs.py
```

## Phase-2 Results — Multi-K Evaluation

We report Top-K evaluation against the reference glossary (`data/gold_glossary.csv`) for K ∈ {50, 100, 200}.
Baseline is TF-IDF (`demo/outputs/tfidf_baseline_top_terms.csv`) and our model is the phrase-based weakly supervised ranking (`demo/outputs/final_terms.csv`).

| K | Model Precision | Model Recall | Model F1 | Baseline Precision | Baseline Recall | Baseline F1 |
|---|------------------|-------------|----------|--------------------|----------------|------------|
| 50  | 0.060 | 0.030 | 0.040 | 0.040 | 0.020 | 0.027 |
| 100 | 0.060 | 0.060 | 0.060 | 0.020 | 0.020 | 0.020 |
| 200 | 0.040 | 0.080 | 0.053 | 0.010 | 0.020 | 0.013 |

Observation: the proposed method consistently improves recall (and F1) as K increases, suggesting better semantic coverage compared to frequency-based ranking.

---

## Notes

- Experiments were run once per configuration due to the deterministic nature of the pipeline.
- Random seeds were fixed where applicable.
- The focus of the project is correct implementation and analysis rather than large-scale quantitative benchmarking.
