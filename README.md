# Weakly Supervised Glossary Extraction for the Economics Domain

## Course: Machine Learning  
**Instructor:** Dr. Pishgoo 

### Team Members
- Narges Aliheydari  
- Fatemeh Gheysari  
- Alireza Farzaneh  

---

# 1. Project Overview

This project addresses the task of **automatic domain glossary extraction** from an unlabeled corpus in the Economics domain.

Given raw domain-specific text, the objective is to extract and rank terms and phrases that represent important domain terminology.

The system is implemented in two structured development phases:

- **Phase 1:** Baseline (TF-IDF + EDA + evaluation)
- **Phase 2:** Improvements (Phrase extraction + Weak supervision + Hybrid ranking)

Evaluation is performed against a reference glossary using Top-K metrics.

---

# 2. Problem Definition

The goal is to extract a ranked list of economic terms from a domain corpus without using labeled training data.

This is treated as a **weakly supervised terminology extraction task**, where supervision is limited to:

- A small list of economics seed words
- A small list of general (non-domain) seed words
- A reference gold glossary used only for evaluation

Output:
```
demo/outputs/final_terms.csv
```

Evaluation metrics:
- Precision@K
- Recall@K
- F1@K  
for K ∈ {50, 100, 200}

---

# 3. Dataset

### Corpus
`data/raw/economics_sample.txt`

Expanded domain text from economics sources.

### Gold Glossary
`data/gold_glossary.csv`

Contains 100 reference economics terms collected from Wikipedia.

---

# 4. Project Structure
```
glossex_final/
│
├── .git/
├── .venv/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── seeds/
│   └── gold_glossary.xlsx
│
├── demo/
│   ├── outputs/
│   └── demo.ipynb
│
├── docs/
│   └── phase-1-report.md
│
├── evaluation/
│
├── notebooks/
│   └── eda_exploration.ipynb
│
├── scripts/
│
├── src/
│   ├── baselines/
│   ├── evaluation/
│   ├── glossex/
│   └── utils/
│
├── README.md
├── requirements.txt
├── experiments.md
├── LICENSE
└── .gitignore
```


### Core Modules

| File | Description |
|------|------------|
| preprocess.py | Text preprocessing |
| embeddings.py | Token embedding generation |
| clustering.py | Initial clustering pipeline |
| tfidf_baseline.py | TF-IDF baseline |
| phrases.py | Phrase extraction (bigrams & trigrams) |
| rank_phrases.py | Seed-guided semantic ranking |
| hybrid_rank.py | Hybrid TF-IDF + embedding scoring |
| eval_topk.py | Multi-K evaluation |

---

# 5. Phase 1 — Baseline

Phase 1 includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- TF-IDF ranking baseline
- Evaluation using Top-K metrics

### Baseline Result (@K = 50)

| K | Precision | Recall | F1 |
|---|-----------|--------|----|
| 50 | 0.368 | 0.081 | 0.133 |

Observation:  
TF-IDF performs reasonably well due to frequency bias but struggles with multi-word glossary terms.

EDA figures are available in:
```
results/figures/
```

---

# 6. Phase 2 — Improvements

Phase 2 improves over the baseline using:

## 6.1 Phrase-Level Candidate Extraction
Instead of ranking individual tokens, we extract bigram and trigram phrases using TF-IDF.

Implementation:
```
src/glossex/phrases.py
```

---

## 6.2 Weak Supervision via Seed Words

Two small seed lists are used:

- `data/seeds/economics.txt`
- `data/seeds/general.txt`

Each phrase is ranked using:

score = similarity_to_economic_seeds − similarity_to_general_seeds

Implementation:
```
src/glossex/rank_phrases.py
```

---

## 6.3 Hybrid Scoring

To combine frequency and semantics:

Hybrid Score = α · TF-IDF + (1 − α) · Embedding Score

Implementation:
```
src/glossex/hybrid_rank.py
```

Final output:
```
demo/outputs/final_terms.csv
```

---

# 7. Phase 2 Results

Evaluation against the gold glossary:

| K | Model Precision | Model Recall | Model F1 | Baseline F1 |
|---|------------------|-------------|----------|------------|
| 50  | 0.060 | 0.030 | 0.040 | 0.027 |
| 100 | 0.060 | 0.060 | 0.060 | 0.020 |
| 200 | 0.040 | 0.080 | 0.053 | 0.013 |

Observation:  
The proposed method consistently improves recall and F1 compared to the TF-IDF baseline, especially at higher K values, indicating improved semantic coverage of domain terminology.

---

# 8. How to Run

## 1. Create Virtual Environment

```
python -m venv .venv
.venv\Scripts\activate
```

## 2. Install Dependencies

```
pip install -r requirements.txt
```

## 3. Run Baseline

```
python src/baselines/tfidf_baseline.py
```

## 4. Run Improved Model

```
python src/glossex/phrases.py
python src/glossex/rank_phrases.py
python src/glossex/hybrid_rank.py
python scripts/make_phrase_outputs.py
```

## 5. Evaluate

```
python evaluation/eval_topk.py
```

---

# 9. Key Contributions

- Phrase-level glossary extraction
- Weakly supervised ranking via seed words
- Hybrid frequency + semantic scoring
- Multi-K evaluation framework
- Structured phase-based Git workflow (5 commits per phase)

---

# 10. Limitations & Future Work

- Small corpus limits coverage
- Exact string matching underestimates performance
- Phrase embeddings use simple averaging

Future improvements:
- Larger corpus
- Contextual phrase embeddings
- Partial-match evaluation
- Morphological normalization

---

# 11. Academic Integrity

This project was implemented specifically for the Machine Learning course and follows the required structured phase-based development process.
