Glossary Extraction for Economic Texts using NLP

This project implements a complete NLP-based pipeline for automatic extraction of domain-specific glossary terms from economic texts.
The goal is to identify meaningful economic concepts from raw text using representation learning, clustering, and semantic filtering, and to compare the proposed approach with a TF-IDF baseline.

📌 Problem Description

Glossary extraction is an important task in Natural Language Processing with applications in education, information retrieval, and domain-specific knowledge construction.
In this project, we focus on extracting economic terminology from textual corpora using machine learning techniques.

🧠 Method Overview

The proposed pipeline consists of the following steps:

Preprocessing

Tokenization and normalization of raw economic text

Removal of invalid or noisy tokens

Embedding Extraction

Use of a pretrained Transformer-based model (DistilBERT) to generate contextual embeddings for tokens

The model is used strictly as a feature extractor (no text generation)

Clustering

Grouping semantically similar tokens into clusters based on embedding similarity

Semantic Filtering

Identification of economically relevant clusters using seed-based cosine similarity

Comparison between economic seed words and general-purpose seed words

Tunable parameters to control the precision–coverage trade-off

Glossary Generation

Extraction of candidate economic terms from selected clusters

📊 Baseline Method

A TF-IDF-based baseline is implemented for comparison.
This baseline extracts high-frequency terms without using semantic embeddings or clustering.

📈 Evaluation

The extracted terms are evaluated against a manually curated gold glossary of economic terms using:

Precision@K

Recall@K

F1-score@K

Evaluations are reported for multiple values of K (50, 100, 200).
Light normalization is applied to reduce the effect of morphological and multi-word variations.

🔍 Key Observations

The TF-IDF baseline performs better on exact-match metrics due to lexical overlap with the gold glossary.

The proposed embedding-based method extracts more semantically rich and domain-specific terms, including multi-word expressions.

Increasing corpus size and tuning filtering parameters significantly improve coverage and stability.

Results highlight the limitations of exact-match evaluation for glossary extraction tasks.

🛠️ Project Structure
glossex_final/
│
├── data/
│   ├── raw/                 # Raw economic text corpus
│   ├── processed/           # Preprocessed tokens, embeddings, clusters
│   └── seeds/               # Economic and general seed word lists
│
├── src/glossex/
│   ├── preprocess.py
│   ├── embeddings.py
│   ├── clustering.py
│   └── filtering.py
│
├── scripts/
│   └── make_demo_outputs.py
│
├── evaluation/
│   └── eval_topk.py
│
└── demo/outputs/
    ├── final_terms.csv
    └── tfidf_baseline_top_terms.csv

🚀 How to Run

Activate virtual environment:

.venv\Scripts\activate


Run the pipeline:
python src/glossex/preprocess.py
python src/glossex/embeddings.py
python src/glossex/clustering.py
python src/glossex/filtering.py
python scripts/make_demo_outputs.py


Run evaluation:

python evaluation/eval_topk.py

📚 Course Information
Course: Machine Learning
Instructor: Dr. Pishgar
Group Members:
Narges Aliheydari
Fatemeh Gheisari
Alireza Farzaneh