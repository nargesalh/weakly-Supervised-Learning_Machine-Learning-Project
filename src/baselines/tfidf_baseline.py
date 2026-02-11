import json
from collections import Counter
from typing import List, Tuple

import numpy as np


def tfidf_top_terms(tokens: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
    """
    Very simple TF-IDF-like baseline for a single document:
    - TF = term frequency in the domain corpus
    - IDF is approximated using general-language frequency (wordfreq Zipf scale)
      to penalize common words.
    This keeps the baseline lightweight and reproducible.
    """
    from wordfreq import zipf_frequency

    counts = Counter(tokens)
    total = sum(counts.values())

    scores = {}
    for term, c in counts.items():
        tf = c / total
        general_zipf = zipf_frequency(term, "en")
        # Higher Zipf => more common => lower idf
        idf_like = 1.0 / (1.0 + max(general_zipf, 0.0))
        scores[term] = tf * idf_like

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return top


def main():
    with open("data/processed/processed_tokens.json", "r", encoding="utf-8") as f:
        tokens = json.load(f)

    top = tfidf_top_terms(tokens, top_k=20)

    print("TF-IDF baseline (top 20):")
    for term, score in top:
        print(f"{term}\t{score:.6f}")


if __name__ == "__main__":
    main()
