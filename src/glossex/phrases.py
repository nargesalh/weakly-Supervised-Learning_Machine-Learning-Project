import json
import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


def load_corpus_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_phrases_tfidf(
    corpus_text: str,
    top_n: int = 5000,
    ngram_range: Tuple[int, int] = (2, 3),
    min_df: int = 2,
) -> List[str]:
    """
    Extract candidate terms/phrases using TF-IDF over n-grams.
    Returns a list of normalized candidate phrases.
    """
    # Split to pseudo-docs (lines) to make TF-IDF meaningful
    docs = [d.strip() for d in corpus_text.splitlines() if d.strip()]
    if len(docs) < 5:
        docs = [corpus_text]

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        stop_words="english",
    )
    X = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()
    scores = X.sum(axis=0).A1  # aggregate tf-idf across docs

    # Top N by score
    idx = scores.argsort()[::-1][:top_n]
    candidates = [normalize_space(terms[i]) for i in idx]

    # Remove too-short / junk
    clean = []
    for c in candidates:
        if len(c) < 3:
            continue
        # drop phrases that are only 1 very common short word
        if len(c.split()) == 1 and len(c) <= 3:
            continue
        clean.append(c)

    # unique keep order
    seen = set()
    out = []
    for c in clean:
        if c not in seen:
            out.append(c)
            seen.add(c)

    return out


def main():
    corpus = load_corpus_text("data/raw/economics_sample.txt")
    candidates = extract_phrases_tfidf(corpus, top_n=5000, ngram_range=(2, 3), min_df=2)

    with open("data/processed/phrase_candidates.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    print(f"Saved {len(candidates)} phrase candidates to data/processed/phrase_candidates.json")


if __name__ == "__main__":
    main()
