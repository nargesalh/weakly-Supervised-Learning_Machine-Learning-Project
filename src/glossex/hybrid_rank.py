import json, re, csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def load_docs(path="data/raw/economics_sample.txt"):
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    docs = [d.strip() for d in txt.splitlines() if d.strip()]
    return docs if len(docs) >= 5 else [txt]

def minmax(x):
    x = np.array(x, dtype=float)
    if len(x)==0: return x
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-9)

def main(alpha=0.8):
    ranked = json.load(open("data/processed/ranked_phrases.json", "r", encoding="utf-8"))
    terms = [norm(r["term"]) for r in ranked]
    emb_scores = np.array([float(r["score"]) for r in ranked], dtype=float)

    docs = load_docs()
    vec = TfidfVectorizer(lowercase=True, ngram_range=(2,3), min_df=2,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
                          stop_words="english")
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    scores = X.sum(axis=0).A1
    tfidf_map = {norm(vocab[i]): float(scores[i]) for i in range(len(vocab))}

    tfidf_scores = np.array([tfidf_map.get(t, 0.0) for t in terms], dtype=float)

    # normalize to comparable scale
    emb_n = minmax(emb_scores)
    tfidf_n = minmax(tfidf_scores)

    hybrid = alpha * tfidf_n + (1 - alpha) * emb_n

    order = np.argsort(-hybrid)
    out = [{"term": terms[i], "score": float(hybrid[i]), "tfidf": float(tfidf_scores[i]), "emb": float(emb_scores[i])} for i in order]

    json.dump(out, open("data/processed/ranked_hybrid.json", "w", encoding="utf-8"), indent=2)

    # write final_terms.csv used by eval
    with open("demo/outputs/final_terms.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["term","score"])
        w.writeheader()
        for r in out:
            w.writerow({"term": r["term"], "score": r["score"]})

    print(f"Saved hybrid ranking with alpha={alpha} -> demo/outputs/final_terms.csv")

if __name__ == "__main__":
    main()
