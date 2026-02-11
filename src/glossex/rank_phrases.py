import json
import re
import numpy as np
from typing import List


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_seed_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip().lower() for line in f if line.strip()]


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())



def mean_seed_sim(vec: np.ndarray, seeds: List[str], embeds: dict) -> float:
    sims = []
    for s in seeds:
        if s in embeds:
            sims.append(cosine_similarity(vec, np.array(embeds[s], dtype=float)))
    return float(np.mean(sims)) if sims else 0.0


def main():
    # token embeddings from your existing pipeline
    with open("data/processed/lemma_embeddings.json", "r", encoding="utf-8") as f:
        token_embeds = json.load(f)

    # phrase candidates from TF-IDF n-grams
    with open("data/processed/phrase_candidates.json", "r", encoding="utf-8") as f:
        phrases = json.load(f)

    econ_seeds = load_seed_list("data/seeds/economics.txt")
    general_seeds = load_seed_list("data/seeds/general.txt")

    ranked = []
    for ph in phrases:
        ph = normalize_space(ph)
        # drop phrases containing very short tokens
        if any(len(w) <= 2 for w in ph.split()):
            continue
        # drop phrases where all words look like verbs (-ing / -ed)
        if all(w.endswith(("ing", "ed")) for w in ph.split()):
            continue

        parts = [p for p in ph.split() if p in token_embeds]
        if not parts:
            continue

        v = np.mean([np.array(token_embeds[p], dtype=float) for p in parts], axis=0)

        econ = mean_seed_sim(v, econ_seeds, token_embeds)
        gen = mean_seed_sim(v, general_seeds, token_embeds)
        score = econ - gen  # economic-ness score

        ranked.append((ph, score, econ, gen))

    ranked.sort(key=lambda x: x[1], reverse=True)

    out = [{"term": t, "score": s, "econ": e, "gen": g} for (t, s, e, g) in ranked]

    with open("data/processed/ranked_phrases.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Ranked {len(out)} phrases -> data/processed/ranked_phrases.json")
    print("Top 10:")
    for row in out[:10]:
        print(row["term"], "score=", round(row["score"], 4))


if __name__ == "__main__":
    main()
