import json
import numpy as np
from typing import Dict, List


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_seed_list(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def filter_clusters(
    clusters: Dict[str, List[str]],
    embeddings: Dict[str, List[float]],
    econ_seeds: List[str],
    general_seeds: List[str],
    margin: float = 0.02,          # هرچقدر بزرگ‌تر، سخت‌گیرانه‌تر
    top_n_clusters: int = 80,       # حداقل این تعداد خوشه را نگه می‌داریم
) -> Dict[str, List[str]]:
    scored = []  # (cid, score, tokens)

    for cid, tokens in clusters.items():
        econ_scores = []
        gen_scores = []

        for token in tokens:
            if token not in embeddings:
                continue
            v = np.array(embeddings[token])

            for s in econ_seeds:
                if s in embeddings:
                    econ_scores.append(cosine_similarity(v, np.array(embeddings[s])))

            for s in general_seeds:
                if s in embeddings:
                    gen_scores.append(cosine_similarity(v, np.array(embeddings[s])))

        # اگر یکی از لیست‌ها خالی بود، از این خوشه رد شو
        if not econ_scores or not gen_scores:
            continue

        econ_mean = float(np.mean(econ_scores))
        gen_mean = float(np.mean(gen_scores))
        score = econ_mean - gen_mean  # هرچی بزرگ‌تر، اقتصادی‌تر

        scored.append((cid, score, tokens))

    # مرتب‌سازی بر اساس score نزولی
    scored.sort(key=lambda x: x[1], reverse=True)

    filtered = {}

    # 1) اول خوشه‌هایی که از margin رد می‌شوند را نگه دار
    for cid, score, tokens in scored:
        if score >= margin:
            filtered[cid] = tokens

    # 2) اگر هنوز کم بود، top_n_clusters خوشه‌ی اول را هم اضافه کن
    if len(filtered) < top_n_clusters:
        for cid, score, tokens in scored[:top_n_clusters]:
            filtered[cid] = tokens

    return filtered



def main():
    with open("data/processed/clusters.json", "r") as f:
        clusters = json.load(f)

    with open("data/processed/lemma_embeddings.json", "r") as f:
        embeddings = json.load(f)

    econ_seeds = load_seed_list("data/seeds/economics.txt")
    general_seeds = load_seed_list("data/seeds/general.txt")

    filtered = filter_clusters(
    clusters, embeddings, econ_seeds, general_seeds,
    margin=0.02,
    top_n_clusters=80
)


    with open("data/processed/final_terms.json", "w") as f:
        json.dump(filtered, f, indent=2)

    print(
        f"Filtering completed. "
        f"{len(filtered)} clusters selected as economics-related."
    )


if __name__ == "__main__":
    main()
