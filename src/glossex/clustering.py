import json
from typing import Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster_embeddings(
    embeddings: Dict[str, List[float]], n_clusters: int
) -> Dict[int, List[str]]:
    """
    Cluster embeddings using Agglomerative Clustering.
    Returns a mapping from cluster_id to list of tokens.
    """
    tokens = list(embeddings.keys())
    vectors = np.array([embeddings[t] for t in tokens])

    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(vectors)

    clusters = {}
    for token, label in zip(tokens, labels):
        clusters.setdefault(int(label), []).append(token)

    return clusters


def main():
    with open("data/processed/lemma_embeddings.json", "r") as f:
        embeddings = json.load(f)

    vocab_size = len(embeddings)
    n_clusters = max(2, vocab_size // 4)

    clusters = cluster_embeddings(embeddings, n_clusters)

    with open("data/processed/clusters.json", "w") as f:
        json.dump(clusters, f, indent=2)

    print(
        f"Clustering completed: {vocab_size} tokens "
        f"into {n_clusters} clusters."
    )


if __name__ == "__main__":
    main()
