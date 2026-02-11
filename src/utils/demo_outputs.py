import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_top_saliency_table(saliency_scores: Dict[str, float], out_csv: str, top_k: int = 20):
    items = sorted(saliency_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    df = pd.DataFrame(items, columns=["term", "saliency_score"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def save_final_terms_table(final_terms_clusters: Dict[str, List[str]], out_csv: str):
    rows = []
    for cid, terms in final_terms_clusters.items():
        for t in terms:
            rows.append({"cluster_id": cid, "term": t})
    df = pd.DataFrame(rows).sort_values(["cluster_id", "term"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
