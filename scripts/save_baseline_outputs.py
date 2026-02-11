import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import json
import pandas as pd

from src.baselines.tfidf_baseline import tfidf_top_terms


def main():
    out_dir = project_root / "demo" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = project_root / "data" / "processed" / "processed_tokens.json"
    with open(tokens_path, "r", encoding="utf-8") as f:
        tokens = json.load(f)

    top = tfidf_top_terms(tokens, top_k=20)
    df = pd.DataFrame(top, columns=["term", "score"])

    out_path = out_dir / "tfidf_baseline_top_terms.csv"
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
