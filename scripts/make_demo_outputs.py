import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.glossex.saliency import compute_saliency
from src.utils.demo_outputs import save_top_saliency_table, save_final_terms_table


def main():
    tokens_path = project_root / "data" / "processed" / "processed_tokens.json"
    final_terms_path = project_root / "data" / "processed" / "final_terms.json"
    out_dir = project_root / "demo" / "outputs"

    with open(tokens_path, "r", encoding="utf-8") as f:
        tokens = json.load(f)

    saliency = compute_saliency(tokens)
    df_sal = save_top_saliency_table(saliency, str(out_dir / "top_saliency_terms.csv"), top_k=20)

    with open(final_terms_path, "r", encoding="utf-8") as f:
        final_clusters = json.load(f)

    df_final = save_final_terms_table(final_clusters, str(out_dir / "final_terms.csv"))

    print("Saved:")
    print("-", out_dir / "top_saliency_terms.csv", f"({len(df_sal)} rows)")
    print("-", out_dir / "final_terms.csv", f"({len(df_final)} rows)")


if __name__ == "__main__":
    main()
