import csv
import pandas as pd


def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["term"].strip().lower() for row in reader if row.get("term")]


def load_gold(path):
    df = pd.read_csv(path)
    col = df.columns[0]
    return set(df[col].astype(str).str.strip().str.lower())


def evaluate_at_k(preds, gold, k):
    topk = preds[:k]
    correct = sum(1 for term in topk if term in gold)

    precision = correct / k if k > 0 else 0
    recall = correct / len(gold) if len(gold) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return precision, recall, f1


def main():
    model_preds = load_predictions("demo/outputs/final_terms.csv")
    baseline_preds = load_predictions("demo/outputs/tfidf_baseline_top_terms.csv")
    gold = load_gold("data/gold_glossary.csv")

    for k in [50, 100, 200]:
        print(f"\n=== Evaluation @ {k} ===")

        mp, mr, mf = evaluate_at_k(model_preds, gold, k)
        bp, br, bf = evaluate_at_k(baseline_preds, gold, k)

        print("\nModel:")
        print(f"Precision: {mp:.3f}")
        print(f"Recall:    {mr:.3f}")
        print(f"F1-score:  {mf:.3f}")

        print("\nBaseline (TF-IDF):")
        print(f"Precision: {bp:.3f}")
        print(f"Recall:    {br:.3f}")
        print(f"F1-score:  {bf:.3f}")


if __name__ == "__main__":
    main()
