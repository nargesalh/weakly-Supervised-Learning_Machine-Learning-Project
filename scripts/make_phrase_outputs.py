import csv
import json

def main():
    with open("data/processed/ranked_phrases.json", "r", encoding="utf-8") as f:
        ranked = json.load(f)

    out_path = "demo/outputs/final_terms.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["term", "score"])
        w.writeheader()
        for row in ranked:
            w.writerow({"term": row["term"], "score": row["score"]})

    print(f"Saved: {out_path} ({len(ranked)} rows)")

if __name__ == "__main__":
    main()
