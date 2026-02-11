import csv

K = 50  # top-K terms


def normalize(term: str) -> str:
    term = term.lower().strip()
    term = term.replace("-", " ")
    term = term.split()[0]          # فقط head word
    if term.endswith("s"):
        term = term[:-1]
    return term



def load_terms(path, term_col_candidates=("term", "Term", "token", "text")):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # اگر DictReader نتونه هدر بخونه، خطا می‌خوره؛ پس مطمئنیم CSV هدر دارد
        header = reader.fieldnames or []

        # ستون term را پیدا کن
        term_col = None
        for c in term_col_candidates:
            if c in header:
                term_col = c
                break

        # اگر پیدا نکرد، از اولین ستون استفاده کن (fallback)
        if term_col is None:
            term_col = header[0]

        terms = []
        for row in reader:
            val = normalize((row.get(term_col) or ""))
            if val:
                terms.append(val)
        return terms


def evaluate(predicted, gold):
    predicted_set = set(predicted)
    tp = len(predicted_set & gold)
    precision = tp / len(predicted_set) if predicted_set else 0
    recall = tp / len(gold) if gold else 0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0)
    return precision, recall, f1


gold_path = "data/gold_glossary.csv"
model_path = "demo/outputs/final_terms.csv"
baseline_path = "demo/outputs/tfidf_baseline_top_terms.csv"

gold_terms = set(load_terms(gold_path))
model_terms = load_terms(model_path)[:K]
baseline_terms = load_terms(baseline_path)[:K]

model_p, model_r, model_f1 = evaluate(model_terms, gold_terms)
base_p, base_r, base_f1 = evaluate(baseline_terms, gold_terms)

print(f"=== Evaluation @ {K} ===\n")

print("Model:")
print(f"Precision: {model_p:.3f}")
print(f"Recall:    {model_r:.3f}")
print(f"F1-score:  {model_f1:.3f}\n")

print("Baseline (TF-IDF):")
print(f"Precision: {base_p:.3f}")
print(f"Recall:    {base_r:.3f}")
print(f"F1-score:  {base_f1:.3f}")
