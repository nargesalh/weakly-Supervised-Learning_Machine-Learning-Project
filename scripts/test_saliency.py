import json
import sys
from pathlib import Path

# Add project root to PYTHONPATH
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.glossex.saliency import compute_saliency


with open("data/processed/processed_tokens.json", "r") as f:
    tokens = json.load(f)

scores = compute_saliency(tokens)

top_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top salient terms:")
for term, score in top_terms:
    print(term, round(score, 3))
