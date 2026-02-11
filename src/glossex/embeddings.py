import json
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def compute_embeddings(
    tokens: List[str], model_name: str = "distilbert-base-uncased"
) -> Dict[str, List[float]]:
    """
    Compute contextualized embeddings for each token using BERT.
    Each token is embedded independently (single-token context).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = {}

    with torch.no_grad():
        for token in tqdm(set(tokens), desc="Computing embeddings"):
            encoded = tokenizer(
                token,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            outputs = model(**encoded)
            vector = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            embeddings[token] = vector

    return embeddings


def main():
    with open("data/processed/processed_tokens.json", "r") as f:
        tokens = json.load(f)

    embeddings = compute_embeddings(tokens)

    with open("data/processed/lemma_embeddings.json", "w") as f:
        json.dump(embeddings, f)

    print(
        f"Computed embeddings for {len(embeddings)} tokens "
        f"and saved to data/processed/lemma_embeddings.json"
    )


if __name__ == "__main__":
    main()
