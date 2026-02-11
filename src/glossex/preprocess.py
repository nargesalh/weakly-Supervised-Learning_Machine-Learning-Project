import json
import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources (first run only)
for pkg in ["punkt_tab", "punkt", "stopwords"]:
    nltk.download(pkg, quiet=True)


def preprocess_text(text: str):
    """
    Tokenize text, lowercase tokens, and remove stopwords and non-alphabetic tokens.
    """
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha()]

    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    return tokens


def main():
    input_path = Path("data/raw/economics_sample.txt")
    output_path = Path("data/processed/processed_tokens.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = preprocess_text(text)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokens, f, indent=2)

    print(f"Preprocessing completed. {len(tokens)} tokens saved to {output_path}")


if __name__ == "__main__":
    main()
