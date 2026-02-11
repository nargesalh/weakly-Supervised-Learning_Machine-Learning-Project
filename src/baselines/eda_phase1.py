import re
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# Paths
RAW_PATH = Path("data/raw/economics_sample.txt")
FIG_PATH = Path("results/figures")
FIG_PATH.mkdir(parents=True, exist_ok=True)

# Load text
text = RAW_PATH.read_text(encoding="utf-8", errors="ignore")

# Basic stats
words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
word_lengths = [len(w) for w in words]

# 1️⃣ Word length distribution
plt.figure()
plt.hist(word_lengths, bins=20)
plt.title("Word Length Distribution")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.savefig(FIG_PATH / "word_length_distribution.png")
plt.close()

# 2️⃣ Sentence length distribution
sentences = re.split(r"[.!?]+", text)
sentence_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]

plt.figure()
plt.hist(sentence_lengths, bins=20)
plt.title("Sentence Length Distribution")
plt.xlabel("Words per sentence")
plt.ylabel("Frequency")
plt.savefig(FIG_PATH / "sentence_length_distribution.png")
plt.close()

# 3️⃣ Top frequent words
counter = Counter(words)
most_common = counter.most_common(20)
labels, values = zip(*most_common)

plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.xticks(rotation=45)
plt.title("Top 20 Frequent Words")
plt.tight_layout()
plt.savefig(FIG_PATH / "top_20_words.png")
plt.close()

# 4️⃣ Zipf plot
freqs = sorted(counter.values(), reverse=True)
ranks = range(1, len(freqs) + 1)

plt.figure()
plt.loglog(ranks[:100], freqs[:100])
plt.title("Zipf Plot (Top 100)")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.savefig(FIG_PATH / "zipf_plot.png")
plt.close()

# 5️⃣ Vocabulary growth
unique_counts = []
seen = set()
for w in words:
    seen.add(w)
    unique_counts.append(len(seen))

plt.figure()
plt.plot(unique_counts)
plt.title("Vocabulary Growth")
plt.xlabel("Word index")
plt.ylabel("Unique words")
plt.savefig(FIG_PATH / "vocab_growth.png")
plt.close()

# 6️⃣ Token type ratio
alpha = sum(w.isalpha() for w in words)
numeric = sum(w.isnumeric() for w in words)

plt.figure()
plt.bar(["Alphabetic", "Numeric"], [alpha, numeric])
plt.title("Token Type Distribution")
plt.savefig(FIG_PATH / "token_type_distribution.png")
plt.close()

print("EDA completed. Figures saved to results/figures/")
