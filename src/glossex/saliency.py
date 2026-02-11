from collections import Counter
from typing import List, Dict

from wordfreq import zipf_frequency
import math


def compute_saliency(tokens: List[str]) -> Dict[str, float]:
    """
    Compute a saliency score for each token based on its frequency
    in the domain corpus compared to general language frequency.

    Higher score => more domain-specific.
    """
    token_counts = Counter(tokens)
    total_tokens = sum(token_counts.values())

    saliency_scores = {}

    for token, freq in token_counts.items():
        domain_prob = freq / total_tokens

        # General language frequency (Zipf scale)
        general_freq = zipf_frequency(token, "en")

        # Convert Zipf frequency to probability-like value
        general_prob = 10 ** general_freq if general_freq > 0 else 1e-9

        # Saliency score (log-ratio)
        score = math.log(domain_prob / general_prob + 1e-9)
        saliency_scores[token] = score

    return saliency_scores
