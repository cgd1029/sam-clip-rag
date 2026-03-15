from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarity for rows of a against rows of b."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return np.matmul(a, b.T)


def classify_part(
    part_embedding: np.ndarray,
    category_embeddings: np.ndarray,
    categories: Iterable[str],
) -> Dict[str, float]:
    scores = cosine_similarity(part_embedding, category_embeddings)[0]
    return {category: float(score) for category, score in zip(categories, scores)}


def top_label(score_map: Dict[str, float]) -> str:
    return max(score_map, key=score_map.get)


def ranked_labels(score_map: Dict[str, float]) -> List[str]:
    return [k for k, _ in sorted(score_map.items(), key=lambda item: item[1], reverse=True)]
