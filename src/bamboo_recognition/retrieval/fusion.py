from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List


class MultiPartFusion:
    """Fuse part-level retrieval scores into species-level scores."""

    def __init__(self, default_weights: Dict[str, float]) -> None:
        self.default_weights = default_weights

    def fuse(self, part_hits: Iterable[Dict]) -> List[Dict]:
        species_scores: Dict[str, float] = defaultdict(float)
        species_meta: Dict[str, Dict] = {}

        for hit in part_hits:
            part_name = hit["part"]
            similarity = hit["score"]
            weight = hit.get("weight") or self.default_weights.get(part_name, 1.0)
            species = hit["species"]
            species_scores[species] += weight * similarity
            species_meta[species] = {
                "species_id": hit["species_id"],
                "species": species,
                "rhizome_type": hit.get("rhizome_type"),
                "branch_type": hit.get("branch_type"),
            }

        ranked = sorted(species_scores.items(), key=lambda item: item[1], reverse=True)
        return [
            {
                **species_meta[species],
                "score": score,
            }
            for species, score in ranked
        ]
