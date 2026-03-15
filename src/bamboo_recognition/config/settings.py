from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


PART_CATEGORIES = [
    "rhizome",
    "culm",
    "leaf",
    "sheath",
    "branch",
    "flower/fruit",
]


@dataclass(slots=True)
class ModelSettings:
    sam_model_type: str = "vit_h"
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"


@dataclass(slots=True)
class MilvusSettings:
    host: str = "localhost"
    port: str = "19530"
    db_name: str = "bamboo_db"
    part_vector_collection: str = "part_vectors"


@dataclass(slots=True)
class RetrievalSettings:
    top_k: int = 5
    default_part_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "rhizome": 1.0,
            "culm": 0.9,
            "leaf": 0.6,
            "sheath": 0.7,
            "branch": 0.8,
            "flower/fruit": 0.95,
        }
    )
    supported_modes: List[str] = field(default_factory=lambda: ["image_to_text", "image_to_image"])


@dataclass(slots=True)
class AppSettings:
    model: ModelSettings = field(default_factory=ModelSettings)
    milvus: MilvusSettings = field(default_factory=MilvusSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
