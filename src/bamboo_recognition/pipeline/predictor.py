from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from PIL import Image

from bamboo_recognition.config.settings import AppSettings, PART_CATEGORIES
from bamboo_recognition.models.embedding import CLIPEmbedder
from bamboo_recognition.models.segmentation import SAMSegmenter
from bamboo_recognition.retrieval.fusion import MultiPartFusion
from bamboo_recognition.retrieval.similarity import classify_part, top_label
from bamboo_recognition.storage.milvus_store import MilvusStore, SpeciesFilter


@dataclass(slots=True)
class PredictionRequest:
    image: Image.Image
    retrieval_mode: str = "image_to_text"
    top_k: int = 5
    rhizome_type: Optional[str] = None
    branch_type: Optional[str] = None


class BambooPredictor:
    """End-to-end bamboo species predictor."""

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        self.settings = settings or AppSettings()
        self.segmenter = SAMSegmenter(
            model_type=self.settings.model.sam_model_type,
            checkpoint=self.settings.model.sam_checkpoint,
            device=self.settings.model.device,
        )
        self.embedder = CLIPEmbedder(
            model_name=self.settings.model.clip_model_name,
            device=self.settings.model.device,
        )
        self.vector_store = MilvusStore(
            host=self.settings.milvus.host,
            port=self.settings.milvus.port,
            collection_name=self.settings.milvus.part_vector_collection,
        )
        self.fusion = MultiPartFusion(self.settings.retrieval.default_part_weights)
        self._part_label_embeddings = self.embedder.encode_texts(
            [f"a bamboo {part}" for part in PART_CATEGORIES]
        )

    def predict(self, request: PredictionRequest) -> Dict:
        self._validate_mode(request.retrieval_mode)
        segmented_parts = self.segmenter.segment(request.image)
        part_images = [part.image for part in segmented_parts]
        part_embeddings = self.embedder.encode_images(part_images)

        species_filter = SpeciesFilter(
            rhizome_type=request.rhizome_type,
            branch_type=request.branch_type,
        )
        all_hits: List[Dict] = []
        part_predictions: List[Dict] = []

        for segmented_part, embedding in zip(segmented_parts, part_embeddings):
            score_map = classify_part(embedding, self._part_label_embeddings, PART_CATEGORIES)
            part_label = top_label(score_map)
            part_predictions.append(
                {
                    "bbox": segmented_part.bbox,
                    "predicted_part": part_label,
                    "part_scores": score_map,
                }
            )
            hits = self.vector_store.search(
                vector=embedding.tolist(),
                mode=request.retrieval_mode,
                top_k=request.top_k,
                species_filter=species_filter,
                part=part_label,
            )
            all_hits.extend(hits)

        fused = self.fusion.fuse(all_hits)
        return {
            "parts": part_predictions,
            "species_predictions": fused[: request.top_k],
            "retrieval_mode": request.retrieval_mode,
        }

    def _validate_mode(self, mode: str) -> None:
        if mode not in self.settings.retrieval.supported_modes:
            supported = ", ".join(self.settings.retrieval.supported_modes)
            raise ValueError(f"Unsupported retrieval mode '{mode}'. Expected one of: {supported}")
