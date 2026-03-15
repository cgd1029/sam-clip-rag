from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError:  # pragma: no cover - optional dependency for runtime
    SamAutomaticMaskGenerator = None
    sam_model_registry = {}


@dataclass(slots=True)
class SegmentedPart:
    image: Image.Image
    mask: np.ndarray
    bbox: List[int]


class SAMSegmenter:
    """Segment bamboo image into candidate parts with Segment Anything."""

    def __init__(self, model_type: str, checkpoint: str, device: str = "cpu") -> None:
        if SamAutomaticMaskGenerator is None:
            raise ImportError(
                "segment-anything is required for SAMSegmenter. "
                "Install it and ensure checkpoints are available."
            )
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        self._mask_generator = SamAutomaticMaskGenerator(sam)

    def segment(self, image: Image.Image) -> List[SegmentedPart]:
        image_np = np.array(image.convert("RGB"))
        masks = self._mask_generator.generate(image_np)
        segmented_parts: List[SegmentedPart] = []

        for mask_data in masks:
            mask = mask_data["segmentation"]
            x, y, w, h = mask_data["bbox"]
            crop = Image.fromarray(image_np[y : y + h, x : x + w]).copy()
            segmented_parts.append(SegmentedPart(image=crop, mask=mask, bbox=[x, y, w, h]))
        return segmented_parts
