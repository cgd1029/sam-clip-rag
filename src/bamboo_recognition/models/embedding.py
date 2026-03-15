from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbedder:
    """Wrapper around OpenAI CLIP for text and image embeddings."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @staticmethod
    def _normalize(embedding: torch.Tensor) -> torch.Tensor:
        return embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    def encode_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        image_list: List[Image.Image] = list(images)
        if not image_list:
            return np.empty((0, self.model.config.projection_dim), dtype=np.float32)

        inputs = self.processor(images=image_list, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        image_embeds = self._normalize(image_embeds)
        return image_embeds.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.empty((0, self.model.config.projection_dim), dtype=np.float32)

        inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
        text_embeds = self._normalize(text_embeds)
        return text_embeds.cpu().numpy().astype(np.float32)
