from __future__ import annotations

from typing import Dict, Iterable, List

from bamboo_recognition.models.embedding import CLIPEmbedder
from bamboo_recognition.storage.milvus_store import MilvusStore


def build_part_vector_records(
    rows: Iterable[Dict],
    embedder: CLIPEmbedder,
) -> List[Dict]:
    records = list(rows)
    texts = [record["description"] for record in records]
    text_vectors = embedder.encode_texts(texts)

    for record, text_vector in zip(records, text_vectors):
        record["text_vector"] = text_vector.tolist()
        if "image_vector" not in record:
            record["image_vector"] = text_vector.tolist()
    return records


def ingest_records(rows: Iterable[Dict], store: MilvusStore, embedder: CLIPEmbedder) -> None:
    records = build_part_vector_records(rows, embedder)
    store.insert_part_vectors(records)
