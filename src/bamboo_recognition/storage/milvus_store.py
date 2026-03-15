from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility


@dataclass(slots=True)
class SpeciesFilter:
    rhizome_type: Optional[str] = None
    branch_type: Optional[str] = None


class MilvusStore:
    """Milvus operations for bamboo knowledge vectors."""

    def __init__(self, host: str, port: str, collection_name: str) -> None:
        self.collection_name = collection_name
        connections.connect(alias="default", host=host, port=port)
        self.collection = self._ensure_collection(collection_name)

    def _ensure_collection(self, name: str) -> Collection:
        if utility.has_collection(name):
            collection = Collection(name)
            collection.load()
            return collection

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="species_id", dtype=DataType.INT64),
            FieldSchema(name="species", dtype=DataType.VARCHAR, max_length=120),
            FieldSchema(name="part", dtype=DataType.VARCHAR, max_length=40),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="weight", dtype=DataType.FLOAT),
            FieldSchema(name="rhizome_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="branch_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=512),
        ]
        schema = CollectionSchema(fields=fields, description="Bamboo part vectors")
        collection = Collection(name=name, schema=schema)
        index_config = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 1024},
        }
        collection.create_index(field_name="text_vector", index_params=index_config)
        collection.create_index(field_name="image_vector", index_params=index_config)
        collection.load()
        return collection

    def insert_part_vectors(self, records: Iterable[Dict[str, Any]]) -> None:
        batch = list(records)
        if not batch:
            return
        columns = {key: [item[key] for item in batch] for key in batch[0].keys()}
        self.collection.insert(list(columns.values()))
        self.collection.flush()

    def search(
        self,
        vector: List[float],
        mode: str,
        top_k: int,
        species_filter: Optional[SpeciesFilter] = None,
        part: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        field_name = "text_vector" if mode == "image_to_text" else "image_vector"
        expr = self._build_expr(species_filter=species_filter, part=part)
        output_fields = [
            "id",
            "species_id",
            "species",
            "part",
            "description",
            "image_path",
            "weight",
            "rhizome_type",
            "branch_type",
        ]
        result = self.collection.search(
            data=[vector],
            anns_field=field_name,
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
        )
        hits = []
        for hit in result[0]:
            row = dict(hit.entity)
            row["score"] = float(hit.score)
            hits.append(row)
        return hits

    @staticmethod
    def _build_expr(species_filter: Optional[SpeciesFilter], part: Optional[str]) -> str:
        clauses = []
        if species_filter and species_filter.rhizome_type:
            clauses.append(f'rhizome_type == "{species_filter.rhizome_type}"')
        if species_filter and species_filter.branch_type:
            clauses.append(f'branch_type == "{species_filter.branch_type}"')
        if part:
            clauses.append(f'part == "{part}"')
        return " and ".join(clauses) if clauses else ""
