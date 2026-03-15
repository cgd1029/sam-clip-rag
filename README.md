# sam-clip-rag

Multimodal bamboo species recognition prototype using **SAM + CLIP + Milvus + FastAPI**.

## Features

- **Image segmentation** with Segment Anything (SAM).
- **Part semantic identification** with CLIP over bamboo part classes:
  - rhizome
  - culm
  - leaf
  - sheath
  - branch
  - flower/fruit
- **Milvus vector retrieval** with two modes:
  - image-to-text (query image embeddings vs `text_vector`)
  - image-to-image (query image embeddings vs `image_vector`)
- **Prior botanical filtering** by:
  - `rhizome_type`
  - `branch_type`
- **Multi-part fusion**:
  - `score = Σ(weight_i × similarity_i)`
- **Top-K species predictions** output.

## Project structure

```text
src/bamboo_recognition/
  api/app.py                 # FastAPI entrypoint
  config/settings.py         # App/model/retrieval settings
  models/
    segmentation.py          # SAM segmentation wrapper
    embedding.py             # CLIP embedder wrapper
  storage/milvus_store.py    # Milvus schema + insert/search + filtering
  retrieval/
    similarity.py            # cosine similarity + part classification
    fusion.py                # weighted multi-part score fusion
  pipeline/predictor.py      # end-to-end inference pipeline
  utils/ingest.py            # helper for vector ingestion
tests/
  test_fusion.py
  test_similarity.py
```

## Database schema mapping

The Milvus collection `part_vectors` stores core fields corresponding to the requested schema:

- `id`
- `species_id`
- `species`
- `part`
- `description`
- `image_path`
- `weight`
- `rhizome_type`
- `branch_type`
- `text_vector`
- `image_vector`

> Note: Milvus is used as the vector store. A relational `species` table can be mirrored in SQL if needed, while `rhizome_type` and `branch_type` are duplicated in vector records for efficient pre-filtering.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

For SAM runtime, install Meta's `segment-anything` package and download a checkpoint such as `sam_vit_h_4b8939.pth`.

## Run API

```bash
uvicorn bamboo_recognition.api.app:app --reload
```

## Example prediction request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "image=@sample_bamboo.jpg" \
  -F "retrieval_mode=image_to_text" \
  -F "top_k=5" \
  -F "rhizome_type=sympodial" \
  -F "branch_type=2-branch"
```

## Core pipeline flow

1. Segment image into candidate parts with SAM.
2. Encode each part image with CLIP.
3. Classify each segment into bamboo part categories using CLIP similarity to part prompts.
4. Apply prior filter (`rhizome_type`, `branch_type`) in Milvus expression.
5. Retrieve nearest vectors (`COSINE`) from text or image vector field.
6. Fuse part-level evidence with weighted scoring.
7. Return top-K species.

