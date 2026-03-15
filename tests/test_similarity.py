import numpy as np

from bamboo_recognition.retrieval.similarity import classify_part, cosine_similarity, top_label


def test_cosine_similarity_matrix_multiply_for_normalized_embeddings():
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    sim = cosine_similarity(a, b)

    np.testing.assert_allclose(sim, np.eye(2), atol=1e-6)


def test_classify_part_and_top_label():
    categories = ["leaf", "culm"]
    part_embedding = np.array([0.9, 0.1], dtype=np.float32)
    category_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    score_map = classify_part(part_embedding, category_embeddings, categories)

    assert top_label(score_map) == "leaf"
