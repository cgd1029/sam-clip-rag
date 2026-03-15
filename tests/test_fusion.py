from bamboo_recognition.retrieval.fusion import MultiPartFusion


def test_fusion_weighted_scoring_and_ranking():
    fusion = MultiPartFusion(default_weights={"leaf": 0.5, "culm": 1.0})
    part_hits = [
        {"species_id": 1, "species": "A", "part": "leaf", "score": 0.8, "weight": None},
        {"species_id": 2, "species": "B", "part": "leaf", "score": 0.7, "weight": None},
        {"species_id": 1, "species": "A", "part": "culm", "score": 0.9, "weight": None},
    ]

    fused = fusion.fuse(part_hits)

    assert fused[0]["species"] == "A"
    assert fused[0]["score"] == 1.3
    assert fused[1]["species"] == "B"
