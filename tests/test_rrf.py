from ragkit.elasticsearch.search import reciprocal_rank_fuse


def test_reciprocal_rank_fuse_merges_lists() -> None:
    fused = reciprocal_rank_fuse(
        [
            [{"_id": "a"}, {"_id": "b"}],
            [{"_id": "b"}, {"_id": "c"}],
        ],
        rank_constant=60,
    )
    ids = [hit["_id"] for hit in fused]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}
