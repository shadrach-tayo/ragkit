from ragkit import RagConfig, RagPipeline, RetrievalResult


def test_retrieval_result_payloads() -> None:
    result = RetrievalResult(
        docs=["hello"],
        metadata=[{"source": "a.md", "page": 1}],
        rerank=[{"index": 0}],
        es_hits=[{"_id": "1"}],
        total=1,
        strategy="vector",
    )
    assert result.as_vector_payload()["docs"] == ["hello"]
    assert result.as_es_payload()["results"][0]["_id"] == "1"


def test_pipeline_constructs_without_backends() -> None:
    pipeline = RagPipeline(RagConfig(index_name="demo", rerank=False, strategy="vector"))
    assert pipeline.config.index_name == "demo"
