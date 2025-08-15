from modules.knowledge_processor import KnowledgeProcessor


def test_chunking_basic():
    kp = KnowledgeProcessor()
    sample_article = {
        "id": "kb_test",
        "title": "Sample Title",
        "content": " ".join([f"word{i}" for i in range(300)]),
        "category": "test",
        "tags": ["sample"],
        "priority": "low"
    }
    docs = kp._process_documents([sample_article])  # type: ignore (private ok for test)
    # Expect > 1 chunk
    assert len(docs) > 1
    # All chunks should reference parent & have total_chunks set consistently
    parent_ids = {d["parent_id"] for d in docs}
    assert parent_ids == {"kb_test"}
    total_chunks_values = {d["total_chunks"] for d in docs}
    assert len(total_chunks_values) == 1
    # Chunk indices should be sequential starting at 0
    indices = sorted(d["chunk_index"] for d in docs)
    assert indices == list(range(len(docs)))
