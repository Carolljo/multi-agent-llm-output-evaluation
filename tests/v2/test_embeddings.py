from src.v2.rag.embeddings import OllamaEmbeddingModel


def test_single_embedding():
    model = OllamaEmbeddingModel()

    embedding = model.embed(
        "Machine learning allows computers to learn from data."
    )

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(value, float) for value in embedding)


def test_batch_embeddings():
    model = OllamaEmbeddingModel()

    texts = [
        "Machine learning uses data.",
        "Retrieval augmented generation uses external evidence.",
    ]

    embeddings = model.embed_many(texts)

    assert len(embeddings) == len(texts)
    assert all(isinstance(vector, list) for vector in embeddings)
    assert all(len(vector) > 0 for vector in embeddings)