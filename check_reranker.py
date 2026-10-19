from src.v2.models.schemas import Evidence
from src.v2.rag.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker()

evidence = [
    Evidence(
        evidence_id="E1",
        document_id="doc1",
        chunk_id="chunk1",
        content=(
            "RAG retrieves relevant external documents "
            "to ground a language model's response."
        ),
    ),
    Evidence(
        evidence_id="E2",
        document_id="doc2",
        chunk_id="chunk1",
        content=(
            "Photosynthesis converts light energy into "
            "chemical energy in plants."
        ),
    ),
]

results = reranker.rerank(
    query="How does retrieval augmented generation use documents?",
    evidence=evidence,
    top_k=2,
)

for item in results:
    print(
        item.evidence_id,
        item.metadata.get("reranker_score"),
        item.content,
    )