# V2 Verified Answer Memory Redesign

## Implemented

- Added `src/v2/memory/` with a persistent ChromaDB verified-answer store.
- Added semantic answer-memory retrieval with configurable similarity threshold.
- Stores the validated reference answer, claims, evidence references, and supporting evidence snapshot.
- Only `ReferenceValidation.status == "valid"` can be persisted.
- Added memory-hit/memory-miss routing to the autonomous LangGraph workflow.
- Memory hits skip fresh evidence retrieval, reference generation, and reference validation.
- Memory misses use the existing RAG/reference pipeline and persist the validated result before evaluation.
- Added API fields: `memory_hit` and `memory_similarity`.
- Added Streamlit visibility for the memory path.
- Added `ANSWER_MEMORY_MIN_SIMILARITY` environment variable; default `0.85`.
- Added workflow and memory-focused tests.
- Updated API version to `2.1.0`.

## Important runtime test note

The uploaded ZIP does not contain the project's Windows virtual environment. The container used for static validation does not have the project's runtime dependencies (`langgraph`, `chromadb`, `groq`, etc.), so full pytest/runtime integration was not executed here.

Static compilation (`compileall`) and Pydantic schema round-trip validation passed.

## Next run in the project venv

```powershell
python -m pytest -q tests/v2/memory/test_answer_memory.py
python -m pytest -q tests/v2/test_autonomous_workflow.py
python -m pytest -q tests/v2
```

After those pass, run one controlled API test and verify the first request is a memory miss and the repeated semantically equivalent question is a memory hit.
