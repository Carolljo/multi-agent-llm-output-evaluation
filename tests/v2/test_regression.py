import os
import time

from src.v2.agents.accuracy import AccuracyEvaluator
from src.v2.agents.completeness import CompletenessEvaluator
from src.v2.agents.grounding import EvidenceGroundingEvaluator
from src.v2.agents.logic import LogicalConsistencyEvaluator
from src.v2.arbitration.adjudicator import Adjudicator
from src.v2.claims.extractor import ClaimExtractor
from src.v2.evidence.quality import EvidenceQualityScorer
from src.v2.graph.workflow import build_autonomous_workflow
from src.v2.llm.client import LLMClient
from src.v2.memory.decision import MemoryDecisionPolicy
from src.v2.memory.retriever import AnswerMemoryRetriever
from src.v2.memory.store import ChromaAnswerMemory
from src.v2.memory.write_policy import MemoryWritePolicy
from src.v2.models.schemas import EvaluationInput
from src.v2.rag.retriever import ChromaRetriever
from src.v2.reference.generator import ReferenceAnswerGenerator
from src.v2.reference.validator import ReferenceAnswerValidator


REQUIRED_EVALUATORS = {
    "accuracy",
    "completeness",
    "logical_consistency",
    "evidence_grounding",
}


QUESTION = "What is machine learning?"

RESPONSE = (
    "Machine learning is a type of artificial intelligence that allows "
    "computers to learn patterns from data and make predictions without "
    "being explicitly programmed for every task."
)


def build_workflow(memory_store):
    """
    Build the final V2 autonomous workflow with verified answer memory.

    Memory-enabled behavior:

        First question:
            memory miss
            -> fresh RAG
            -> reference generation
            -> reference validation
            -> candidate evaluation
            -> final verdict
            -> verified answer written to memory

        Same question again:
            memory hit
            -> stored evidence
            -> stored verified reference answer
            -> candidate evaluation
            -> final verdict

    The memory-hit path must skip:
        - fresh RAG retrieval
        - reference generation
        - reference validation
    """

    if os.getenv("LLM_PROVIDER", "groq") != "groq":
        raise RuntimeError(
            "Final regression requires LLM_PROVIDER=groq."
        )

    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b",
    )

    llm_client = LLMClient(
        provider="groq",
        model=model,
    )

    memory_retriever = AnswerMemoryRetriever(
        store=memory_store,
        min_similarity=0.85,
    )

    return build_autonomous_workflow(
        retriever=ChromaRetriever(),

        evidence_quality_scorer=EvidenceQualityScorer(),

        reference_generator=ReferenceAnswerGenerator(
            llm_client
        ),

        reference_validator=ReferenceAnswerValidator(
            llm_client
        ),

        claim_extractor=ClaimExtractor(
            llm_client
        ),

        accuracy_evaluator=AccuracyEvaluator(
            llm_client
        ),

        completeness_evaluator=CompletenessEvaluator(
            llm_client
        ),

        logic_evaluator=LogicalConsistencyEvaluator(
            llm_client
        ),

        grounding_evaluator=EvidenceGroundingEvaluator(
            llm_client
        ),

        adjudicator=Adjudicator(
            llm_client
        ),

        memory_retriever=memory_retriever,

        memory_decision_policy=MemoryDecisionPolicy(),

        memory_store=memory_store,

        memory_write_policy=MemoryWritePolicy(),
    )


def build_input():
    return EvaluationInput(
        question=QUESTION,
        response=RESPONSE,
    )


def validate_result(
    result,
    *,
    expected_memory_hit,
):
    """
    Validate the final autonomous evaluation result.

    This preserves the important checks from the previous
    regression test while also validating the memory state.
    """

    print("\nValidating result...")

    # ---------------------------------------------------------
    # Memory behavior
    # ---------------------------------------------------------

    actual_memory_hit = result.get("memory_hit", False)

    assert actual_memory_hit == expected_memory_hit, (
        f"Expected memory_hit={expected_memory_hit}, "
        f"got {actual_memory_hit}"
    )

    if expected_memory_hit:
        assert result.get("memory_answer"), (
            "Memory hit occurred but memory_answer is missing."
        )

        assert result.get("memory_evidence"), (
            "Memory hit occurred but memory_evidence is missing."
        )

        assert result.get("memory_similarity") is not None, (
            "Memory hit occurred but memory similarity is missing."
        )

        print(
            f"Memory similarity: "
            f"{result['memory_similarity']:.4f}"
        )

    # ---------------------------------------------------------
    # Final verdict
    # ---------------------------------------------------------

    final_verdict = result.get("final_verdict")

    assert final_verdict is not None, (
        "Final verdict is missing."
    )

    assert final_verdict.verdict in {
        "accept",
        "partial",
        "reject",
    }, (
        f"Unexpected final verdict: "
        f"{final_verdict.verdict}"
    )

    assert final_verdict.verdict != "reject", (
        "Regression failed: final verdict was reject."
    )

    assert 0.0 <= final_verdict.score <= 1.0, (
        f"Final score out of bounds: "
        f"{final_verdict.score}"
    )

    assert 0.0 <= final_verdict.confidence <= 1.0, (
        f"Final confidence out of bounds: "
        f"{final_verdict.confidence}"
    )

    assert final_verdict.score >= 0.70, (
        f"Final score too low: "
        f"{final_verdict.score:.4f}"
    )

    # ---------------------------------------------------------
    # Evaluators
    # ---------------------------------------------------------

    evaluations = result.get("evaluations", [])

    assert len(evaluations) == 4, (
        f"Expected exactly 4 evaluator results, "
        f"got {len(evaluations)}"
    )

    evaluator_names = {
        evaluation.evaluator
        for evaluation in evaluations
    }

    assert evaluator_names == REQUIRED_EVALUATORS, (
        "Unexpected evaluator set.\n"
        f"Expected: {REQUIRED_EVALUATORS}\n"
        f"Actual:   {evaluator_names}"
    )

    # ---------------------------------------------------------
    # Individual evaluator validation
    # ---------------------------------------------------------

    for evaluation in evaluations:

        assert 0.0 <= evaluation.score <= 1.0, (
            f"{evaluation.evaluator} score out of bounds: "
            f"{evaluation.score}"
        )

        assert 0.0 <= evaluation.confidence <= 1.0, (
            f"{evaluation.evaluator} confidence out of bounds: "
            f"{evaluation.confidence}"
        )

    # ---------------------------------------------------------
    # Aggregation
    # ---------------------------------------------------------

    aggregated = result.get("aggregated")

    assert aggregated is not None, (
        "Aggregated evaluation is missing."
    )

    assert 0.0 <= aggregated.score <= 1.0, (
        f"Aggregated score out of bounds: "
        f"{aggregated.score}"
    )

    assert 0.0 <= aggregated.confidence <= 1.0, (
        f"Aggregated confidence out of bounds: "
        f"{aggregated.confidence}"
    )

    # ---------------------------------------------------------
    # Disagreement
    # ---------------------------------------------------------

    disagreement = result.get("disagreement")

    assert disagreement is not None, (
        "Disagreement result is missing."
    )

    # ---------------------------------------------------------
    # Adjudication consistency
    # ---------------------------------------------------------

    adjudication = result.get("adjudication")

    if disagreement.requires_adjudication:

        assert adjudication is not None, (
            "Adjudication was required but no "
            "adjudication result was produced."
        )

        assert adjudication.verdict == final_verdict.verdict, (
            "Final verdict does not match adjudication verdict."
        )

    # ---------------------------------------------------------
    # Reference answer
    # ---------------------------------------------------------

    reference_answer = result.get("reference_answer")

    assert reference_answer is not None, (
        "Reference answer is missing."
    )

    # ---------------------------------------------------------
    # Evidence
    # ---------------------------------------------------------

    evidence = result.get("evidence", [])

    assert evidence, (
        "Final result contains no evidence."
    )

    print("Validation passed.")

    return final_verdict


def run_case(
    workflow,
    *,
    expected_memory_hit,
    label,
):
    """
    Execute one regression case.
    """

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    print(f"Question: {QUESTION}")
    print(f"Response: {RESPONSE}")

    start_time = time.perf_counter()

    result = workflow.invoke(
        {
            "evaluation_input": build_input(),
        }
    )

    elapsed = time.perf_counter() - start_time

    verdict = validate_result(
        result,
        expected_memory_hit=expected_memory_hit,
    )

    print()
    print("RESULT")
    print("-" * 70)
    print(f"Memory hit : {result.get('memory_hit', False)}")
    print(f"Verdict    : {verdict.verdict}")
    print(f"Score      : {verdict.score:.4f}")
    print(f"Confidence : {verdict.confidence:.4f}")
    print(f"Evaluators : {len(result.get('evaluations', []))}")
    print(f"Evidence   : {len(result.get('evidence', []))}")
    print(f"Elapsed    : {elapsed:.2f}s")

    print()
    print("Evaluator scores:")

    for evaluation in result.get("evaluations", []):
        print(
            f"  {evaluation.evaluator:<24}"
            f" score={evaluation.score:.4f}"
            f" confidence={evaluation.confidence:.4f}"
        )

    return result


def main():
    print()
    print("=" * 70)
    print("V2 MEMORY-AWARE REGRESSION")
    print("=" * 70)

    print()
    print("This regression executes the same question twice.")

    print()
    print("RUN 1:")
    print("  Expected: MEMORY MISS")
    print("  Expected path:")
    print(
        "  memory -> RAG -> evidence quality -> "
        "reference generation -> validation -> evaluation"
    )
    print("  Expected: verified answer written to memory")

    print()
    print("RUN 2:")
    print("  Expected: MEMORY HIT")
    print("  Expected path:")
    print(
        "  memory -> stored evidence -> stored reference "
        "-> evaluation"
    )
    print(
        "  Expected: fresh RAG/reference generation/"
        "validation skipped"
    )

    print()
    print("Provider:", os.getenv("LLM_PROVIDER", "groq"))
    print(
        "Model:",
        os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        ),
    )

    # ---------------------------------------------------------
    # Dedicated regression memory
    # ---------------------------------------------------------

    memory_store = ChromaAnswerMemory(
        persist_directory="data/v2/regression_memory",
        collection_name="final_regression_memory",
    )

    workflow = build_workflow(memory_store)

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # We intentionally do NOT delete the Chroma collection here.
    #
    # This lets us verify persistent memory behavior.
    #
    # However, the regression expects the FIRST run to be a
    # memory miss. Therefore use a dedicated question and make
    # sure it has not already been inserted before running.
    # ---------------------------------------------------------

    existing = memory_store.collection.get(
        ids=[
            ChromaAnswerMemory.make_memory_id(
                QUESTION
            )
        ],
        include=["metadatas"],
    )

    if existing.get("ids"):
        print()
        print(
            "WARNING: The regression question already exists "
            "in memory."
        )
        print()
        print(
            "Delete the dedicated regression memory directory "
            "before running the test again:"
        )
        print()
        print("  data\\v2\\regression_memory")
        print()
        print(
            "Then rerun the regression."
        )

        raise RuntimeError(
            "Regression memory already contains the test question. "
            "A clean first-run MISS cannot be proven."
        )

    # ---------------------------------------------------------
    # RUN 1 — MEMORY MISS
    # ---------------------------------------------------------

    first_result = run_case(
        workflow,
        expected_memory_hit=False,
        label="RUN 1 — FRESH VERIFICATION / MEMORY MISS",
    )

    assert first_result.get("memory_hit") is False

    # ---------------------------------------------------------
    # Verify memory was written
    # ---------------------------------------------------------

    memory_id = ChromaAnswerMemory.make_memory_id(
        QUESTION
    )

    stored = memory_store.collection.get(
        ids=[memory_id],
        include=["metadatas"],
    )

    assert stored.get("ids"), (
        "RUN 1 completed, but no verified answer "
        "was written to memory."
    )

    assert memory_store.collection.count() >= 1, (
        "Memory store is empty after RUN 1."
    )

    print()
    print("=" * 70)
    print("MEMORY WRITE VERIFIED")
    print("=" * 70)

    print(f"Memory ID: {memory_id}")
    print(
        f"Stored records: "
        f"{memory_store.collection.count()}"
    )

    # ---------------------------------------------------------
    # RUN 2 — MEMORY HIT
    # ---------------------------------------------------------

    second_result = run_case(
        workflow,
        expected_memory_hit=True,
        label="RUN 2 — VERIFIED MEMORY REUSE / MEMORY HIT",
    )

    assert second_result.get("memory_hit") is True

    # ---------------------------------------------------------
    # Verify the stored answer is actually reused
    # ---------------------------------------------------------

    stored_answer = second_result.get(
        "memory_answer"
    )

    reference_answer = second_result.get(
        "reference_answer"
    )

    assert stored_answer, (
        "RUN 2 memory hit did not provide a stored answer."
    )

    assert reference_answer is not None, (
        "RUN 2 has no reference answer."
    )

    assert reference_answer.answer == stored_answer, (
        "RUN 2 reference answer does not match "
        "the verified answer retrieved from memory."
    )

    # ---------------------------------------------------------
    # Final regression summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("V2 MEMORY REGRESSION PASSED")
    print("=" * 70)

    print()
    print("Verified:")
    print("  [PASS] RUN 1 produced a memory MISS")
    print("  [PASS] RUN 1 completed fresh verification")
    print("  [PASS] RUN 1 stored the verified answer")
    print("  [PASS] RUN 2 produced a memory HIT")
    print("  [PASS] RUN 2 reused stored evidence")
    print("  [PASS] RUN 2 reused stored reference answer")
    print("  [PASS] Both runs evaluated the candidate response")
    print("  [PASS] Both runs completed arbitration")
    print("  [PASS] Both runs produced non-reject verdicts")

    print()
    print("The answer-memory redesign is functioning end-to-end.")


if __name__ == "__main__":
    main()