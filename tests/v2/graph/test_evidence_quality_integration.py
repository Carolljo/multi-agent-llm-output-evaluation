from unittest.mock import MagicMock

from src.v2.graph.workflow import build_autonomous_workflow


def build_dependencies():
    return {
        "retriever": MagicMock(),
        "reference_generator": MagicMock(),
        "reference_validator": MagicMock(),
        "claim_extractor": MagicMock(),
        "accuracy_evaluator": MagicMock(),
        "completeness_evaluator": MagicMock(),
        "logic_evaluator": MagicMock(),
        "grounding_evaluator": MagicMock(),
        "adjudicator": MagicMock(),
    }


def get_edges(workflow):
    return {
        (edge.source, edge.target)
        for edge in workflow.get_graph().edges
    }


def test_memory_evidence_flows_through_quality_before_reference():
    dependencies = build_dependencies()

    workflow = build_autonomous_workflow(
        **dependencies,
        memory_retriever=MagicMock(),
    )

    edges = get_edges(workflow)

    assert ("use_memory_evidence", "score_evidence_quality") in edges
    assert ("score_evidence_quality", "generate_reference") in edges


def test_fresh_retrieval_flows_through_quality_before_reference():
    dependencies = build_dependencies()

    workflow = build_autonomous_workflow(
        **dependencies,
    )

    edges = get_edges(workflow)

    assert ("retrieve_evidence", "score_evidence_quality") in edges
    assert ("score_evidence_quality", "generate_reference") in edges


def test_evidence_quality_node_exists_in_workflow():
    dependencies = build_dependencies()

    workflow = build_autonomous_workflow(
        **dependencies,
        memory_retriever=MagicMock(),
    )

    nodes = workflow.get_graph().nodes

    assert "score_evidence_quality" in nodes


def test_memory_and_fresh_paths_share_same_quality_node():
    dependencies = build_dependencies()

    workflow = build_autonomous_workflow(
        **dependencies,
        memory_retriever=MagicMock(),
    )

    edges = get_edges(workflow)

    assert ("use_memory_evidence", "score_evidence_quality") in edges
    assert ("retrieve_evidence", "score_evidence_quality") in edges
    assert ("score_evidence_quality", "generate_reference") in edges