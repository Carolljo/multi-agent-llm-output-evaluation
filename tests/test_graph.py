from src.graph.graph import create_evaluation_graph


def test_graph_compiles():
    graph = create_evaluation_graph()

    assert graph is not None