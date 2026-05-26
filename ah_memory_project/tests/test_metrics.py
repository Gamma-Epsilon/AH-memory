import numpy as np

from ah_memory.config import RetrievalConfig
from ah_memory.knowledge import KnowledgeHypergraph
from ah_memory.memory import RetrievalResult
from ah_memory.metrics import (
    average_recovery_accuracy,
    close_pattern_confusion_rate,
    close_pattern_discrimination,
    evaluate_memory,
    exact_match_accuracy,
    structural_metrics,
)
from ah_memory.patterns import PatternTestCase


def test_functional_metrics_are_perfect_for_exact_restoration():
    tests = [
        PatternTestCase(
            pattern_index=0,
            original=np.array([1, 0, 1, 0]),
            distorted=np.array([1, -1, 1, 0]),
            noise_level=0.0,
            missing_level=0.25,
        ),
        PatternTestCase(
            pattern_index=1,
            original=np.array([0, 1, 0, 1]),
            distorted=np.array([0, 1, -1, 1]),
            noise_level=0.0,
            missing_level=0.25,
        ),
    ]
    results = [
        RetrievalResult(0, 0, np.array([1, 0, 1, 0]), 1.0, True, 1.0),
        RetrievalResult(1, 1, np.array([0, 1, 0, 1]), 1.0, True, 1.0),
    ]

    assert average_recovery_accuracy(results, tests) == 1.0
    assert exact_match_accuracy(results) == 1.0


def test_close_pattern_metrics_count_discrimination_and_confusion():
    results = [
        RetrievalResult(0, 0, np.array([1, 1, 0]), 1.0, True, 1.0),
        RetrievalResult(1, 0, np.array([1, 1, 0]), 0.67, False, 0.8),
        RetrievalResult(2, 2, np.array([0, 0, 1]), 1.0, True, 1.0),
    ]
    close_pairs = [(0, 1)]

    assert close_pattern_discrimination(results, close_pairs) == 0.5
    assert close_pattern_confusion_rate(results, close_pairs) == 0.5


def test_structural_metrics_describe_hypergraph():
    hypergraph = KnowledgeHypergraph(4)
    hypergraph.add_hyperedge((0, 1))
    hypergraph.add_hyperedge((1, 2, 3))

    metrics = structural_metrics(hypergraph, candidate_pool_size=4)

    assert metrics["hyperedge_count"] == 2
    assert metrics["average_hyperedge_size"] == 2.5
    assert metrics["density"] == 0.5
    assert metrics["max_vertex_degree"] == 2
    assert metrics["hyperedge_size_histogram"] == {2: 1, 3: 1}


def test_evaluate_memory_returns_functional_and_structural_metrics():
    patterns = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    tests = [
        PatternTestCase(
            pattern_index=0,
            original=patterns[0],
            distorted=np.array([1, 1, -1, -1]),
            noise_level=0.0,
            missing_level=0.5,
        ),
        PatternTestCase(
            pattern_index=1,
            original=patterns[1],
            distorted=np.array([-1, -1, 1, 1]),
            noise_level=0.0,
            missing_level=0.5,
        ),
    ]

    metrics = evaluate_memory(
        patterns=patterns,
        topology=[(0, 1), (2, 3)],
        tests=tests,
        retrieval_config=RetrievalConfig(),
        candidate_pool_size=4,
    )

    assert metrics["average_recovery_accuracy"] == 1.0
    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["hyperedge_count"] == 2
    assert metrics["density"] == 0.5
