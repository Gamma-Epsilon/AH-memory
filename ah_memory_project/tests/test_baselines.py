import numpy as np

from ah_memory.baselines import (
    evaluate_baselines,
    frequency_topology,
    pairwise_topology,
    random_topology,
)
from ah_memory.config import BaselineConfig, PatternGenerationConfig, RetrievalConfig
from ah_memory.patterns import PatternTestCase


def test_random_topology_uses_requested_size_and_seeded_randomness():
    candidate_pool = [(0, 1), (0, 2), (1, 2), (2, 3)]
    first_rng = np.random.default_rng(7)
    second_rng = np.random.default_rng(7)

    first = random_topology(candidate_pool, min_size=2, max_size=2, rng=first_rng)
    second = random_topology(candidate_pool, min_size=2, max_size=2, rng=second_rng)

    assert len(first) == 2
    assert len(second) == 2
    assert first == second


def test_frequency_topology_selects_most_active_edges():
    patterns = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
        ]
    )
    candidate_pool = [(0, 1), (0, 2), (2, 3)]

    selected = frequency_topology(patterns, candidate_pool, selected_count=2)

    assert selected == [(0, 1), (0, 2)]


def test_pairwise_topology_keeps_only_edges_of_size_two():
    candidate_pool = [(0, 1), (0, 1, 2), (2, 3)]

    selected = pairwise_topology(candidate_pool)

    assert selected == [(0, 1), (2, 3)]


def test_evaluate_baselines_uses_same_metrics_and_close_threshold():
    patterns = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]])
    tests = [
        PatternTestCase(0, patterns[0], np.array([1, 1, -1, -1]), 0.0, 0.5),
        PatternTestCase(1, patterns[1], np.array([1, 0, -1, -1]), 0.0, 0.5),
        PatternTestCase(2, patterns[2], np.array([-1, -1, 1, 1]), 0.0, 0.5),
    ]
    configs = {
        "baseline_config": BaselineConfig(
            n_random_runs=2,
            min_selected_hyperedges=1,
            max_selected_hyperedges=2,
            selected_hyperedges=2,
            seed=4,
        ),
        "retrieval_config": RetrievalConfig(),
        "generation_config": PatternGenerationConfig(
            pattern_count=3,
            dimension=4,
            active_fraction=0.5,
            cluster_count=1,
            close_pair_distance=1,
        ),
    }

    table = evaluate_baselines(
        patterns=patterns,
        tests=tests,
        candidate_pool=[(0, 1), (0, 2), (2, 3)],
        configs=configs,
    )

    assert set(table["baseline"]) == {"random", "frequency", "pairwise", "dense"}
    assert set(table["close_pattern_distance"]) == {1}
    assert "average_recovery_accuracy" in table.columns
    assert "close_pattern_discrimination" in table.columns
    assert "close_pattern_confusion_rate" in table.columns
    assert "hyperedge_count" in table.columns
    assert "mean_vertex_degree" in table.columns
