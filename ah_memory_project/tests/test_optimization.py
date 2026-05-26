import numpy as np

from ah_memory.config import OptimizationConfig, PatternGenerationConfig, RetrievalConfig
from ah_memory.optimization import AHMemoryTopologyProblem, run_nsga2
from ah_memory.patterns import PatternTestCase


def test_topology_problem_uses_generation_config_for_close_threshold():
    patterns = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 1],
        ]
    )
    tests = [
        PatternTestCase(0, patterns[0], np.array([1, 0, 0, 0]), 0.0, 0.0),
        PatternTestCase(1, patterns[1], np.array([1, 0, 0, 0]), 0.0, 0.0),
        PatternTestCase(2, patterns[2], np.array([0, 0, 1, 1]), 0.0, 0.0),
    ]
    generation_config = PatternGenerationConfig(
        pattern_count=3,
        dimension=4,
        active_fraction=0.5,
        cluster_count=1,
        close_pair_distance=1,
    )
    problem = AHMemoryTopologyProblem(
        patterns=patterns,
        tests=tests,
        candidate_pool=[(0, 1), (2, 3)],
        retrieval_config=RetrievalConfig(),
        generation_config=generation_config,
        optimization_config=OptimizationConfig(population_size=4, n_generations=2),
    )
    out = {}

    problem._evaluate(np.array([True, True]), out)

    assert out["metrics"]["close_pattern_distance"] == 1
    assert out["metrics"]["close_pair_count"] == 1
    assert out["F"].shape == (4,)


def test_run_nsga2_short_check_returns_table_with_close_threshold():
    patterns = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]])
    tests = [
        PatternTestCase(0, patterns[0], np.array([1, 1, -1, -1]), 0.0, 0.5),
        PatternTestCase(1, patterns[1], np.array([1, 0, -1, -1]), 0.0, 0.5),
        PatternTestCase(2, patterns[2], np.array([-1, -1, 1, 1]), 0.0, 0.5),
    ]
    configs = {
        "optimization_config": OptimizationConfig(
            population_size=6,
            n_generations=2,
            min_selected_hyperedges=1,
            max_selected_hyperedges=2,
            seed=5,
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

    result, table = run_nsga2(
        patterns=patterns,
        tests=tests,
        candidate_pool=[(0, 1), (0, 2), (2, 3)],
        configs=configs,
    )

    assert result.F is not None
    assert not table.empty
    assert "close_pattern_distance" in table.columns
    assert set(table["close_pattern_distance"]) == {1}
