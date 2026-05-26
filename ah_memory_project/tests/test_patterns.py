import numpy as np

from ah_memory.config import PatternGenerationConfig
from ah_memory.patterns import (
    corrupt_pattern,
    find_close_pairs,
    generate_clustered_patterns,
    prepare_pattern_tests,
)


def test_clustered_patterns_have_expected_shape_and_activity():
    config = PatternGenerationConfig(
        pattern_count=12,
        dimension=20,
        active_fraction=0.25,
        cluster_count=3,
        intra_cluster_flips=4,
        seed=7,
    )

    patterns = generate_clustered_patterns(config)

    assert patterns.shape == (12, 20)
    assert set(np.unique(patterns)).issubset({0, 1})
    assert np.all(patterns.sum(axis=1) == 5)


def test_corruption_flips_bits_and_adds_missing_values():
    pattern = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    rng = np.random.default_rng(3)

    distorted = corrupt_pattern(pattern, noise_level=0.25, missing_level=0.25, rng=rng)

    assert np.sum(distorted == -1) == 2
    known_mask = distorted != -1
    assert np.any(distorted[known_mask] != pattern[known_mask])


def test_prepare_pattern_tests_uses_all_levels_and_trials():
    config = PatternGenerationConfig(
        pattern_count=2,
        dimension=8,
        active_fraction=0.5,
        cluster_count=1,
        noise_levels=(0.0, 0.25),
        missing_levels=(0.0, 0.25),
        trials_per_pattern=3,
        seed=11,
    )
    patterns = generate_clustered_patterns(config)

    tests = prepare_pattern_tests(patterns, config)

    assert len(tests) == 2 * 2 * 2 * 3
    assert {test.pattern_index for test in tests} == {0, 1}


def test_find_close_pairs_detects_similar_patterns():
    patterns = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 1],
        ]
    )

    pairs = find_close_pairs(patterns, max_distance=1)

    assert pairs == [(0, 1)]
