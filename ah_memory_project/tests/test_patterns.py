import numpy as np

from ah_memory.config import PatternGenerationConfig
from ah_memory.patterns import (
    corrupt_pattern,
    find_close_pairs,
    generate_clustered_pattern_data,
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
        close_pair_distance=4,
        seed=7,
    )

    patterns = generate_clustered_patterns(config)

    assert patterns.shape == (12, 20)
    assert set(np.unique(patterns)).issubset({0, 1})
    assert np.all(patterns.sum(axis=1) == 5)


def test_cluster_centers_respect_minimum_hamming_distance():
    config = PatternGenerationConfig(
        pattern_count=12,
        dimension=20,
        active_fraction=0.25,
        cluster_count=3,
        close_pair_distance=4,
        seed=7,
    )

    data = generate_clustered_pattern_data(config)
    distances = [
        _hamming_distance(left, right)
        for left_index, left in enumerate(data.centers)
        for right in data.centers[left_index + 1 :]
    ]

    assert min(distances) >= data.used_center_distance
    assert data.used_center_distance >= max(2 * config.close_pair_distance, int(0.25 * config.dimension))


def test_intercluster_distance_is_larger_than_intracluster_distance():
    config = PatternGenerationConfig(
        pattern_count=18,
        dimension=24,
        active_fraction=0.25,
        cluster_count=3,
        close_pair_distance=4,
        seed=13,
    )

    data = generate_clustered_pattern_data(config)
    inner_distances = []
    outer_distances = []

    for left_index, left in enumerate(data.patterns):
        for right_index in range(left_index + 1, len(data.patterns)):
            distance = _hamming_distance(left, data.patterns[right_index])
            if data.cluster_labels[left_index] == data.cluster_labels[right_index]:
                inner_distances.append(distance)
            else:
                outer_distances.append(distance)

    assert np.mean(outer_distances) > np.mean(inner_distances)


def test_clustered_pattern_generation_is_reproducible_with_same_seed():
    config = PatternGenerationConfig(
        pattern_count=9,
        dimension=18,
        active_fraction=0.33,
        cluster_count=3,
        close_pair_distance=3,
        seed=21,
    )

    first = generate_clustered_pattern_data(config)
    second = generate_clustered_pattern_data(config)

    assert np.array_equal(first.patterns, second.patterns)
    assert np.array_equal(first.cluster_labels, second.cluster_labels)
    assert np.array_equal(first.centers, second.centers)


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
        close_pair_distance=2,
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


def _hamming_distance(left, right):
    return int(np.sum(left != right))
