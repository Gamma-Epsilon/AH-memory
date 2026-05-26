"""Генерация паттернов и искаженных входов."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ah_memory.config import PatternGenerationConfig


@dataclass(frozen=True)
class PatternTestCase:
    """Описывает один тестовый вход для проверки восстановления."""

    pattern_index: int
    original: np.ndarray
    distorted: np.ndarray
    noise_level: float
    missing_level: float


def generate_clustered_patterns(config: PatternGenerationConfig) -> np.ndarray:
    """Создает бинарные паттерны с близостью внутри кластеров."""

    rng = np.random.default_rng(config.seed)
    active_count = _active_count(config.dimension, config.active_fraction)
    centers = np.vstack(
        [_make_binary_pattern(config.dimension, active_count, rng) for _ in range(config.cluster_count)]
    )
    patterns = []

    for index in range(config.pattern_count):
        center = centers[index % config.cluster_count]
        patterns.append(_mutate_near_center(center, config.intra_cluster_flips, rng))

    return np.asarray(patterns, dtype=int)


def corrupt_pattern(
    pattern: np.ndarray,
    noise_level: float,
    missing_level: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Вносит инверсии и пропуски в один бинарный паттерн."""

    _validate_level(noise_level, "Уровень шума")
    _validate_level(missing_level, "Уровень пропусков")
    generator = rng or np.random.default_rng()
    distorted = np.asarray(pattern, dtype=int).copy()
    dimension = distorted.size

    flip_count = _level_to_count(dimension, noise_level)
    missing_count = _level_to_count(dimension, missing_level)

    if flip_count > 0:
        flip_indices = generator.choice(dimension, size=flip_count, replace=False)
        distorted[flip_indices] = 1 - distorted[flip_indices]

    if missing_count > 0:
        missing_indices = generator.choice(dimension, size=missing_count, replace=False)
        distorted[missing_indices] = -1

    return distorted


def prepare_pattern_tests(
    patterns: np.ndarray,
    config: PatternGenerationConfig,
) -> list[PatternTestCase]:
    """Формирует набор тестов для разных уровней искажений."""

    generator = np.random.default_rng(config.seed + 1)
    prepared_patterns = np.asarray(patterns, dtype=int)
    tests: list[PatternTestCase] = []

    for pattern_index, original in enumerate(prepared_patterns):
        for noise_level in config.noise_levels:
            for missing_level in config.missing_levels:
                for _ in range(config.trials_per_pattern):
                    distorted = corrupt_pattern(original, noise_level, missing_level, generator)
                    tests.append(
                        PatternTestCase(
                            pattern_index=pattern_index,
                            original=original.copy(),
                            distorted=distorted,
                            noise_level=float(noise_level),
                            missing_level=float(missing_level),
                        )
                    )

    return tests


def find_close_pairs(patterns: np.ndarray, max_distance: int) -> list[tuple[int, int]]:
    """Находит пары паттернов с малым расстоянием Хэмминга."""

    if max_distance < 0:
        raise ValueError("Порог расстояния не может быть отрицательным.")

    prepared_patterns = np.asarray(patterns, dtype=int)
    pairs: list[tuple[int, int]] = []

    for left_index in range(len(prepared_patterns)):
        for right_index in range(left_index + 1, len(prepared_patterns)):
            distance = int(np.sum(prepared_patterns[left_index] != prepared_patterns[right_index]))
            if distance <= max_distance:
                pairs.append((left_index, right_index))

    return pairs


def _make_binary_pattern(
    dimension: int,
    active_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pattern = np.zeros(dimension, dtype=int)
    active_indices = rng.choice(dimension, size=active_count, replace=False)
    pattern[active_indices] = 1
    return pattern


def _mutate_near_center(
    center: np.ndarray,
    flips: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pattern = center.copy()
    active_indices = np.flatnonzero(pattern == 1)
    inactive_indices = np.flatnonzero(pattern == 0)
    replacement_count = min(flips // 2, len(active_indices), len(inactive_indices))

    if replacement_count == 0:
        return pattern

    turn_off = rng.choice(active_indices, size=replacement_count, replace=False)
    turn_on = rng.choice(inactive_indices, size=replacement_count, replace=False)
    pattern[turn_off] = 0
    pattern[turn_on] = 1
    return pattern


def _active_count(dimension: int, active_fraction: float) -> int:
    return max(1, min(dimension - 1, int(round(dimension * active_fraction))))


def _level_to_count(dimension: int, level: float) -> int:
    return min(dimension, int(round(dimension * level)))


def _validate_level(level: float, label: str) -> None:
    if level < 0 or level > 1:
        raise ValueError(f"{label} должен быть в диапазоне от нуля до единицы.")
