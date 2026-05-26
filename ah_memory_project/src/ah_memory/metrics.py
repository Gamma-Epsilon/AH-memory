"""Функциональные и структурные метрики памяти."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from ah_memory.config import RetrievalConfig
from ah_memory.knowledge import Hyperedge, KnowledgeHypergraph
from ah_memory.memory import AHMemory, RetrievalResult
from ah_memory.patterns import find_close_pairs
from ah_memory.symbols import SymbolSet


def average_recovery_accuracy(results: list[RetrievalResult], tests: list[Any]) -> float:
    """Считает среднюю точность восстановления по известным позициям входа."""

    if not results:
        return 0.0

    accuracies = [_known_position_accuracy(result, test) for result, test in zip(results, tests)]
    return float(np.mean(accuracies)) if accuracies else 0.0


def exact_match_accuracy(results: list[RetrievalResult]) -> float:
    """Считает долю полных совпадений восстановленного и истинного паттерна."""

    if not results:
        return 0.0

    matches = [bool(result.exact_match) for result in results if result.exact_match is not None]
    if not matches:
        return 0.0
    return float(np.mean(matches))


def noise_robustness(results: list[RetrievalResult], tests: list[Any]) -> dict[str, Any]:
    """Возвращает среднюю точность и зависимость качества от уровней искажений."""

    grouped_by_noise: dict[float, list[float]] = defaultdict(list)
    grouped_by_pair: dict[str, list[float]] = defaultdict(list)

    for result, test in zip(results, tests):
        accuracy = _known_position_accuracy(result, test)
        noise_level = float(getattr(test, "noise_level", 0.0))
        missing_level = float(getattr(test, "missing_level", 0.0))
        grouped_by_noise[noise_level].append(accuracy)
        grouped_by_pair[f"noise={noise_level};missing={missing_level}"].append(accuracy)

    return {
        "overall_accuracy": average_recovery_accuracy(results, tests),
        "accuracy_by_noise": {
            level: float(np.mean(values)) for level, values in sorted(grouped_by_noise.items())
        },
        "accuracy_by_noise_and_missing": {
            level_pair: float(np.mean(values)) for level_pair, values in sorted(grouped_by_pair.items())
        },
    }


def close_pattern_discrimination(
    results: list[RetrievalResult],
    close_pairs: list[tuple[int, int]],
) -> float:
    """Считает долю правильных восстановлений для паттернов с близкими соседями."""

    neighbor_map = _neighbor_map(close_pairs)
    relevant = [
        result
        for result in results
        if result.original_index is not None and result.original_index in neighbor_map
    ]
    if not relevant:
        return 0.0

    correct = [result.restored_index == result.original_index for result in relevant]
    return float(np.mean(correct))


def close_pattern_confusion_rate(
    results: list[RetrievalResult],
    close_pairs: list[tuple[int, int]],
) -> float:
    """Считает долю путаницы исходного паттерна с его близкими соседями."""

    neighbor_map = _neighbor_map(close_pairs)
    relevant = [
        result
        for result in results
        if result.original_index is not None and result.original_index in neighbor_map
    ]
    if not relevant:
        return 0.0

    confused = [
        result.restored_index in neighbor_map[int(result.original_index)]
        for result in relevant
    ]
    return float(np.mean(confused))


def functional_metrics(
    results: list[RetrievalResult],
    tests: list[Any],
    close_pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    """Объединяет функциональные показатели качества восстановления."""

    robustness = noise_robustness(results, tests)
    return {
        "average_recovery_accuracy": average_recovery_accuracy(results, tests),
        "exact_match_accuracy": exact_match_accuracy(results),
        "noise_robustness": robustness["overall_accuracy"],
        "accuracy_by_noise": robustness["accuracy_by_noise"],
        "accuracy_by_noise_and_missing": robustness["accuracy_by_noise_and_missing"],
        "close_pattern_discrimination": close_pattern_discrimination(results, close_pairs),
        "close_pattern_confusion_rate": close_pattern_confusion_rate(results, close_pairs),
    }


def structural_metrics(hypergraph: KnowledgeHypergraph, candidate_pool_size: int) -> dict[str, Any]:
    """Считает структурные характеристики гиперграфа памяти."""

    edge_count = len(hypergraph.hyperedges)
    sizes = hypergraph.edge_sizes()
    degrees = vertex_degrees(hypergraph)

    return {
        "hyperedge_count": edge_count,
        "average_hyperedge_size": float(np.mean(sizes)) if sizes else 0.0,
        "density": edge_count / candidate_pool_size if candidate_pool_size > 0 else 0.0,
        "max_vertex_degree": int(np.max(degrees)) if len(degrees) else 0,
        "mean_vertex_degree": float(np.mean(degrees)) if len(degrees) else 0.0,
        "vertex_degree_variance": float(np.var(degrees)) if len(degrees) else 0.0,
        "degree_gini": gini_index(degrees),
        "degree_entropy": degree_entropy(degrees),
        "hyperedge_size_histogram": hyperedge_size_histogram(hypergraph),
    }


def vertex_degrees(hypergraph: KnowledgeHypergraph) -> np.ndarray:
    """Возвращает степени вершин по числу входящих гиперребер."""

    degrees = np.zeros(hypergraph.dimension, dtype=int)
    for edge in hypergraph.hyperedges:
        for symbol in edge.symbols:
            degrees[symbol] += 1
    return degrees


def gini_index(values: np.ndarray) -> float:
    """Считает индекс Джини для распределения степеней."""

    prepared = np.asarray(values, dtype=float)
    if prepared.size == 0 or np.all(prepared == 0):
        return 0.0

    sorted_values = np.sort(prepared)
    n = sorted_values.size
    weighted_sum = np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_values)
    return float(weighted_sum / (n * np.sum(sorted_values)))


def degree_entropy(values: np.ndarray) -> float:
    """Считает энтропию нормированного распределения степеней."""

    prepared = np.asarray(values, dtype=float)
    total = float(np.sum(prepared))
    if prepared.size == 0 or total == 0.0:
        return 0.0

    probabilities = prepared[prepared > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def hyperedge_size_histogram(hypergraph: KnowledgeHypergraph) -> dict[int, int]:
    """Возвращает распределение гиперребер по кратности."""

    return dict(sorted(Counter(hypergraph.edge_sizes()).items()))


def evaluate_memory(
    patterns: np.ndarray,
    topology: list[Hyperedge | tuple[int, ...] | list[int]],
    tests: list[Any],
    retrieval_config: RetrievalConfig | None,
    candidate_pool_size: int,
) -> dict[str, Any]:
    """Создает память с заданной топологией и возвращает все метрики."""

    prepared_patterns = np.asarray(patterns, dtype=int)
    if prepared_patterns.ndim != 2:
        raise ValueError("Паттерны должны быть двумерной матрицей.")

    symbol_set = SymbolSet(prepared_patterns.shape[1])
    knowledge = KnowledgeHypergraph(symbol_set.dimension)
    for edge in topology:
        knowledge.add_hyperedge(_edge_symbols(edge))

    memory = AHMemory(symbol_set, knowledge=knowledge, retrieval_config=retrieval_config)
    memory.fit(prepared_patterns)
    results = memory.retrieve_batch(tests)
    close_pairs = find_close_pairs(prepared_patterns, max_distance=_close_pair_threshold(tests))

    metrics = functional_metrics(results, tests, close_pairs)
    metrics.update(structural_metrics(memory.knowledge, candidate_pool_size))
    return metrics


def _known_position_accuracy(result: RetrievalResult, test: Any) -> float:
    original = getattr(test, "original", None)
    distorted = getattr(test, "distorted", None)

    if original is None:
        return float(result.accuracy) if result.accuracy is not None else 0.0

    prepared_original = np.asarray(original)
    restored = np.asarray(result.restored_pattern)
    if distorted is None:
        mask = np.ones(prepared_original.shape, dtype=bool)
    else:
        mask = np.asarray(distorted) != -1

    if not np.any(mask):
        return 0.0
    return float(np.mean(restored[mask] == prepared_original[mask]))


def _neighbor_map(close_pairs: list[tuple[int, int]]) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    for left, right in close_pairs:
        neighbors[int(left)].add(int(right))
        neighbors[int(right)].add(int(left))
    return neighbors


def _edge_symbols(edge: Hyperedge | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(edge, Hyperedge):
        return edge.symbols
    return tuple(int(symbol) for symbol in edge)


def _close_pair_threshold(tests: list[Any]) -> int:
    distances = []
    for test in tests:
        original = getattr(test, "original", None)
        distorted = getattr(test, "distorted", None)
        if original is None or distorted is None:
            continue
        mask = np.asarray(distorted) != -1
        distances.append(int(np.sum(np.asarray(original)[mask] != np.asarray(distorted)[mask])))
    return max(distances) if distances else 0
