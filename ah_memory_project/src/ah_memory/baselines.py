"""Базовые топологии для сравнения с оптимизированными решениями."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ah_memory.config import BaselineConfig, PatternGenerationConfig, RetrievalConfig
from ah_memory.knowledge import Hyperedge
from ah_memory.metrics import evaluate_memory


def random_topology(
    candidate_pool: list[Hyperedge | tuple[int, ...] | list[int]],
    min_size: int,
    max_size: int,
    rng: np.random.Generator,
) -> list[Hyperedge | tuple[int, ...] | list[int]]:
    """Выбирает случайное подмножество гиперребер из пула."""

    pool_size = len(candidate_pool)
    if pool_size == 0:
        return []
    size = int(rng.integers(min_size, max_size + 1))
    size = max(0, min(size, pool_size))
    indices = rng.choice(pool_size, size=size, replace=False)
    return [candidate_pool[int(index)] for index in indices]


def frequency_topology(
    patterns: np.ndarray,
    candidate_pool: list[Hyperedge | tuple[int, ...] | list[int]],
    selected_count: int,
) -> list[Hyperedge | tuple[int, ...] | list[int]]:
    """Выбирает гиперребра с наибольшей частотой совместной активации."""

    ranked = sorted(
        candidate_pool,
        key=lambda edge: _joint_activation_frequency(patterns, edge),
        reverse=True,
    )
    return ranked[: max(0, min(selected_count, len(ranked)))]


def pairwise_topology(
    candidate_pool: list[Hyperedge | tuple[int, ...] | list[int]],
    max_count: int | None = None,
) -> list[Hyperedge | tuple[int, ...] | list[int]]:
    """Оставляет только парные связи, соответствующие обычному графу."""

    selected = [edge for edge in candidate_pool if len(_edge_symbols(edge)) == 2]
    if max_count is None:
        return selected
    return selected[: max(0, min(max_count, len(selected)))]


def dense_topology(
    candidate_pool: list[Hyperedge | tuple[int, ...] | list[int]],
    max_count: int,
) -> list[Hyperedge | tuple[int, ...] | list[int]]:
    """Берет плотную топологию с большим числом гиперребер."""

    return list(candidate_pool[: max(0, min(max_count, len(candidate_pool)))])


def evaluate_baselines(
    patterns: np.ndarray,
    tests: list[Any],
    candidate_pool: list[Hyperedge | tuple[int, ...] | list[int]],
    configs: dict[str, Any] | BaselineConfig | None = None,
) -> pd.DataFrame:
    """Строит базовые топологии и оценивает их теми же метриками, что и NSGA-II."""

    baseline_config, retrieval_config, generation_config = _resolve_configs(configs)
    max_selected = baseline_config.max_selected_hyperedges or len(candidate_pool)
    selected_count = baseline_config.selected_hyperedges or max_selected
    rng = np.random.default_rng(baseline_config.seed)
    rows: list[dict[str, Any]] = []

    for run_index in range(baseline_config.n_random_runs):
        topology = random_topology(
            candidate_pool,
            baseline_config.min_selected_hyperedges,
            max_selected,
            rng,
        )
        rows.append(
            _evaluate_baseline_row(
                "random",
                run_index,
                patterns,
                tests,
                topology,
                candidate_pool,
                retrieval_config,
                generation_config,
            )
        )

    fixed_topologies = [
        ("frequency", frequency_topology(patterns, candidate_pool, selected_count)),
        ("pairwise", pairwise_topology(candidate_pool, max_selected)),
        ("dense", dense_topology(candidate_pool, max_selected)),
    ]
    for name, topology in fixed_topologies:
        rows.append(
            _evaluate_baseline_row(
                name,
                0,
                patterns,
                tests,
                topology,
                candidate_pool,
                retrieval_config,
                generation_config,
            )
        )

    return pd.DataFrame(rows)


def _evaluate_baseline_row(
    baseline_name: str,
    run_index: int,
    patterns: np.ndarray,
    tests: list[Any],
    topology: list[Hyperedge | tuple[int, ...] | list[int]],
    candidate_pool: list[Hyperedge | tuple[int, ...] | list[int]],
    retrieval_config: RetrievalConfig,
    generation_config: PatternGenerationConfig,
) -> dict[str, Any]:
    metrics = evaluate_memory(
        patterns=patterns,
        topology=topology,
        tests=tests,
        retrieval_config=retrieval_config,
        candidate_pool_size=len(candidate_pool),
        generation_config=generation_config,
    )
    row = {
        "baseline": baseline_name,
        "run_index": run_index,
        "selected_hyperedges": len(topology),
    }
    row.update(metrics)
    return row


def _resolve_configs(
    configs: dict[str, Any] | BaselineConfig | None,
) -> tuple[BaselineConfig, RetrievalConfig, PatternGenerationConfig]:
    if configs is None:
        return BaselineConfig(), RetrievalConfig(), PatternGenerationConfig()
    if isinstance(configs, BaselineConfig):
        return configs, RetrievalConfig(), PatternGenerationConfig()

    baseline_config = configs.get("baseline_config", BaselineConfig())
    retrieval_config = configs.get("retrieval_config", RetrievalConfig())
    generation_config = configs.get("generation_config", PatternGenerationConfig())
    return baseline_config, retrieval_config, generation_config


def _joint_activation_frequency(
    patterns: np.ndarray,
    edge: Hyperedge | tuple[int, ...] | list[int],
) -> float:
    symbols = _edge_symbols(edge)
    prepared = np.asarray(patterns)
    if prepared.size == 0:
        return 0.0
    return float(np.mean(np.all(prepared[:, symbols] == 1, axis=1)))


def _edge_symbols(edge: Hyperedge | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(edge, Hyperedge):
        return edge.symbols
    return tuple(int(symbol) for symbol in edge)
