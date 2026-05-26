"""Эволюционная оптимизация топологии гиперграфа."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize

from ah_memory.config import OptimizationConfig, PatternGenerationConfig, RetrievalConfig
from ah_memory.knowledge import Hyperedge
from ah_memory.metrics import evaluate_memory


class AHMemoryTopologyProblem(ElementwiseProblem):
    """Описывает многокритериальный поиск топологии АГ-памяти."""

    def __init__(
        self,
        patterns: np.ndarray,
        tests: list[Any],
        candidate_pool: list[Hyperedge | tuple[int, ...] | list[int]],
        retrieval_config: RetrievalConfig,
        generation_config: PatternGenerationConfig,
        optimization_config: OptimizationConfig,
    ) -> None:
        self.patterns = np.asarray(patterns, dtype=int)
        self.tests = tests
        self.candidate_pool = candidate_pool
        self.retrieval_config = retrieval_config
        self.generation_config = generation_config
        self.optimization_config = optimization_config
        self.max_selected = optimization_config.max_selected_hyperedges or len(candidate_pool)
        self.evaluated_metrics: list[dict[str, Any]] = []

        super().__init__(
            n_var=len(candidate_pool),
            n_obj=4,
            xl=0,
            xu=1,
            vtype=bool,
        )

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Оценивает одну бинарную топологию по четырем целям."""

        genome = np.asarray(x, dtype=bool)
        selected_count = int(np.sum(genome))
        topology = [edge for edge, selected in zip(self.candidate_pool, genome) if selected]

        metrics = evaluate_memory(
            patterns=self.patterns,
            topology=topology,
            tests=self.tests,
            retrieval_config=self.retrieval_config,
            candidate_pool_size=len(self.candidate_pool),
            generation_config=self.generation_config,
        )
        penalty = self._constraint_penalty(selected_count)
        complexity = self._complexity_objective(metrics)

        out["F"] = np.array(
            [
                1.0 - metrics["average_recovery_accuracy"] + penalty,
                1.0 - metrics["noise_robustness"] + penalty,
                1.0 - metrics["close_pattern_discrimination"] + penalty,
                complexity + penalty,
            ],
            dtype=float,
        )

        stored_metrics = dict(metrics)
        stored_metrics["selected_hyperedges"] = selected_count
        stored_metrics["constraint_penalty"] = penalty
        stored_metrics["complexity_objective"] = complexity
        self.evaluated_metrics.append(stored_metrics)
        out["metrics"] = stored_metrics

    def metrics_for_genome(self, genome: np.ndarray) -> dict[str, Any]:
        """Повторно считает метрики для решения на Парето-фронте."""

        selected = np.asarray(genome, dtype=bool)
        topology = [edge for edge, is_selected in zip(self.candidate_pool, selected) if is_selected]
        metrics = evaluate_memory(
            patterns=self.patterns,
            topology=topology,
            tests=self.tests,
            retrieval_config=self.retrieval_config,
            candidate_pool_size=len(self.candidate_pool),
            generation_config=self.generation_config,
        )
        selected_count = int(np.sum(selected))
        metrics["selected_hyperedges"] = selected_count
        metrics["constraint_penalty"] = self._constraint_penalty(selected_count)
        metrics["complexity_objective"] = self._complexity_objective(metrics)
        return metrics

    def _constraint_penalty(self, selected_count: int) -> float:
        if selected_count < self.optimization_config.min_selected_hyperedges:
            return float(self.optimization_config.min_selected_hyperedges - selected_count)
        if selected_count > self.max_selected:
            return float(selected_count - self.max_selected)
        return 0.0

    def _complexity_objective(self, metrics: dict[str, Any]) -> float:
        edge_norm = metrics["hyperedge_count"] / max(1, len(self.candidate_pool))
        mean_size_norm = metrics["average_hyperedge_size"] / max(1, self.patterns.shape[1])
        degree_variance_norm = metrics["vertex_degree_variance"] / max(1, len(self.candidate_pool))
        return float(
            0.35 * edge_norm
            + 0.35 * metrics["density"]
            + 0.15 * mean_size_norm
            + 0.15 * degree_variance_norm
        )


def run_nsga2(
    patterns: np.ndarray,
    tests: list[Any],
    candidate_pool: list[Hyperedge | tuple[int, ...] | list[int]],
    configs: dict[str, Any] | OptimizationConfig | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Запускает NSGA-II и возвращает результат с таблицей решений."""

    optimization_config, retrieval_config, generation_config = _resolve_configs(configs)
    problem = AHMemoryTopologyProblem(
        patterns=patterns,
        tests=tests,
        candidate_pool=candidate_pool,
        retrieval_config=retrieval_config,
        generation_config=generation_config,
        optimization_config=optimization_config,
    )
    mutation_probability = optimization_config.mutation_probability or 1.0 / max(1, len(candidate_pool))
    algorithm = NSGA2(
        pop_size=optimization_config.population_size,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=optimization_config.crossover_probability),
        mutation=BitflipMutation(prob=mutation_probability),
        eliminate_duplicates=optimization_config.eliminate_duplicates,
    )

    result = minimize(
        problem,
        algorithm,
        ("n_gen", optimization_config.n_generations),
        seed=optimization_config.seed,
        verbose=False,
    )
    return result, _pareto_metrics_table(problem, result)


def _resolve_configs(
    configs: dict[str, Any] | OptimizationConfig | None,
) -> tuple[OptimizationConfig, RetrievalConfig, PatternGenerationConfig]:
    if configs is None:
        return OptimizationConfig(), RetrievalConfig(), PatternGenerationConfig()
    if isinstance(configs, OptimizationConfig):
        return configs, RetrievalConfig(), PatternGenerationConfig()

    optimization_config = configs.get("optimization_config", OptimizationConfig())
    retrieval_config = configs.get("retrieval_config", RetrievalConfig())
    generation_config = configs.get("generation_config", PatternGenerationConfig())
    return optimization_config, retrieval_config, generation_config


def _pareto_metrics_table(problem: AHMemoryTopologyProblem, result: Any) -> pd.DataFrame:
    if result.X is None:
        return pd.DataFrame()

    genomes = np.atleast_2d(result.X)
    objectives = np.atleast_2d(result.F)
    rows: list[dict[str, Any]] = []

    for solution_index, genome in enumerate(genomes):
        metrics = problem.metrics_for_genome(genome)
        row = {
            "solution_index": solution_index,
            "objective_recovery_error": float(objectives[solution_index, 0]),
            "objective_noise_error": float(objectives[solution_index, 1]),
            "objective_close_pattern_error": float(objectives[solution_index, 2]),
            "objective_complexity": float(objectives[solution_index, 3]),
            "genome": "".join("1" if value else "0" for value in np.asarray(genome, dtype=bool)),
        }
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)
