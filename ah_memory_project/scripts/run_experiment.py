"""Запуск полного эксперимента по оптимизации топологии АГ-памяти."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ah_memory.baselines import evaluate_baselines
from ah_memory.config import (
    BaselineConfig,
    ExperimentConfig,
    HypergraphConfig,
    OptimizationConfig,
    PatternGenerationConfig,
    RetrievalConfig,
)
from ah_memory.optimization import run_nsga2
from ah_memory.patterns import generate_clustered_pattern_data, prepare_pattern_tests


def main() -> None:
    args = _parse_args()
    experiment_config, generation_config, hypergraph_config, optimization_config = _make_configs(args)
    retrieval_config = RetrievalConfig()
    baseline_config = BaselineConfig(
        n_random_runs=args.baseline_runs,
        min_selected_hyperedges=optimization_config.min_selected_hyperedges,
        max_selected_hyperedges=optimization_config.max_selected_hyperedges,
        selected_hyperedges=optimization_config.max_selected_hyperedges,
        seed=experiment_config.seed,
    )

    np.random.seed(experiment_config.seed)
    output_dir = _resolve_output_dir(experiment_config.output_dir)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pattern_data = generate_clustered_pattern_data(generation_config)
    tests = prepare_pattern_tests(pattern_data.patterns, generation_config)
    candidate_pool, candidate_table = build_candidate_pool(
        pattern_data.patterns,
        hypergraph_config,
        experiment_config.seed,
    )

    nsga_result, pareto_table = run_nsga2(
        pattern_data.patterns,
        tests,
        candidate_pool,
        {
            "optimization_config": optimization_config,
            "retrieval_config": retrieval_config,
            "generation_config": generation_config,
        },
    )
    nsga_objectives = getattr(nsga_result, "evaluated_metrics_table", pd.DataFrame())
    baseline_table = evaluate_baselines(
        pattern_data.patterns,
        tests,
        candidate_pool,
        {
            "baseline_config": baseline_config,
            "retrieval_config": retrieval_config,
            "generation_config": generation_config,
        },
    )

    _save_tables(tables_dir, pattern_data.patterns, candidate_table, pareto_table, nsga_objectives, baseline_table)
    summary = _build_summary(
        pattern_data.metadata,
        generation_config,
        hypergraph_config,
        optimization_config,
        baseline_config,
        candidate_pool,
        tests,
        pareto_table,
        nsga_objectives,
        baseline_table,
    )
    _write_json(reports_dir / "experiment_summary.json", summary)
    _append_report(output_dir / "report.md", summary, pareto_table, baseline_table)


def build_candidate_pool(
    patterns: np.ndarray,
    config: HypergraphConfig,
    seed: int,
) -> tuple[list[tuple[int, ...]], pd.DataFrame]:
    """Строит пул гиперребер из частотных сочетаний и случайного дополнения."""

    rng = np.random.default_rng(seed)
    dimension = int(patterns.shape[1])
    random_target = int(round(config.candidate_pool_size * config.random_fraction))
    frequent_target = max(0, config.candidate_pool_size - random_target)
    frequencies = _joint_activation_frequencies(patterns, config)
    selected: list[tuple[int, ...]] = [edge for edge, _ in frequencies.most_common(frequent_target)]
    selected_set = set(selected)

    attempts = 0
    max_attempts = config.candidate_pool_size * 200
    while len(selected) < config.candidate_pool_size and attempts < max_attempts:
        attempts += 1
        size = int(rng.integers(config.min_hyperedge_size, config.max_hyperedge_size + 1))
        edge = tuple(sorted(rng.choice(dimension, size=size, replace=False).astype(int).tolist()))
        if edge in selected_set:
            continue
        selected.append(edge)
        selected_set.add(edge)

    if len(selected) < config.candidate_pool_size:
        for size in range(config.min_hyperedge_size, config.max_hyperedge_size + 1):
            for edge in itertools.combinations(range(dimension), size):
                if edge in selected_set:
                    continue
                selected.append(edge)
                selected_set.add(edge)
                if len(selected) == config.candidate_pool_size:
                    break
            if len(selected) == config.candidate_pool_size:
                break

    rows = [
        {
            "edge_id": index,
            "symbols": " ".join(str(symbol) for symbol in edge),
            "size": len(edge),
            "joint_activation_frequency": frequencies.get(edge, 0.0),
        }
        for index, edge in enumerate(selected)
    ]
    return selected, pd.DataFrame(rows)


def _joint_activation_frequencies(patterns: np.ndarray, config: HypergraphConfig) -> Counter:
    frequencies: Counter = Counter()
    for pattern in patterns:
        active_symbols = np.flatnonzero(pattern == 1).tolist()
        for size in range(config.min_hyperedge_size, config.max_hyperedge_size + 1):
            if len(active_symbols) < size:
                continue
            for edge in itertools.combinations(active_symbols, size):
                frequencies[tuple(edge)] += 1

    total = max(1, len(patterns))
    for edge in list(frequencies):
        frequencies[edge] = frequencies[edge] / total
    return frequencies


def _save_tables(
    tables_dir: Path,
    patterns: np.ndarray,
    candidate_table: pd.DataFrame,
    pareto_table: pd.DataFrame,
    nsga_objectives: pd.DataFrame,
    baseline_table: pd.DataFrame,
) -> None:
    pd.DataFrame(patterns).to_csv(tables_dir / "patterns.csv", index=False)
    candidate_table.to_csv(tables_dir / "candidate_pool.csv", index=False)
    pareto_table.to_csv(tables_dir / "pareto_solutions.csv", index=False)
    nsga_objectives.to_csv(tables_dir / "nsga_objectives.csv", index=False)
    baseline_table.to_csv(tables_dir / "baseline_metrics.csv", index=False)


def _build_summary(
    generation_metadata: dict[str, Any],
    generation_config: PatternGenerationConfig,
    hypergraph_config: HypergraphConfig,
    optimization_config: OptimizationConfig,
    baseline_config: BaselineConfig,
    candidate_pool: list[tuple[int, ...]],
    tests: list[Any],
    pareto_table: pd.DataFrame,
    nsga_objectives: pd.DataFrame,
    baseline_table: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "generation": generation_metadata,
        "close_pattern_distance_used": generation_config.close_pair_distance,
        "hypergraph": {
            "candidate_pool_size": len(candidate_pool),
            "min_hyperedge_size": hypergraph_config.min_hyperedge_size,
            "max_hyperedge_size": hypergraph_config.max_hyperedge_size,
            "random_fraction": hypergraph_config.random_fraction,
        },
        "tests": {
            "test_count": len(tests),
            "noise_levels": list(generation_config.noise_levels),
            "missing_levels": list(generation_config.missing_levels),
            "trials_per_pattern": generation_config.trials_per_pattern,
        },
        "optimization": {
            "population_size": optimization_config.population_size,
            "n_generations": optimization_config.n_generations,
            "parallel_workers": optimization_config.parallel_workers,
            "min_selected_hyperedges": optimization_config.min_selected_hyperedges,
            "max_selected_hyperedges": optimization_config.max_selected_hyperedges,
            "evaluated_solution_count": int(len(nsga_objectives)),
            "pareto_solution_count": int(len(pareto_table)),
        },
        "baselines": {
            "random_runs": baseline_config.n_random_runs,
            "rows": int(len(baseline_table)),
        },
        "tables": {
            "patterns": "tables/patterns.csv",
            "candidate_pool": "tables/candidate_pool.csv",
            "pareto_solutions": "tables/pareto_solutions.csv",
            "nsga_objectives": "tables/nsga_objectives.csv",
            "baseline_metrics": "tables/baseline_metrics.csv",
        },
        "pareto_summary": _table_summary(pareto_table),
        "baseline_summary": _baseline_summary(baseline_table),
        "hypotheses": _hypothesis_summary(pareto_table, baseline_table),
    }


def _table_summary(table: pd.DataFrame) -> dict[str, Any]:
    if table.empty:
        return {}
    return {
        "best_recovery_accuracy": _max_or_zero(table, "average_recovery_accuracy"),
        "best_noise_robustness": _max_or_zero(table, "noise_robustness"),
        "best_close_pattern_discrimination": _max_or_zero(table, "close_pattern_discrimination"),
        "lowest_complexity": _min_or_zero(table, "complexity_objective"),
        "mean_hyperedge_count": _mean_or_zero(table, "hyperedge_count"),
    }


def _baseline_summary(table: pd.DataFrame) -> dict[str, Any]:
    if table.empty:
        return {}
    grouped = table.groupby("baseline", dropna=False)
    return {
        name: {
            "best_recovery_accuracy": _max_or_zero(group, "average_recovery_accuracy"),
            "best_noise_robustness": _max_or_zero(group, "noise_robustness"),
            "best_close_pattern_discrimination": _max_or_zero(group, "close_pattern_discrimination"),
            "mean_hyperedge_count": _mean_or_zero(group, "hyperedge_count"),
        }
        for name, group in grouped
    }


def _hypothesis_summary(pareto_table: pd.DataFrame, baseline_table: pd.DataFrame) -> dict[str, str]:
    if pareto_table.empty or baseline_table.empty:
        return {
            "H1": "Недостаточно данных для предварительного вывода.",
            "H2": "Недостаточно данных для предварительного вывода.",
            "H3": "Недостаточно данных для предварительного вывода.",
        }

    pareto_accuracy = _max_or_zero(pareto_table, "average_recovery_accuracy")
    pareto_robustness = _max_or_zero(pareto_table, "noise_robustness")
    pareto_close = _max_or_zero(pareto_table, "close_pattern_discrimination")
    baseline_accuracy = _max_or_zero(baseline_table, "average_recovery_accuracy")
    baseline_robustness = _max_or_zero(baseline_table, "noise_robustness")
    baseline_close = _max_or_zero(baseline_table, "close_pattern_discrimination")

    return {
        "H1": "Предварительно поддерживается." if pareto_accuracy >= baseline_accuracy and pareto_robustness >= baseline_robustness else "Пока не подтверждается.",
        "H2": "Требует анализа Парето-фронта и структурных диапазонов.",
        "H3": "Предварительно поддерживается." if pareto_close >= baseline_close else "Пока не подтверждается.",
    }


def _append_report(report_path: Path, summary: dict[str, Any], pareto_table: pd.DataFrame, baseline_table: pd.DataFrame) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "\n\n" if report_path.exists() and report_path.read_text(encoding="utf-8").strip() else ""
    generation = summary["generation"]
    pareto = summary["pareto_summary"]
    baseline = summary["baseline_summary"]
    hypotheses = summary["hypotheses"]

    lines = [
        "## Основной эксперимент",
        "",
        "Сгенерирован набор бинарных паттернов для проверки АГ-памяти. Использовано "
        f"{generation['n_vertices']} вершин, {generation['pattern_count']} паттернов и "
        f"{generation['cluster_count']} кластера. Доля активных битов составила "
        f"{generation['active_fraction']}. Параметр внутрикластерной близости равен "
        f"{generation['close_pattern_distance']}.",
        "",
        "Межкластерное расстояние контролировалось по расстоянию Хэмминга. Исходный целевой "
        f"порог равен {generation['requested_center_distance']}, фактически использованный "
        f"порог равен {generation['used_center_distance']}, нижняя допустимая граница равна "
        f"{generation['minimum_center_distance']}. "
        + (
            f"Порог был снижен: {generation['threshold_reduction_reason']} "
            "Фактически использованный порог остался выше нижней допустимой границы."
            if generation["threshold_was_reduced"]
            else "Снижение порога не потребовалось."
        ),
        "",
        f"Для различения близких паттернов использован порог {summary['close_pattern_distance_used']}. "
        "Этот порог взят из конфигурации генерации данных и сохранён в таблицах решений.",
        "",
        f"Пул гиперребер содержит {summary['hypergraph']['candidate_pool_size']} кандидатов. "
        f"Минимальная кратность равна {summary['hypergraph']['min_hyperedge_size']}, "
        f"максимальная кратность равна {summary['hypergraph']['max_hyperedge_size']}.",
        "",
        f"NSGA-II выполнен с популяцией {summary['optimization']['population_size']} и числом поколений "
        f"{summary['optimization']['n_generations']}. Использовано работников: "
        f"{summary['optimization']['parallel_workers']}. На Парето-фронте получено "
        f"{summary['optimization']['pareto_solution_count']} решений.",
        "",
        "Созданы таблицы: "
        f"{summary['tables']['patterns']}, {summary['tables']['candidate_pool']}, "
        f"{summary['tables']['pareto_solutions']}, {summary['tables']['nsga_objectives']}, "
        f"{summary['tables']['baseline_metrics']}.",
        "",
        "Краткая сводка по Парето-фронту: "
        f"лучшая точность восстановления {pareto.get('best_recovery_accuracy', 0):.3f}, "
        f"лучшая устойчивость {pareto.get('best_noise_robustness', 0):.3f}, "
        f"лучшее различение близких паттернов {pareto.get('best_close_pattern_discrimination', 0):.3f}, "
        f"минимальная структурная сложность {pareto.get('lowest_complexity', 0):.3f}.",
        "",
        "Краткая сводка по базовым топологиям: "
        + _format_baseline_summary(baseline),
        "",
        "Предварительные выводы по гипотезам: "
        f"H1 — {hypotheses['H1']} H2 — {hypotheses['H2']} H3 — {hypotheses['H3']}",
    ]
    with report_path.open("a", encoding="utf-8") as file:
        file.write(prefix + "\n".join(lines) + "\n")


def _format_baseline_summary(summary: dict[str, Any]) -> str:
    if not summary:
        return "нет данных."
    parts = []
    for name, values in summary.items():
        parts.append(
            f"{name}: точность {values['best_recovery_accuracy']:.3f}, "
            f"устойчивость {values['best_noise_robustness']:.3f}, "
            f"различение {values['best_close_pattern_discrimination']:.3f}"
        )
    return "; ".join(parts) + "."


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _max_or_zero(table: pd.DataFrame, column: str) -> float:
    if column not in table or table.empty:
        return 0.0
    return float(table[column].max())


def _min_or_zero(table: pd.DataFrame, column: str) -> float:
    if column not in table or table.empty:
        return 0.0
    return float(table[column].min())


def _mean_or_zero(table: pd.DataFrame, column: str) -> float:
    if column not in table or table.empty:
        return 0.0
    return float(table[column].mean())


def _resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _make_configs(args: argparse.Namespace) -> tuple[ExperimentConfig, PatternGenerationConfig, HypergraphConfig, OptimizationConfig]:
    experiment_config = ExperimentConfig(seed=args.seed, output_dir=args.output_dir)
    generation_config = PatternGenerationConfig(
        pattern_count=args.n_patterns,
        dimension=args.n_vertices,
        active_fraction=args.active_fraction,
        cluster_count=args.n_clusters,
        close_pair_distance=args.close_pattern_distance,
        noise_levels=tuple(args.noise_levels),
        missing_levels=tuple(args.missing_levels),
        trials_per_pattern=args.trials_per_pattern,
        seed=args.seed,
    )
    hypergraph_config = HypergraphConfig(
        candidate_pool_size=args.candidate_pool_size,
        min_hyperedge_size=args.min_hyperedge_size,
        max_hyperedge_size=args.max_hyperedge_size,
        random_fraction=args.random_fraction,
    )
    optimization_config = OptimizationConfig(
        population_size=args.population_size,
        n_generations=args.n_generations,
        crossover_probability=args.crossover_probability,
        mutation_probability=args.mutation_probability,
        min_selected_hyperedges=args.min_selected_hyperedges,
        max_selected_hyperedges=args.max_selected_hyperedges,
        parallel_workers=args.parallel_workers,
        seed=args.seed,
    )
    return experiment_config, generation_config, hypergraph_config, optimization_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск эксперимента АГ-памяти.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--n-generations", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--n-vertices", type=int, default=32)
    parser.add_argument("--n-patterns", type=int, default=24)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--active-fraction", type=float, default=0.25)
    parser.add_argument("--close-pattern-distance", type=int, default=6)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    parser.add_argument("--missing-levels", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--trials-per-pattern", type=int, default=1)
    parser.add_argument("--candidate-pool-size", type=int, default=40)
    parser.add_argument("--min-hyperedge-size", type=int, default=2)
    parser.add_argument("--max-hyperedge-size", type=int, default=4)
    parser.add_argument("--random-fraction", type=float, default=0.25)
    parser.add_argument("--min-selected-hyperedges", type=int, default=2)
    parser.add_argument("--max-selected-hyperedges", type=int, default=20)
    parser.add_argument("--crossover-probability", type=float, default=0.9)
    parser.add_argument("--mutation-probability", type=float, default=None)
    parser.add_argument("--baseline-runs", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main()
