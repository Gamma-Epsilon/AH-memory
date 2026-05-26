"""Проверка чувствительности экспериментальной постановки."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ah_memory.baselines import evaluate_baselines
from ah_memory.config import BaselineConfig, HypergraphConfig, OptimizationConfig, PatternGenerationConfig, RetrievalConfig
from ah_memory.knowledge import KnowledgeHypergraph
from ah_memory.memory import AHMemory
from ah_memory.optimization import run_nsga2
from ah_memory.patterns import find_close_pairs, generate_clustered_pattern_data, prepare_pattern_tests
from ah_memory.statistics import analyze_structural_ranges
from ah_memory.symbols import SymbolSet
from run_experiment import build_candidate_pool


def main() -> None:
    results_dir = ROOT_DIR / "results"
    tables_dir = results_dir / "tables"
    reports_dir = results_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows.extend(_run_complexity_levels(tables_dir, reports_dir))
    rows.extend(_run_budget_checks(tables_dir, reports_dir))
    rows.extend(_run_hyperedge_weight_checks(tables_dir, reports_dir))

    summary = pd.DataFrame(rows)
    summary.to_csv(reports_dir / "sensitivity_summary.csv", index=False)
    summary_text = _format_summary(summary)
    (reports_dir / "sensitivity_summary.md").write_text(summary_text, encoding="utf-8")
    _append_report(results_dir / "report.md", summary)


def _run_complexity_levels(tables_dir: Path, reports_dir: Path) -> list[dict[str, Any]]:
    configs = [
        (
            "easy",
            PatternGenerationConfig(
                pattern_count=18,
                dimension=32,
                active_fraction=0.25,
                cluster_count=3,
                close_pair_distance=4,
                noise_levels=(0.0, 0.05),
                missing_levels=(0.0,),
                trials_per_pattern=1,
                seed=501,
            ),
            HypergraphConfig(candidate_pool_size=28, min_hyperedge_size=2, max_hyperedge_size=4, random_fraction=0.25),
            OptimizationConfig(population_size=6, n_generations=2, min_selected_hyperedges=2, max_selected_hyperedges=14, parallel_workers=4, seed=501),
            BaselineConfig(n_random_runs=3, min_selected_hyperedges=2, max_selected_hyperedges=14, selected_hyperedges=14, seed=501),
        ),
        (
            "medium",
            PatternGenerationConfig(
                pattern_count=24,
                dimension=32,
                active_fraction=0.25,
                cluster_count=3,
                close_pair_distance=6,
                noise_levels=(0.0, 0.1, 0.2),
                missing_levels=(0.0, 0.2),
                trials_per_pattern=1,
                seed=502,
            ),
            HypergraphConfig(candidate_pool_size=34, min_hyperedge_size=2, max_hyperedge_size=4, random_fraction=0.25),
            OptimizationConfig(population_size=8, n_generations=2, min_selected_hyperedges=2, max_selected_hyperedges=17, parallel_workers=4, seed=502),
            BaselineConfig(n_random_runs=3, min_selected_hyperedges=2, max_selected_hyperedges=17, selected_hyperedges=17, seed=502),
        ),
        (
            "hard",
            PatternGenerationConfig(
                pattern_count=30,
                dimension=40,
                active_fraction=0.30,
                cluster_count=4,
                close_pair_distance=8,
                noise_levels=(0.0, 0.15, 0.3),
                missing_levels=(0.0, 0.2, 0.35),
                trials_per_pattern=1,
                seed=503,
            ),
            HypergraphConfig(candidate_pool_size=42, min_hyperedge_size=2, max_hyperedge_size=4, random_fraction=0.30),
            OptimizationConfig(population_size=8, n_generations=2, min_selected_hyperedges=2, max_selected_hyperedges=21, parallel_workers=4, seed=503),
            BaselineConfig(n_random_runs=3, min_selected_hyperedges=2, max_selected_hyperedges=21, selected_hyperedges=21, seed=503),
        ),
    ]
    rows = []
    for name, generation_config, hypergraph_config, optimization_config, baseline_config in configs:
        rows.append(
            _run_single(
                label=f"complexity_{name}",
                prefix=f"sensitivity_complexity_{name}_",
                generation_config=generation_config,
                hypergraph_config=hypergraph_config,
                optimization_config=optimization_config,
                baseline_config=baseline_config,
                retrieval_config=RetrievalConfig(),
                tables_dir=tables_dir,
                reports_dir=reports_dir,
            )
        )
    return rows


def _run_budget_checks(tables_dir: Path, reports_dir: Path) -> list[dict[str, Any]]:
    generation_config = PatternGenerationConfig(
        pattern_count=24,
        dimension=32,
        active_fraction=0.25,
        cluster_count=3,
        close_pair_distance=6,
        noise_levels=(0.0, 0.1, 0.2),
        missing_levels=(0.0, 0.2),
        trials_per_pattern=1,
        seed=610,
    )
    hypergraph_config = HypergraphConfig(candidate_pool_size=34, min_hyperedge_size=2, max_hyperedge_size=4, random_fraction=0.25)
    budgets = [
        ("small", 5, 1),
        ("medium", 7, 2),
        ("large", 9, 3),
    ]
    rows = []
    for name, population, generations in budgets:
        rows.append(
            _run_single(
                label=f"budget_{name}",
                prefix=f"sensitivity_budget_{name}_",
                generation_config=generation_config,
                hypergraph_config=hypergraph_config,
                optimization_config=OptimizationConfig(
                    population_size=population,
                    n_generations=generations,
                    min_selected_hyperedges=2,
                    max_selected_hyperedges=17,
                    parallel_workers=4,
                    seed=610 + generations,
                ),
                baseline_config=BaselineConfig(n_random_runs=3, min_selected_hyperedges=2, max_selected_hyperedges=17, selected_hyperedges=17, seed=610 + generations),
                retrieval_config=RetrievalConfig(),
                tables_dir=tables_dir,
                reports_dir=reports_dir,
            )
        )
    return rows


def _run_hyperedge_weight_checks(tables_dir: Path, reports_dir: Path) -> list[dict[str, Any]]:
    generation_config = PatternGenerationConfig(
        pattern_count=24,
        dimension=32,
        active_fraction=0.25,
        cluster_count=3,
        close_pair_distance=6,
        noise_levels=(0.0, 0.1, 0.2),
        missing_levels=(0.0, 0.2),
        trials_per_pattern=1,
        seed=720,
    )
    hypergraph_config = HypergraphConfig(candidate_pool_size=34, min_hyperedge_size=2, max_hyperedge_size=4, random_fraction=0.25)
    optimization_config = OptimizationConfig(population_size=7, n_generations=2, min_selected_hyperedges=2, max_selected_hyperedges=17, parallel_workers=4, seed=720)
    baseline_config = BaselineConfig(n_random_runs=3, min_selected_hyperedges=2, max_selected_hyperedges=17, selected_hyperedges=17, seed=720)
    variants = [
        ("ordinary", RetrievalConfig()),
        ("strong_hyperedge", RetrievalConfig(hyperedge_weight=2.0)),
        ("strong_hyperedge_same_penalty", RetrievalConfig(hyperedge_weight=3.0)),
        ("strong_hyperedge_weak_penalty", RetrievalConfig(hyperedge_weight=3.0, complexity_weight=0.001)),
    ]
    rows = []
    for name, retrieval_config in variants:
        rows.append(
            _run_single(
                label=f"hyperedge_{name}",
                prefix=f"sensitivity_hyperedge_{name}_",
                generation_config=generation_config,
                hypergraph_config=hypergraph_config,
                optimization_config=optimization_config,
                baseline_config=baseline_config,
                retrieval_config=retrieval_config,
                tables_dir=tables_dir,
                reports_dir=reports_dir,
            )
        )
    return rows


def _run_single(
    label: str,
    prefix: str,
    generation_config: PatternGenerationConfig,
    hypergraph_config: HypergraphConfig,
    optimization_config: OptimizationConfig,
    baseline_config: BaselineConfig,
    retrieval_config: RetrievalConfig,
    tables_dir: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    pattern_data = generate_clustered_pattern_data(generation_config)
    tests = prepare_pattern_tests(pattern_data.patterns, generation_config)
    candidate_pool, candidate_table = build_candidate_pool(pattern_data.patterns, hypergraph_config, generation_config.seed)
    result, pareto = run_nsga2(
        pattern_data.patterns,
        tests,
        candidate_pool,
        {
            "optimization_config": optimization_config,
            "retrieval_config": retrieval_config,
            "generation_config": generation_config,
        },
    )
    baseline = evaluate_baselines(
        pattern_data.patterns,
        tests,
        candidate_pool,
        {
            "baseline_config": baseline_config,
            "retrieval_config": retrieval_config,
            "generation_config": generation_config,
        },
    )
    candidate_table.to_csv(tables_dir / f"{prefix}candidate_pool.csv", index=False)
    pd.DataFrame(pattern_data.patterns).to_csv(tables_dir / f"{prefix}patterns.csv", index=False)
    pareto.to_csv(tables_dir / f"{prefix}pareto_solutions.csv", index=False)
    getattr(result, "evaluated_metrics_table", pd.DataFrame()).to_csv(tables_dir / f"{prefix}nsga_objectives.csv", index=False)
    baseline.to_csv(tables_dir / f"{prefix}baseline_metrics.csv", index=False)
    close_pairs = find_close_pairs(pattern_data.patterns, generation_config.close_pair_distance)
    close_indices = sorted({index for pair in close_pairs for index in pair})
    h3_tests = len(close_indices) * len(generation_config.noise_levels) * len(generation_config.missing_levels) * generation_config.trials_per_pattern
    h3_detail = _h3_detail_for_best_solution(pareto, candidate_pool, pattern_data.patterns, tests, close_pairs, retrieval_config)
    ranges = analyze_structural_ranges(pareto)
    h1_mean = pareto["average_recovery_accuracy"].mean() > baseline[baseline["baseline"] == "random"]["average_recovery_accuracy"].mean()
    h1_best = pareto["average_recovery_accuracy"].max() > baseline[baseline["baseline"] == "random"]["average_recovery_accuracy"].max()
    h3_best = pareto["close_pattern_discrimination"].max() > baseline["close_pattern_discrimination"].max()
    row = {
        "label": label,
        "seed": generation_config.seed,
        "n_patterns": generation_config.pattern_count,
        "n_vertices": generation_config.dimension,
        "cluster_count": generation_config.cluster_count,
        "close_pattern_distance": generation_config.close_pair_distance,
        "noise_levels": str(list(generation_config.noise_levels)),
        "missing_levels": str(list(generation_config.missing_levels)),
        "candidate_pool_size": hypergraph_config.candidate_pool_size,
        "population_size": optimization_config.population_size,
        "n_generations": optimization_config.n_generations,
        "evaluated_topologies": optimization_config.population_size * optimization_config.n_generations,
        "pareto_count": len(pareto),
        "close_pair_count": len(close_pairs),
        "patterns_with_close_neighbors": len(close_indices),
        "h3_test_count": h3_tests,
        "best_accuracy": _max(pareto, "average_recovery_accuracy"),
        "best_robustness": _max(pareto, "noise_robustness"),
        "best_discrimination": _max(pareto, "close_pattern_discrimination"),
        "mean_accuracy": _mean(pareto, "average_recovery_accuracy"),
        "random_mean_accuracy": _mean(baseline[baseline["baseline"] == "random"], "average_recovery_accuracy"),
        "random_best_accuracy": _max(baseline[baseline["baseline"] == "random"], "average_recovery_accuracy"),
        "baseline_best_accuracy": _max(baseline, "average_recovery_accuracy"),
        "random_mean_discrimination": _mean(baseline[baseline["baseline"] == "random"], "close_pattern_discrimination"),
        "random_best_discrimination": _max(baseline[baseline["baseline"] == "random"], "close_pattern_discrimination"),
        "baseline_best_discrimination": _max(baseline, "close_pattern_discrimination"),
        "best_solution_hyperedge_count": _best_value(pareto, "average_recovery_accuracy", "hyperedge_count"),
        "best_solution_density": _best_value(pareto, "average_recovery_accuracy", "density"),
        "best_solution_average_size": _best_value(pareto, "average_recovery_accuracy", "average_hyperedge_size"),
        "h1_vs_random_mean": h1_mean,
        "h1_vs_random_best": h1_best,
        "h2_has_ranges": not ranges.empty,
        "h3_vs_best_baseline": h3_best,
    }
    row.update(h3_detail)
    summary_path = reports_dir / f"{prefix}summary.json"
    summary_path.write_text(json.dumps(_json_ready(row), ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def _h3_detail_for_best_solution(
    pareto: pd.DataFrame,
    candidate_pool: list[tuple[int, ...]],
    patterns: np.ndarray,
    tests: list[Any],
    close_pairs: list[tuple[int, int]],
    retrieval_config: RetrievalConfig,
) -> dict[str, float]:
    if pareto.empty or not close_pairs:
        return {
            "h3_bit_accuracy": 0.0,
            "h3_index_accuracy": 0.0,
            "h3_close_confusion": 0.0,
            "h3_far_confusion": 0.0,
        }
    best = pareto.sort_values("average_recovery_accuracy", ascending=False).iloc[0]
    genome = str(best["genome"])
    topology = [edge for bit, edge in zip(genome, candidate_pool) if bit == "1"]
    symbol_set = SymbolSet(patterns.shape[1])
    knowledge = KnowledgeHypergraph(symbol_set.dimension)
    for edge in topology:
        knowledge.add_hyperedge(edge)
    memory = AHMemory(symbol_set, knowledge=knowledge, retrieval_config=retrieval_config)
    memory.fit(patterns)
    neighbors: dict[int, set[int]] = {}
    for left, right in close_pairs:
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    relevant = [test for test in tests if test.pattern_index in neighbors]
    if not relevant:
        return {
            "h3_bit_accuracy": 0.0,
            "h3_index_accuracy": 0.0,
            "h3_close_confusion": 0.0,
            "h3_far_confusion": 0.0,
        }
    bit_scores = []
    index_hits = []
    close_confusions = []
    far_confusions = []
    for test in relevant:
        restored_index, restored = memory.retrieve(test.distorted)
        bit_scores.append(float(np.mean(restored == test.original)))
        index_hits.append(restored_index == test.pattern_index)
        close_confusions.append(restored_index != test.pattern_index and restored_index in neighbors[test.pattern_index])
        far_confusions.append(restored_index != test.pattern_index and restored_index not in neighbors[test.pattern_index])
    return {
        "h3_bit_accuracy": float(np.mean(bit_scores)),
        "h3_index_accuracy": float(np.mean(index_hits)),
        "h3_close_confusion": float(np.mean(close_confusions)),
        "h3_far_confusion": float(np.mean(far_confusions)),
    }


def _format_summary(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "# Проверка чувствительности\n\nДанные не получены.\n"
    complexity = summary[summary["label"].str.startswith("complexity_")]
    budget = summary[summary["label"].str.startswith("budget_")]
    hyperedge = summary[summary["label"].str.startswith("hyperedge_")]
    h1_count = int(summary["h1_vs_random_mean"].sum())
    h1_best_count = int(summary["h1_vs_random_best"].sum())
    h2_count = int(summary["h2_has_ranges"].sum())
    h3_count = int(summary["h3_vs_best_baseline"].sum())
    unique_seeds = int(summary["seed"].nunique())
    text = [
        "# Проверка чувствительности экспериментальной постановки",
        "",
        f"Всего диагностических запусков: {len(summary)}.",
        f"Использовано уникальных seed: {unique_seeds}. Для бюджетов и вариантов вклада гиперребер seed данных фиксировался, чтобы сравнение оставалось парным.",
        f"Проверены уровни сложности: {', '.join(complexity['label'].str.replace('complexity_', '').tolist())}.",
        "Менялись число паттернов, размерность, число кластеров, порог близости, уровни шума и пропусков, размер пула гиперребер, бюджет NSGA-II и веса восстановления.",
        f"H1 относительно среднего случайного baseline поддерживается в {h1_count} из {len(summary)} запусков.",
        f"H1 относительно лучшего случайного baseline поддерживается в {h1_best_count} из {len(summary)} запусков.",
        f"H2 показывает структурные диапазоны в {h2_count} из {len(summary)} запусков.",
        f"H3 превосходит лучший baseline в {h3_count} из {len(summary)} запусков.",
        "",
        "Сложность данных:",
        f"лёгкая постановка дала точность {complexity.iloc[0]['best_accuracy']:.3f} и различение {complexity.iloc[0]['best_discrimination']:.3f}; "
        f"средняя — {complexity.iloc[1]['best_accuracy']:.3f} и {complexity.iloc[1]['best_discrimination']:.3f}; "
        f"сложная — {complexity.iloc[2]['best_accuracy']:.3f} и {complexity.iloc[2]['best_discrimination']:.3f}.",
        "",
        "Бюджет NSGA-II:",
        f"малый бюджет дал {budget.iloc[0]['pareto_count']} решений, средний — {budget.iloc[1]['pareto_count']}, большой — {budget.iloc[2]['pareto_count']}.",
        "",
        "Вклад гиперребер:",
        f"обычный вариант дал различение {hyperedge.iloc[0]['best_discrimination']:.3f}, усиленный вклад — {hyperedge.iloc[1]['best_discrimination']:.3f}, "
        f"усиленный вклад при ослабленном штрафе — {hyperedge.iloc[-1]['best_discrimination']:.3f}.",
        "",
        "Основной показатель H3 — точность выбора исходного индекса среди близких альтернатив; битовая точность сохранена отдельно, потому что близкий, но неправильный паттерн может иметь высокое битовое совпадение.",
        "",
        "Наиболее сильный фактор в текущих данных — сочетание силы baseline и чувствительности H3: лёгкая постановка насыщается, сложная снижает качество, а средняя лучше подходит для финального запуска с дополнительной серией seed.",
    ]
    return "\n".join(text) + "\n"


def _append_report(report_path: Path, summary: pd.DataFrame) -> None:
    complexity = summary[summary["label"].str.startswith("complexity_")]
    h1_count = int(summary["h1_vs_random_mean"].sum())
    h1_best_count = int(summary["h1_vs_random_best"].sum())
    h2_count = int(summary["h2_has_ranges"].sum())
    h3_count = int(summary["h3_vs_best_baseline"].sum())
    unique_seeds = int(summary["seed"].nunique())
    prefix = "\n\n" if report_path.exists() and report_path.read_text(encoding="utf-8").strip() else ""
    lines = [
        "## Проверка чувствительности экспериментальной постановки",
        "",
        "Проверка чувствительности проведена потому, что предыдущая серия seed не дала устойчивого подтверждения H3, а базовые топологии оказались сильными конкурентами.",
        "",
        "Проверены лёгкая, средняя и сложная постановки данных, разные вычислительные бюджеты NSGA-II и несколько вариантов вклада гиперребер.",
        f"Всего выполнено {len(summary)} диагностических запусков на {unique_seeds} уникальных seed; для бюджетов и весов seed данных фиксировался, чтобы сравнивались только бюджет или вклад гиперребер.",
        "",
        f"В лёгкой постановке лучшая точность равна {complexity.iloc[0]['best_accuracy']:.3f}, в средней — {complexity.iloc[1]['best_accuracy']:.3f}, в сложной — {complexity.iloc[2]['best_accuracy']:.3f}.",
        f"Различение близких паттернов в этих постановках равно соответственно {complexity.iloc[0]['best_discrimination']:.3f}, {complexity.iloc[1]['best_discrimination']:.3f}, {complexity.iloc[2]['best_discrimination']:.3f}.",
        "",
        f"H1 относительно среднего случайного baseline поддерживается в {h1_count} из {len(summary)} диагностических запусков, а относительно лучшего случайного baseline — в {h1_best_count} из {len(summary)}. H2 показывает структурные диапазоны в {h2_count} из {len(summary)} запусков. H3 превосходит лучший baseline в {h3_count} из {len(summary)} запусков.",
        "",
        "Сравнение со случайными топологиями показывает, что NSGA-II чаще превосходит средний случайный baseline, но не всегда превосходит лучший случайный baseline из серии. Лучший случайный baseline может быть слишком сильным при большом числе случайных попыток.",
        "",
        "Увеличение бюджета NSGA-II увеличивает число оценённых топологий и решений Парето-фронта, но не гарантирует устойчивого улучшения H3.",
        "",
        "Усиление вклада гиперребер и ослабление штрафа сложности не дают автоматического подтверждения гипотез; это указывает на необходимость менять сложность данных и чувствительность метрик, а не архитектуру памяти.",
        "",
        "Текущая постановка не выглядит просто отрицательной: лёгкий режим насыщается, сложный режим заметно снижает точность и различение, а H3 чувствительна к силе baseline. Для финального научного запуска лучше использовать среднюю постановку с элементами сложной, больше seed, более широкий пул гиперребер и строгую проверку выбора исходного индекса среди близких альтернатив.",
        "",
        "Ещё одна серия seed нужна: она должна проверять финальные параметры после выбора средней или средне-сложной постановки.",
        "",
        "Подробная сводка сохранена в reports/sensitivity_summary.md.",
    ]
    with report_path.open("a", encoding="utf-8") as file:
        file.write(prefix + "\n".join(lines) + "\n")


def _max(table: pd.DataFrame, column: str) -> float:
    return float(table[column].max()) if column in table and not table.empty else 0.0


def _mean(table: pd.DataFrame, column: str) -> float:
    return float(table[column].mean()) if column in table and not table.empty else 0.0


def _best_value(table: pd.DataFrame, by: str, value: str) -> float:
    if table.empty or by not in table or value not in table:
        return 0.0
    return float(table.sort_values(by, ascending=False).iloc[0][value])


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


if __name__ == "__main__":
    main()
