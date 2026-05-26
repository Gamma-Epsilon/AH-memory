"""Финальная серия seed перед итоговым отчётом."""

from __future__ import annotations

import json
import sys
import time
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
from ah_memory.statistics import analyze_structural_ranges, run_statistical_analysis
from ah_memory.symbols import SymbolSet
from run_experiment import build_candidate_pool


FINAL_SEEDS = [801, 802, 803, 804, 805]


def main() -> None:
    results_dir = ROOT_DIR / "results"
    tables_dir = results_dir / "tables"
    reports_dir = results_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in FINAL_SEEDS:
        rows.append(_run_seed(seed, tables_dir, reports_dir))

    summary = pd.DataFrame(rows)
    summary.to_csv(reports_dir / "final_summary.csv", index=False)
    (reports_dir / "final_summary.md").write_text(_format_summary(summary), encoding="utf-8")
    _append_report(results_dir / "report.md", summary)


def _run_seed(seed: int, tables_dir: Path, reports_dir: Path) -> dict[str, Any]:
    prefix = f"final_seed_{seed}_"
    summary_path = reports_dir / f"{prefix}summary.json"
    if summary_path.exists():
        row = json.loads(summary_path.read_text(encoding="utf-8"))
        if "baseline_mean_h3_index_accuracy" not in row:
            row = _augment_existing_seed(row, prefix, tables_dir, reports_dir)
        return row

    generation_config = PatternGenerationConfig(
        pattern_count=28,
        dimension=36,
        active_fraction=0.27,
        cluster_count=4,
        close_pair_distance=7,
        noise_levels=(0.0, 0.10, 0.20),
        missing_levels=(0.0, 0.20),
        trials_per_pattern=1,
        seed=seed,
    )
    hypergraph_config = HypergraphConfig(candidate_pool_size=60, min_hyperedge_size=2, max_hyperedge_size=4, random_fraction=0.30)
    optimization_config = OptimizationConfig(
        population_size=20,
        n_generations=15,
        min_selected_hyperedges=2,
        max_selected_hyperedges=30,
        parallel_workers=4,
        seed=seed,
    )
    baseline_config = BaselineConfig(
        n_random_runs=10,
        min_selected_hyperedges=2,
        max_selected_hyperedges=30,
        selected_hyperedges=30,
        seed=seed,
    )
    retrieval_config = RetrievalConfig()
    started = time.perf_counter()

    pattern_data = generate_clustered_pattern_data(generation_config)
    tests = prepare_pattern_tests(pattern_data.patterns, generation_config)
    candidate_pool, candidate_table = build_candidate_pool(pattern_data.patterns, hypergraph_config, seed)
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
    nsga_objectives = getattr(result, "evaluated_metrics_table", pd.DataFrame())

    pd.DataFrame(pattern_data.patterns).to_csv(tables_dir / f"{prefix}patterns.csv", index=False)
    candidate_table.to_csv(tables_dir / f"{prefix}candidate_pool.csv", index=False)
    pareto.to_csv(tables_dir / f"{prefix}pareto_solutions.csv", index=False)
    nsga_objectives.to_csv(tables_dir / f"{prefix}nsga_objectives.csv", index=False)
    baseline.to_csv(tables_dir / f"{prefix}baseline_metrics.csv", index=False)
    statistics = run_statistical_analysis(pareto, baseline, reports_dir, file_prefix=prefix)

    close_pairs = find_close_pairs(pattern_data.patterns, generation_config.close_pair_distance)
    close_indices = sorted({index for pair in close_pairs for index in pair})
    h3_test_count = len(close_indices) * len(generation_config.noise_levels) * len(generation_config.missing_levels) * generation_config.trials_per_pattern
    h3_detail = _h3_detail_for_best_solution(pareto, candidate_pool, pattern_data.patterns, tests, close_pairs, retrieval_config)
    structural_ranges = analyze_structural_ranges(pareto)
    random_baseline = baseline[baseline["baseline"] == "random"]
    top_ranges = _top_ranges(pareto)

    row = {
        "seed": seed,
        "runtime_seconds": time.perf_counter() - started,
        "n_patterns": generation_config.pattern_count,
        "n_vertices": generation_config.dimension,
        "cluster_count": generation_config.cluster_count,
        "close_pattern_distance": generation_config.close_pair_distance,
        "noise_levels": str(list(generation_config.noise_levels)),
        "missing_levels": str(list(generation_config.missing_levels)),
        "candidate_pool_size": hypergraph_config.candidate_pool_size,
        "random_baseline_runs": baseline_config.n_random_runs,
        "population_size": optimization_config.population_size,
        "n_generations": optimization_config.n_generations,
        "evaluated_topologies": int(len(nsga_objectives)),
        "pareto_count": int(len(pareto)),
        "close_pair_count": len(close_pairs),
        "patterns_with_close_neighbors": len(close_indices),
        "h3_test_count": h3_test_count,
        "best_accuracy": _max(pareto, "average_recovery_accuracy"),
        "mean_accuracy": _mean(pareto, "average_recovery_accuracy"),
        "best_robustness": _max(pareto, "noise_robustness"),
        "mean_robustness": _mean(pareto, "noise_robustness"),
        "best_h3_index_accuracy": _max(pareto, "close_pattern_discrimination"),
        "mean_h3_index_accuracy": _mean(pareto, "close_pattern_discrimination"),
        "random_mean_accuracy": _mean(random_baseline, "average_recovery_accuracy"),
        "random_best_accuracy": _max(random_baseline, "average_recovery_accuracy"),
        "random_mean_robustness": _mean(random_baseline, "noise_robustness"),
        "random_best_robustness": _max(random_baseline, "noise_robustness"),
        "random_mean_h3_index_accuracy": _mean(random_baseline, "close_pattern_discrimination"),
        "random_best_h3_index_accuracy": _max(random_baseline, "close_pattern_discrimination"),
        "baseline_best_accuracy": _max(baseline, "average_recovery_accuracy"),
        "baseline_best_robustness": _max(baseline, "noise_robustness"),
        "baseline_mean_h3_index_accuracy": _mean(baseline, "close_pattern_discrimination"),
        "baseline_best_h3_index_accuracy": _max(baseline, "close_pattern_discrimination"),
        "h1_vs_random_mean_accuracy": _mean(pareto, "average_recovery_accuracy") > _mean(random_baseline, "average_recovery_accuracy"),
        "h1_vs_random_mean_robustness": _mean(pareto, "noise_robustness") > _mean(random_baseline, "noise_robustness"),
        "h1_vs_random_best_accuracy": _max(pareto, "average_recovery_accuracy") > _max(random_baseline, "average_recovery_accuracy"),
        "h1_vs_random_best_robustness": _max(pareto, "noise_robustness") > _max(random_baseline, "noise_robustness"),
        "h2_supported": _h2_supported(structural_ranges),
        "h3_vs_random_mean": _mean(pareto, "close_pattern_discrimination") > _mean(random_baseline, "close_pattern_discrimination"),
        "h3_vs_random_best": _max(pareto, "close_pattern_discrimination") > _max(random_baseline, "close_pattern_discrimination"),
        "h3_vs_baseline_mean": _mean(pareto, "close_pattern_discrimination") > _mean(baseline, "close_pattern_discrimination"),
        "h3_vs_best_baseline": _max(pareto, "close_pattern_discrimination") > _max(baseline, "close_pattern_discrimination"),
        "best_solution_hyperedge_count": _best_value(pareto, "average_recovery_accuracy", "hyperedge_count"),
        "best_solution_density": _best_value(pareto, "average_recovery_accuracy", "density"),
        "best_solution_average_size": _best_value(pareto, "average_recovery_accuracy", "average_hyperedge_size"),
        "best_solution_degree_variance": _best_value(pareto, "average_recovery_accuracy", "vertex_degree_variance"),
        "top_hyperedge_count_min": top_ranges.get("hyperedge_count_min", 0.0),
        "top_hyperedge_count_max": top_ranges.get("hyperedge_count_max", 0.0),
        "top_average_size_min": top_ranges.get("average_hyperedge_size_min", 0.0),
        "top_average_size_max": top_ranges.get("average_hyperedge_size_max", 0.0),
        "top_density_min": top_ranges.get("density_min", 0.0),
        "top_density_max": top_ranges.get("density_max", 0.0),
        "top_degree_variance_min": top_ranges.get("vertex_degree_variance_min", 0.0),
        "top_degree_variance_max": top_ranges.get("vertex_degree_variance_max", 0.0),
        "hypothesis_test_count": statistics["test_count"],
        "hypothesis_range_count": statistics["range_count"],
    }
    row.update(h3_detail)
    summary_path.write_text(json.dumps(_json_ready(row), ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def _augment_existing_seed(row: dict[str, Any], prefix: str, tables_dir: Path, reports_dir: Path) -> dict[str, Any]:
    pareto = pd.read_csv(tables_dir / f"{prefix}pareto_solutions.csv")
    baseline = pd.read_csv(tables_dir / f"{prefix}baseline_metrics.csv")
    row["baseline_mean_h3_index_accuracy"] = _mean(baseline, "close_pattern_discrimination")
    row["h3_vs_baseline_mean"] = _mean(pareto, "close_pattern_discrimination") > _mean(baseline, "close_pattern_discrimination")
    row["h3_vs_best_baseline"] = _max(pareto, "close_pattern_discrimination") > _max(baseline, "close_pattern_discrimination")
    (reports_dir / f"{prefix}summary.json").write_text(json.dumps(_json_ready(row), ensure_ascii=False, indent=2), encoding="utf-8")
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
        return _empty_h3_detail()
    best = pareto.sort_values("close_pattern_discrimination", ascending=False).iloc[0]
    topology = [edge for bit, edge in zip(str(best["genome"]), candidate_pool) if bit == "1"]
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
        return _empty_h3_detail()

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
        return "# Финальная серия seed\n\nДанные не получены.\n"
    seeds = ", ".join(str(seed) for seed in summary["seed"].tolist())
    h1_mean = int((summary["h1_vs_random_mean_accuracy"] & summary["h1_vs_random_mean_robustness"]).sum())
    h1_best = int((summary["h1_vs_random_best_accuracy"] & summary["h1_vs_random_best_robustness"]).sum())
    h2 = int(summary["h2_supported"].sum())
    h3_mean = int(summary["h3_vs_random_mean"].sum())
    h3_best = int(summary["h3_vs_random_best"].sum())
    h3_baseline_mean = int(summary["h3_vs_baseline_mean"].sum())
    h3_baseline_best = int(summary["h3_vs_best_baseline"].sum())
    lines = [
        "# Финальная серия seed",
        "",
        f"Seed: {seeds}.",
        "Финальная постановка: 28 паттернов, 36 вершин, 4 кластера, порог близости 7, шум 0.0/0.10/0.20, пропуски 0.0/0.20, пул 60 гиперребер.",
        "Бюджет NSGA-II: population 20, generations 15, параллельные вычисления, 10 случайных baseline на seed.",
        "",
        f"Accuracy: среднее {summary['best_accuracy'].mean():.3f}, разброс {summary['best_accuracy'].std(ddof=0):.3f}.",
        f"Robustness: среднее {summary['best_robustness'].mean():.3f}, разброс {summary['best_robustness'].std(ddof=0):.3f}.",
        f"H3 index accuracy: среднее {summary['h3_index_accuracy'].mean():.3f}, разброс {summary['h3_index_accuracy'].std(ddof=0):.3f}.",
        f"H3 bit accuracy: среднее {summary['h3_bit_accuracy'].mean():.3f}, разброс {summary['h3_bit_accuracy'].std(ddof=0):.3f}.",
        "",
        f"H1 относительно среднего random baseline поддерживается в {h1_mean} из {len(summary)} seed.",
        f"H1 относительно лучшего random baseline поддерживается в {h1_best} из {len(summary)} seed.",
        f"H2 поддерживается в {h2} из {len(summary)} seed.",
        f"H3 относительно среднего random baseline поддерживается в {h3_mean} из {len(summary)} seed.",
        f"H3 относительно лучшего random baseline поддерживается в {h3_best} из {len(summary)} seed.",
        f"H3 относительно среднего значения всех baseline поддерживается в {h3_baseline_mean} из {len(summary)} seed.",
        f"H3 относительно лучшего baseline поддерживается в {h3_baseline_best} из {len(summary)} seed.",
        "",
        f"Структурные диапазоны лучших решений: число гиперребер {summary['top_hyperedge_count_min'].min():.0f}-{summary['top_hyperedge_count_max'].max():.0f}, "
        f"средняя кратность {summary['top_average_size_min'].min():.3f}-{summary['top_average_size_max'].max():.3f}, "
        f"плотность {summary['top_density_min'].min():.3f}-{summary['top_density_max'].max():.3f}.",
        "",
        _final_conclusion(h1_mean, h1_best, h2, h3_baseline_mean, h3_baseline_best, len(summary)),
    ]
    return "\n".join(lines) + "\n"


def _append_report(report_path: Path, summary: pd.DataFrame) -> None:
    h1_mean = int((summary["h1_vs_random_mean_accuracy"] & summary["h1_vs_random_mean_robustness"]).sum())
    h1_best = int((summary["h1_vs_random_best_accuracy"] & summary["h1_vs_random_best_robustness"]).sum())
    h2 = int(summary["h2_supported"].sum())
    h3_mean = int(summary["h3_vs_random_mean"].sum())
    h3_best = int(summary["h3_vs_random_best"].sum())
    h3_baseline_mean = int(summary["h3_vs_baseline_mean"].sum())
    h3_baseline_best = int(summary["h3_vs_best_baseline"].sum())
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    marker = "## Финальная серия seed"
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip()
    prefix = "\n\n" if existing.strip() else ""
    lines = [
        "## Финальная серия seed",
        "",
        "Финальная серия запущена на средне-сложной постановке, потому что проверка чувствительности показала насыщение лёгкого режима и снижение качества в сложном режиме.",
        "",
        "Использованы 28 паттернов, 36 вершин, 4 кластера, порог близости 7, уровни шума 0.0/0.10/0.20 и пропусков 0.0/0.20. Пул кандидатных гиперребер равен 60.",
        "Бюджет NSGA-II: population 20, generations 15, параллельные вычисления. Для каждого seed использовано 10 случайных baseline.",
        f"Запущено seed: {', '.join(str(seed) for seed in summary['seed'].tolist())}.",
        "",
        f"H1 относительно среднего random baseline поддерживается в {h1_mean} из {len(summary)} seed. H1 относительно лучшего random baseline поддерживается в {h1_best} из {len(summary)} seed.",
        f"H2 поддерживается в {h2} из {len(summary)} seed: у верхней четверти решений сохраняются диапазоны числа гиперребер, средней кратности, плотности и распределения степеней.",
        f"H3 по индексной метрике относительно среднего random baseline поддерживается в {h3_mean} из {len(summary)} seed, относительно лучшего random baseline — в {h3_best} из {len(summary)} seed.",
        f"Относительно среднего значения всех baseline H3 поддерживается в {h3_baseline_mean} из {len(summary)} seed, относительно лучшего baseline — в {h3_baseline_best} из {len(summary)} seed.",
        "",
        f"Средняя лучшая accuracy равна {summary['best_accuracy'].mean():.3f}, robustness — {summary['best_robustness'].mean():.3f}, H3 index accuracy — {summary['h3_index_accuracy'].mean():.3f}, H3 bit accuracy — {summary['h3_bit_accuracy'].mean():.3f}.",
        f"Структурные диапазоны лучших решений: число гиперребер {summary['top_hyperedge_count_min'].min():.0f}-{summary['top_hyperedge_count_max'].max():.0f}, средняя кратность {summary['top_average_size_min'].min():.3f}-{summary['top_average_size_max'].max():.3f}, плотность {summary['top_density_min'].min():.3f}-{summary['top_density_max'].max():.3f}.",
        "",
        "Для дальнейшей научной статьи можно использовать осторожный вывод: структурная гипотеза H2 наиболее устойчива, H1 зависит от строгости baseline, H3 чувствительна к сравнению с лучшими простыми топологиями и должна трактоваться по индексу исходного паттерна.",
        "",
        "Ограничения сохраняются: серия включает 5 seed, а лучший random baseline остаётся сильным конкурентом при фиксированном бюджете случайных попыток.",
        "",
        "Подробная сводка сохранена в reports/final_summary.md.",
    ]
    report_path.write_text(existing + prefix + "\n".join(lines) + "\n", encoding="utf-8")


def _final_conclusion(h1_mean: int, h1_best: int, h2: int, h3_mean: int, h3_best: int, total: int) -> str:
    h1 = "H1 подтверждается относительно среднего random baseline" if h1_mean == total else "H1 подтверждается частично относительно среднего random baseline"
    h1 += "; относительно лучшего random baseline вывод частичный." if h1_best < total else "; относительно лучшего random baseline подтверждается."
    h2_text = "H2 подтверждается устойчиво." if h2 == total else "H2 подтверждается частично."
    h3_text = "H3 подтверждается относительно среднего baseline" if h3_mean == total else "H3 подтверждается частично относительно среднего baseline"
    h3_text += "; относительно лучшего baseline вывод частичный." if h3_best < total else "; относительно лучшего baseline подтверждается."
    return f"Итоговый вывод: {h1} {h2_text} {h3_text}"


def _top_ranges(pareto: pd.DataFrame) -> dict[str, float]:
    if pareto.empty:
        return {}
    top = pareto[pareto["average_recovery_accuracy"] >= pareto["average_recovery_accuracy"].quantile(0.75)]
    rows = {}
    for metric in ["hyperedge_count", "average_hyperedge_size", "density", "vertex_degree_variance"]:
        if metric not in top:
            continue
        rows[f"{metric}_min"] = float(top[metric].min())
        rows[f"{metric}_max"] = float(top[metric].max())
    return rows


def _h2_supported(ranges: pd.DataFrame) -> bool:
    return not ranges.empty and {"hyperedge_count", "average_hyperedge_size", "density"}.issubset(set(ranges["metric"]))


def _empty_h3_detail() -> dict[str, float]:
    return {
        "h3_bit_accuracy": 0.0,
        "h3_index_accuracy": 0.0,
        "h3_close_confusion": 0.0,
        "h3_far_confusion": 0.0,
    }


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
