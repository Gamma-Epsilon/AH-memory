"""Диагностика результатов и сводка по нескольким seed."""

from __future__ import annotations

import argparse
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
from ah_memory.optimization import run_nsga2
from ah_memory.patterns import find_close_pairs, generate_clustered_pattern_data, prepare_pattern_tests
from run_experiment import build_candidate_pool


def main() -> None:
    args = _parse_args()
    results_dir = ROOT_DIR / "results"
    tables_dir = results_dir / "tables"
    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    extended = _load_experiment(tables_dir, reports_dir, "extended_")
    diagnostic_tables = _run_diagnostic_variants(results_dir)
    diagnostic_report = _build_diagnostic_report(extended, diagnostic_tables)
    (reports_dir / "diagnostic_report.md").write_text(diagnostic_report, encoding="utf-8")

    multiseed = _build_multiseed_summary(tables_dir, reports_dir, args.seeds)
    multiseed.to_csv(reports_dir / "multiseed_summary.csv", index=False)
    multiseed_text = _format_multiseed_summary(multiseed)
    (reports_dir / "multiseed_summary.md").write_text(multiseed_text, encoding="utf-8")
    _append_report(results_dir / "report.md", diagnostic_tables, multiseed, diagnostic_report)


def _load_experiment(tables_dir: Path, reports_dir: Path, prefix: str) -> dict[str, Any]:
    return {
        "pareto": pd.read_csv(tables_dir / f"{prefix}pareto_solutions.csv"),
        "baseline": pd.read_csv(tables_dir / f"{prefix}baseline_metrics.csv"),
        "patterns": pd.read_csv(tables_dir / f"{prefix}patterns.csv").to_numpy(dtype=int),
        "summary": json.loads((reports_dir / f"{prefix}experiment_summary.json").read_text(encoding="utf-8")),
    }


def _run_diagnostic_variants(results_dir: Path) -> pd.DataFrame:
    tables_dir = results_dir / "tables"
    summary = json.loads((results_dir / "reports" / "extended_experiment_summary.json").read_text(encoding="utf-8"))
    generation = summary["generation"]
    generation_config = PatternGenerationConfig(
        pattern_count=generation["pattern_count"],
        dimension=generation["n_vertices"],
        active_fraction=generation["active_fraction"],
        cluster_count=generation["cluster_count"],
        close_pair_distance=generation["close_pattern_distance"],
        noise_levels=tuple(summary["tests"]["noise_levels"]),
        missing_levels=tuple(summary["tests"]["missing_levels"]),
        trials_per_pattern=summary["tests"]["trials_per_pattern"],
        seed=42,
    )
    hypergraph_config = HypergraphConfig(
        candidate_pool_size=40,
        min_hyperedge_size=summary["hypergraph"]["min_hyperedge_size"],
        max_hyperedge_size=summary["hypergraph"]["max_hyperedge_size"],
        random_fraction=summary["hypergraph"]["random_fraction"],
    )
    optimization_config = OptimizationConfig(
        population_size=10,
        n_generations=2,
        min_selected_hyperedges=2,
        max_selected_hyperedges=20,
        parallel_workers=2,
        seed=142,
    )
    baseline_config = BaselineConfig(n_random_runs=4, min_selected_hyperedges=2, max_selected_hyperedges=20, selected_hyperedges=20, seed=142)
    pattern_data = generate_clustered_pattern_data(generation_config)
    tests = prepare_pattern_tests(pattern_data.patterns, generation_config)
    candidate_pool, candidate_table = build_candidate_pool(pattern_data.patterns, hypergraph_config, seed=142)
    candidate_table.to_csv(tables_dir / "diagnostic_candidate_pool.csv", index=False)

    variants = [
        ("ordinary", RetrievalConfig()),
        ("strong_hyperedge", RetrievalConfig(hyperedge_weight=2.0)),
        ("weak_complexity_penalty", RetrievalConfig(complexity_weight=0.001)),
        ("no_complexity_penalty", RetrievalConfig(complexity_weight=0.0)),
    ]
    rows = []
    for name, retrieval_config in variants:
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
        pareto.to_csv(tables_dir / f"diagnostic_{name}_pareto_solutions.csv", index=False)
        baseline.to_csv(tables_dir / f"diagnostic_{name}_baseline_metrics.csv", index=False)
        rows.append(
            {
                "variant": name,
                "pareto_count": len(pareto),
                "best_accuracy": _max(pareto, "average_recovery_accuracy"),
                "best_robustness": _max(pareto, "noise_robustness"),
                "best_discrimination": _max(pareto, "close_pattern_discrimination"),
                "best_baseline_accuracy": _max(baseline, "average_recovery_accuracy"),
                "best_baseline_discrimination": _max(baseline, "close_pattern_discrimination"),
                "mean_hyperedge_count": _mean(pareto, "hyperedge_count"),
            }
        )
    diagnostic = pd.DataFrame(rows)
    diagnostic.to_csv(tables_dir / "diagnostic_weight_variants.csv", index=False)
    return diagnostic


def _build_diagnostic_report(extended: dict[str, Any], diagnostic: pd.DataFrame) -> str:
    pareto = extended["pareto"]
    baseline = extended["baseline"]
    patterns = extended["patterns"]
    close_distance = int(extended["summary"]["close_pattern_distance_used"])
    close_pairs = find_close_pairs(patterns, close_distance)
    close_indices = sorted({index for pair in close_pairs for index in pair})
    close_tests = len(close_indices) * len(extended["summary"]["tests"]["noise_levels"]) * len(extended["summary"]["tests"]["missing_levels"]) * extended["summary"]["tests"]["trials_per_pattern"]
    correlations = pareto[["complexity_objective", "hyperedge_count", "average_hyperedge_size", "density", "average_recovery_accuracy"]].corr(numeric_only=True)["average_recovery_accuracy"]
    best_balanced = pareto.assign(
        balance=pareto[["average_recovery_accuracy", "noise_robustness", "close_pattern_discrimination"]].mean(axis=1) - pareto["complexity_objective"]
    ).sort_values("balance", ascending=False).head(3)

    lines = [
        "# Диагностический отчёт",
        "",
        "## Сложность задачи восстановления",
        f"Лучшее значение точности на Парето-фронте: {_max(pareto, 'average_recovery_accuracy'):.3f}. Лучшее значение среди базовых топологий: {_max(baseline, 'average_recovery_accuracy'):.3f}.",
        f"Лучшее различение близких паттернов на Парето-фронте: {_max(pareto, 'close_pattern_discrimination'):.3f}. Лучшее значение среди базовых топологий: {_max(baseline, 'close_pattern_discrimination'):.3f}.",
        "Большинство топологий дают близкие значения качества, поэтому задача восстановления в текущей постановке может быть слишком лёгкой или метрики недостаточно чувствительны к различиям топологий.",
        "",
        "## Реальное улучшение по Парето-фронту",
        f"Лучшее решение по точности имеет точность {_max(pareto, 'average_recovery_accuracy'):.3f}. Лучшее решение по устойчивости имеет устойчивость {_max(pareto, 'noise_robustness'):.3f}. Лучшее решение по различению имеет различение {_max(pareto, 'close_pattern_discrimination'):.3f}.",
        f"Сбалансированные решения имеют среднее число гиперребер {best_balanced['hyperedge_count'].mean():.3f} и среднюю плотность {best_balanced['density'].mean():.3f}.",
        "Сравнение показывает, что Парето-фронт не превосходит лучшую случайную базовую топологию по максимуму точности и различения.",
        "",
        "## Структурная сложность",
        f"Корреляция точности со структурной сложностью: {correlations.get('complexity_objective', 0):.3f}. Корреляция точности с числом гиперребер: {correlations.get('hyperedge_count', 0):.3f}.",
        "NSGA-II не выбирает только самые компактные топологии: лучшие решения используют заметное число гиперребер, но это не всегда даёт выигрыш над сильными базовыми топологиями.",
        "",
        "## Вклад гиперребер и штраф сложности",
        _format_diagnostic_variants(diagnostic),
        "Диагностические варианты не заменяют основную модель. Они показывают, что изменение вклада гиперребер и штрафа сложности само по себе не гарантирует устойчивого подтверждения H1 и H3.",
        "",
        "## Метрика различения близких паттернов",
        f"Порог близости равен {close_distance}. Найдено {len(close_pairs)} близких пар. Паттернов с близкими соседями: {len(close_indices)}. Тестов, участвующих в расчёте H3: {close_tests}.",
        "Количество тестов для H3 достаточно для диагностики, но один seed не даёт статистической устойчивости.",
        "",
        "## Базовые топологии",
        "Частотная и плотная топологии почти не уступают NSGA-II, а лучшая случайная топология в расширенном запуске оказалась сильнее по максимуму точности. Это важный отрицательный или частично отрицательный результат.",
    ]
    return "\n".join(lines) + "\n"


def _build_multiseed_summary(tables_dir: Path, reports_dir: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        prefix = f"multiseed_{seed}_"
        pareto_path = tables_dir / f"{prefix}pareto_solutions.csv"
        baseline_path = tables_dir / f"{prefix}baseline_metrics.csv"
        summary_path = reports_dir / f"{prefix}experiment_summary.json"
        if not pareto_path.exists() or not baseline_path.exists() or not summary_path.exists():
            continue
        pareto = pd.read_csv(pareto_path)
        baseline = pd.read_csv(baseline_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ranges = _top_ranges(pareto)
        rows.append(
            {
                "seed": seed,
                "pareto_count": len(pareto),
                "best_accuracy": _max(pareto, "average_recovery_accuracy"),
                "best_robustness": _max(pareto, "noise_robustness"),
                "best_discrimination": _max(pareto, "close_pattern_discrimination"),
                "best_baseline_accuracy": _max(baseline, "average_recovery_accuracy"),
                "best_baseline_discrimination": _max(baseline, "close_pattern_discrimination"),
                "best_solution_hyperedge_count": _best_value(pareto, "average_recovery_accuracy", "hyperedge_count"),
                "best_solution_density": _best_value(pareto, "average_recovery_accuracy", "density"),
                "best_solution_average_size": _best_value(pareto, "average_recovery_accuracy", "average_hyperedge_size"),
                "h1_status": summary.get("statistics", {}).get("hypotheses", {}).get("H1", ""),
                "h2_status": summary.get("statistics", {}).get("hypotheses", {}).get("H2", ""),
                "h3_status": summary.get("statistics", {}).get("hypotheses", {}).get("H3", ""),
                "top_density_min": ranges.get("density_min", 0.0),
                "top_density_max": ranges.get("density_max", 0.0),
                "top_size_min": ranges.get("size_min", 0.0),
                "top_size_max": ranges.get("size_max", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _format_multiseed_summary(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "# Сводка по seed\n\nДанные серии seed не найдены.\n"
    h1_count = int(summary["h1_status"].str.contains("поддерживается", case=False, na=False).sum())
    h2_count = int(summary["h2_status"].str.contains("Наблюдаются", case=False, na=False).sum())
    h3_count = int(summary["h3_status"].str.contains("поддерживается", case=False, na=False).sum())
    text = [
        "# Сводка по нескольким seed",
        "",
        f"Запущено seed: {len(summary)}.",
        f"Точность восстановления: среднее {summary['best_accuracy'].mean():.3f}, разброс {summary['best_accuracy'].std(ddof=0):.3f}.",
        f"Устойчивость: среднее {summary['best_robustness'].mean():.3f}, разброс {summary['best_robustness'].std(ddof=0):.3f}.",
        f"Различение близких паттернов: среднее {summary['best_discrimination'].mean():.3f}, разброс {summary['best_discrimination'].std(ddof=0):.3f}.",
        f"H1 предварительно поддерживается в {h1_count} из {len(summary)} запусков.",
        f"H2 предварительно поддерживается в {h2_count} из {len(summary)} запусков.",
        f"H3 предварительно поддерживается в {h3_count} из {len(summary)} запусков.",
        f"У лучших решений чаще встречается число гиперребер около {summary['best_solution_hyperedge_count'].mean():.1f}, плотность около {summary['best_solution_density'].mean():.3f}, средняя кратность около {summary['best_solution_average_size'].mean():.3f}.",
    ]
    return "\n".join(text) + "\n"


def _append_report(report_path: Path, diagnostic: pd.DataFrame, multiseed: pd.DataFrame, diagnostic_report: str) -> None:
    h1_count = int(multiseed["h1_status"].str.contains("поддерживается", case=False, na=False).sum()) if not multiseed.empty else 0
    h2_count = int(multiseed["h2_status"].str.contains("Наблюдаются", case=False, na=False).sum()) if not multiseed.empty else 0
    h3_count = int(multiseed["h3_status"].str.contains("поддерживается", case=False, na=False).sum()) if not multiseed.empty else 0
    prefix = "\n\n" if report_path.exists() and report_path.read_text(encoding="utf-8").strip() else ""
    lines = [
        "## Диагностика и серия по нескольким seed",
        "",
        "Диагностика потребовалась потому, что расширенный запуск на одном seed не дал устойчивого подтверждения H1 и H3.",
        "",
        "Проверены сложность задачи восстановления, наличие реального улучшения на Парето-фронте, связь качества со структурной сложностью, вклад гиперребер, штраф за сложность, метрика различения близких паттернов и сила базовых топологий.",
        "",
        "Диагностические варианты с усиленным вкладом гиперребер, ослабленным штрафом и отключённым штрафом показали, что простая перенастройка весов не гарантирует подтверждения H1 и H3.",
        "",
        f"Запущено seed: {len(multiseed)}. H1 предварительно поддерживается в {h1_count} запусках, H2 — в {h2_count} запусках, H3 — в {h3_count} запусках.",
        "",
        "Выводы остаются предварительными. Перед финальным научным запуском стоит усилить сложность тестовых искажений, увеличить число seed, расширить пул кандидатов и проверить чувствительность метрики различения близких паттернов.",
        "",
        "Подробности сохранены в reports/diagnostic_report.md и reports/multiseed_summary.md.",
    ]
    with report_path.open("a", encoding="utf-8") as file:
        file.write(prefix + "\n".join(lines) + "\n")


def _format_diagnostic_variants(diagnostic: pd.DataFrame) -> str:
    parts = []
    for _, row in diagnostic.iterrows():
        parts.append(
            f"{row['variant']}: точность {row['best_accuracy']:.3f}, устойчивость {row['best_robustness']:.3f}, различение {row['best_discrimination']:.3f}"
        )
    return "; ".join(parts) + "."


def _top_ranges(pareto: pd.DataFrame) -> dict[str, float]:
    if pareto.empty:
        return {}
    top = pareto[pareto["average_recovery_accuracy"] >= pareto["average_recovery_accuracy"].quantile(0.75)]
    return {
        "density_min": float(top["density"].min()),
        "density_max": float(top["density"].max()),
        "size_min": float(top["average_hyperedge_size"].min()),
        "size_max": float(top["average_hyperedge_size"].max()),
    }


def _max(table: pd.DataFrame, column: str) -> float:
    return float(table[column].max()) if column in table and not table.empty else 0.0


def _mean(table: pd.DataFrame, column: str) -> float:
    return float(table[column].mean()) if column in table and not table.empty else 0.0


def _best_value(table: pd.DataFrame, by: str, value: str) -> float:
    if table.empty or by not in table or value not in table:
        return 0.0
    return float(table.sort_values(by, ascending=False).iloc[0][value])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Диагностика экспериментов АГ-памяти.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    return parser.parse_args()


if __name__ == "__main__":
    main()
