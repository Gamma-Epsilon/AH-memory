"""Построение графиков и статистический анализ сохраненного эксперимента."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ah_memory.statistics import run_statistical_analysis
from ah_memory.visualization import create_all_figures


def main() -> None:
    args = _parse_args()
    prefix = args.file_prefix
    results_dir = _resolve_results_dir(args.results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    reports_dir = results_dir / "reports"

    pareto_df = pd.read_csv(tables_dir / f"{prefix}pareto_solutions.csv")
    baseline_df = pd.read_csv(tables_dir / f"{prefix}baseline_metrics.csv")
    candidate_df = pd.read_csv(tables_dir / f"{prefix}candidate_pool.csv")
    summary_path = reports_dir / f"{prefix}experiment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    figures = create_all_figures(pareto_df, baseline_df, candidate_df, summary, figures_dir, file_prefix=prefix)
    statistics = run_statistical_analysis(pareto_df, baseline_df, reports_dir, file_prefix=prefix)
    summary["figures"] = _relative_paths(figures, results_dir)
    summary["statistics"] = statistics
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_report(results_dir / "report.md", summary, figures, statistics, pareto_df, baseline_df, args.report_title, args.experiment_note)


def _append_report(
    report_path: Path,
    summary: dict[str, Any],
    figures: dict[str, str],
    statistics: dict[str, Any],
    pareto_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    title: str,
    experiment_note: str,
) -> None:
    prefix = "\n\n" if report_path.exists() and report_path.read_text(encoding="utf-8").strip() else ""
    baseline_names = ", ".join(sorted(str(name) for name in baseline_df["baseline"].unique()))
    hypotheses = statistics["hypotheses"]
    figure_lines = [f"- {path}" for path in _relative_paths(figures, report_path.parent).values()]
    if not figure_lines:
        figure_lines = ["- Данных для построения графиков недостаточно."]

    lines = [
        f"## {title}",
        "",
        experiment_note,
        "",
        "Сгенерированы данные с "
        f"{summary['generation']['n_vertices']} вершинами, {summary['generation']['pattern_count']} паттернами "
        f"и {summary['generation']['cluster_count']} кластерами. Порог близости для близких паттернов равен "
        f"{summary['close_pattern_distance_used']}. Межкластерный целевой порог равен "
        f"{summary['generation']['requested_center_distance']}, фактически использованный порог равен "
        f"{summary['generation']['used_center_distance']}.",
        "",
        "По сравнению с умеренным запуском увеличены число поколений, размер популяции, размер пула гиперребер "
        "и число случайных базовых топологий. Использованы параллельные вычисления.",
        "",
        f"На Парето-фронте получено {len(pareto_df)} решений. Лучшая точность восстановления равна "
        f"{pareto_df['average_recovery_accuracy'].max():.3f}, лучшая устойчивость равна "
        f"{pareto_df['noise_robustness'].max():.3f}, лучшее различение близких паттернов равно "
        f"{pareto_df['close_pattern_discrimination'].max():.3f}.",
        "",
        f"Среднее число гиперребер на Парето-фронте равно {pareto_df['hyperedge_count'].mean():.3f}, "
        f"средняя плотность равна {pareto_df['density'].mean():.3f}, средняя кратность гиперребра равна "
        f"{pareto_df['average_hyperedge_size'].mean():.3f}.",
        "",
        "Созданы графики:",
        *figure_lines,
        "",
        "График точности и сложности показывает расположение решений Парето-фронта: по горизонтали отражена структурная сложность, по вертикали — точность восстановления.",
        "График устойчивости и различения показывает соотношение устойчивости к шуму и способности различать близкие паттерны.",
        "Сравнение с базовыми топологиями показывает распределение ключевых метрик для NSGA-II и простых стратегий.",
        "Гистограммы лучшей топологии показывают распределение степеней вершин и размеров гиперребер.",
        "Кривая шума показывает изменение точности восстановления при разных уровнях искажений.",
        "",
        f"На Парето-фронте проанализировано {len(pareto_df)} решений. Сравнение выполнено с базовыми топологиями: {baseline_names}.",
        "",
        f"H1: {hypotheses['H1']}",
        f"H2: {hypotheses['H2']}",
        f"H3: {hypotheses['H3']}",
        "",
        "Дополнительно сохранены таблицы статистических сравнений и диапазонов структурных параметров лучших решений.",
    ]
    with report_path.open("a", encoding="utf-8") as file:
        file.write(prefix + "\n".join(lines) + "\n")


def _relative_paths(paths: dict[str, str], base_dir: Path) -> dict[str, str]:
    relative = {}
    for name, path in paths.items():
        path_obj = Path(path)
        try:
            relative[name] = path_obj.relative_to(base_dir).as_posix()
        except ValueError:
            try:
                relative[name] = path_obj.relative_to(ROOT_DIR / "results").as_posix()
            except ValueError:
                relative[name] = path_obj.as_posix()
    return relative


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Постанализ сохраненного эксперимента АГ-памяти.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--file-prefix", default="")
    parser.add_argument("--report-title", default="Визуализация и статистический анализ")
    parser.add_argument(
        "--experiment-note",
        default="Эксперимент выполнен с умеренными параметрами, поэтому статистические выводы следует считать предварительными.",
    )
    return parser.parse_args()


def _resolve_results_dir(results_dir: str) -> Path:
    path = Path(results_dir)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


if __name__ == "__main__":
    main()
