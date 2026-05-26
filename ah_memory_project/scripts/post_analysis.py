"""Построение графиков и статистический анализ сохраненного эксперимента."""

from __future__ import annotations

import json
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
    results_dir = ROOT_DIR / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    reports_dir = results_dir / "reports"

    pareto_df = pd.read_csv(tables_dir / "pareto_solutions.csv")
    baseline_df = pd.read_csv(tables_dir / "baseline_metrics.csv")
    candidate_df = pd.read_csv(tables_dir / "candidate_pool.csv")
    summary_path = reports_dir / "experiment_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    figures = create_all_figures(pareto_df, baseline_df, candidate_df, summary, figures_dir)
    statistics = run_statistical_analysis(pareto_df, baseline_df, reports_dir)
    summary["figures"] = _relative_paths(figures, results_dir)
    summary["statistics"] = statistics
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_report(results_dir / "report.md", summary, figures, statistics, pareto_df, baseline_df)


def _append_report(
    report_path: Path,
    summary: dict[str, Any],
    figures: dict[str, str],
    statistics: dict[str, Any],
    pareto_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> None:
    prefix = "\n\n" if report_path.exists() and report_path.read_text(encoding="utf-8").strip() else ""
    baseline_names = ", ".join(sorted(str(name) for name in baseline_df["baseline"].unique()))
    hypotheses = statistics["hypotheses"]
    figure_lines = [f"- {path}" for path in _relative_paths(figures, report_path.parent).values()]
    if not figure_lines:
        figure_lines = ["- Данных для построения графиков недостаточно."]

    lines = [
        "## Визуализация и статистический анализ",
        "",
        "Эксперимент выполнен с умеренными параметрами, поэтому статистические выводы следует считать предварительными.",
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


if __name__ == "__main__":
    main()
