"""Статистический анализ результатов экспериментов."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


METRICS = [
    "average_recovery_accuracy",
    "noise_robustness",
    "close_pattern_discrimination",
]


def compare_nsga_vs_random(nsga_df: pd.DataFrame, random_df: pd.DataFrame) -> pd.DataFrame:
    """Сравнивает NSGA-II со случайными топологиями."""

    return _compare_groups(nsga_df, random_df, "random")


def compare_nsga_vs_baselines(nsga_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Сравнивает NSGA-II со всеми доступными базовыми топологиями."""

    rows = []
    for baseline_name, group in baseline_df.groupby("baseline", dropna=False):
        rows.append(_compare_groups(nsga_df, group, str(baseline_name)))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def analyze_structural_ranges(pareto_df: pd.DataFrame) -> pd.DataFrame:
    """Ищет диапазоны структурных параметров для лучших решений."""

    if pareto_df.empty or "average_recovery_accuracy" not in pareto_df:
        return pd.DataFrame()
    threshold = pareto_df["average_recovery_accuracy"].quantile(0.75)
    best = pareto_df[pareto_df["average_recovery_accuracy"] >= threshold]
    rows = []
    for metric in ["average_hyperedge_size", "density", "hyperedge_count"]:
        if metric not in best:
            continue
        rows.append(
            {
                "metric": metric,
                "min": float(best[metric].min()),
                "max": float(best[metric].max()),
                "mean": float(best[metric].mean()),
                "solution_count": int(len(best)),
            }
        )
    return pd.DataFrame(rows)


def test_hypotheses(nsga_df: pd.DataFrame, baseline_df: pd.DataFrame) -> dict[str, str]:
    """Формирует осторожные выводы по гипотезам H1-H3."""

    random_df = baseline_df[baseline_df["baseline"] == "random"] if "baseline" in baseline_df else pd.DataFrame()
    h1 = _h1_summary(nsga_df, random_df, baseline_df)
    h2 = _h2_summary(analyze_structural_ranges(nsga_df), nsga_df)
    h3 = _h3_summary(nsga_df, baseline_df)
    return {"H1": h1, "H2": h2, "H3": h3}


def run_statistical_analysis(
    pareto_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    reports_dir: str | Path,
) -> dict[str, Any]:
    """Сохраняет таблицы статистики и текстовую сводку."""

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    random_df = baseline_df[baseline_df["baseline"] == "random"] if "baseline" in baseline_df else pd.DataFrame()
    random_tests = compare_nsga_vs_random(pareto_df, random_df)
    all_tests = compare_nsga_vs_baselines(pareto_df, baseline_df)
    ranges = analyze_structural_ranges(pareto_df)
    hypotheses = test_hypotheses(pareto_df, baseline_df)

    tests = all_tests if not all_tests.empty else random_tests
    tests.to_csv(reports_path / "hypothesis_tests.csv", index=False)
    ranges.to_csv(reports_path / "hypothesis_ranges.csv", index=False)
    _write_hypothesis_summary(reports_path / "hypothesis_summary.txt", hypotheses)

    return {
        "hypothesis_tests": "reports/hypothesis_tests.csv",
        "hypothesis_ranges": "reports/hypothesis_ranges.csv",
        "hypothesis_summary": "reports/hypothesis_summary.txt",
        "hypotheses": hypotheses,
        "test_count": int(len(tests)),
        "range_count": int(len(ranges)),
    }


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    """Считает размер эффекта Cliff's Delta."""

    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.size == 0 or right_values.size == 0:
        return 0.0
    greater = sum(1 for x in left_values for y in right_values if x > y)
    lower = sum(1 for x in left_values for y in right_values if x < y)
    return float((greater - lower) / (left_values.size * right_values.size))


def _compare_groups(nsga_df: pd.DataFrame, baseline_df: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        if metric not in nsga_df or metric not in baseline_df:
            continue
        nsga_values = nsga_df[metric].dropna().to_numpy(dtype=float)
        baseline_values = baseline_df[metric].dropna().to_numpy(dtype=float)
        if len(nsga_values) == 0 or len(baseline_values) == 0:
            continue
        p_value = mannwhitneyu(nsga_values, baseline_values, alternative="two-sided").pvalue
        rows.append(
            {
                "baseline": baseline_name,
                "metric": metric,
                "nsga_mean": float(np.mean(nsga_values)),
                "baseline_mean": float(np.mean(baseline_values)),
                "p_value": float(p_value),
                "cliffs_delta": cliffs_delta(nsga_values, baseline_values),
                "conclusion": _metric_conclusion(nsga_values, baseline_values, p_value),
                "nsga_count": int(len(nsga_values)),
                "baseline_count": int(len(baseline_values)),
            }
        )
    return pd.DataFrame(rows)


def _metric_conclusion(nsga_values: np.ndarray, baseline_values: np.ndarray, p_value: float) -> str:
    if p_value < 0.05 and np.mean(nsga_values) > np.mean(baseline_values):
        return "NSGA-II выше, статистически значимо"
    if p_value < 0.05 and np.mean(nsga_values) < np.mean(baseline_values):
        return "Базовая топология выше, статистически значимо"
    return "Статистически значимого превосходства не выявлено"


def _h1_summary(nsga_df: pd.DataFrame, random_df: pd.DataFrame, baseline_df: pd.DataFrame) -> str:
    if nsga_df.empty or baseline_df.empty:
        return "Недостаточно данных."
    nsga_accuracy = nsga_df["average_recovery_accuracy"].mean()
    nsga_robustness = nsga_df["noise_robustness"].mean()
    baseline_accuracy = baseline_df["average_recovery_accuracy"].max()
    baseline_robustness = baseline_df["noise_robustness"].max()
    if nsga_accuracy > baseline_accuracy and nsga_robustness > baseline_robustness:
        return "Предварительно поддерживается, но требует подтверждения на большем числе запусков."
    if not random_df.empty and nsga_accuracy > random_df["average_recovery_accuracy"].mean():
        return "Частично поддерживается относительно случайных топологий, вывод предварительный."
    return "На имеющихся данных не подтверждается; вывод предварительный."


def _h2_summary(ranges_df: pd.DataFrame, pareto_df: pd.DataFrame) -> str:
    if ranges_df.empty or pareto_df.empty:
        return "Недостаточно данных."
    return "Наблюдаются диапазоны структурных параметров для верхней четверти решений; вывод предварительный."


def _h3_summary(nsga_df: pd.DataFrame, baseline_df: pd.DataFrame) -> str:
    if nsga_df.empty or baseline_df.empty:
        return "Недостаточно данных."
    if nsga_df["close_pattern_discrimination"].mean() > baseline_df["close_pattern_discrimination"].max():
        return "Предварительно поддерживается, но требует большего числа запусков."
    return "На имеющихся данных не подтверждается; вывод предварительный."


def _write_hypothesis_summary(path: Path, hypotheses: dict[str, str]) -> None:
    text = "\n".join(f"{name}: {summary}" for name, summary in hypotheses.items())
    path.write_text(text + "\n", encoding="utf-8")
