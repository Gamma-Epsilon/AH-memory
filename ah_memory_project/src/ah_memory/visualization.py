"""Визуализация результатов экспериментов."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_accuracy_complexity(pareto_df: pd.DataFrame, output_path: str | Path) -> str | None:
    """Строит график точности восстановления и структурной сложности."""

    if not _has_columns(pareto_df, ["complexity_objective", "average_recovery_accuracy"]):
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pareto_df["complexity_objective"], pareto_df["average_recovery_accuracy"], color="#2f6f9f")
    ax.set_xlabel("Структурная сложность")
    ax.set_ylabel("Точность восстановления")
    ax.set_title("Парето-фронт: точность и сложность")
    ax.grid(alpha=0.25)
    return _save_figure(fig, output_path)


def plot_robustness_discrimination(pareto_df: pd.DataFrame, output_path: str | Path) -> str | None:
    """Строит график устойчивости и различения близких паттернов."""

    if not _has_columns(pareto_df, ["noise_robustness", "close_pattern_discrimination"]):
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pareto_df["noise_robustness"], pareto_df["close_pattern_discrimination"], color="#3f7d44")
    ax.set_xlabel("Устойчивость к шуму")
    ax.set_ylabel("Различение близких паттернов")
    ax.set_title("Парето-фронт: устойчивость и различение")
    ax.grid(alpha=0.25)
    return _save_figure(fig, output_path)


def plot_baseline_comparison(
    pareto_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    output_path: str | Path,
) -> str | None:
    """Сравнивает NSGA-II и базовые топологии по ключевым метрикам."""

    metric_names = [
        "average_recovery_accuracy",
        "noise_robustness",
        "close_pattern_discrimination",
    ]
    if not _has_columns(pareto_df, metric_names) or not _has_columns(baseline_df, ["baseline", *metric_names]):
        return None

    groups = [("NSGA-II", pareto_df)]
    groups.extend((str(name), group) for name, group in baseline_df.groupby("baseline", dropna=False))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    labels = [name for name, _ in groups]
    for axis, metric, title in zip(
        axes,
        metric_names,
        ["Точность восстановления", "Устойчивость", "Различение"],
    ):
        values = [group[metric].dropna().to_numpy() for _, group in groups]
        axis.boxplot(values, labels=labels, showmeans=True)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Значение метрики")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_degree_distribution_for_genome(
    genome: str,
    candidate_df: pd.DataFrame,
    n_vertices: int,
    output_path: str | Path,
) -> str | None:
    """Строит распределение степеней вершин для выбранной топологии."""

    selected_edges = selected_edges_from_genome(genome, candidate_df)
    if not selected_edges:
        return None
    degrees = np.zeros(n_vertices, dtype=int)
    for edge in selected_edges:
        for symbol in edge:
            if 0 <= symbol < n_vertices:
                degrees[symbol] += 1

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(degrees, bins=range(0, int(degrees.max()) + 2), color="#7a4e9f", edgecolor="white")
    ax.set_xlabel("Степень вершины")
    ax.set_ylabel("Количество вершин")
    ax.set_title("Распределение степеней лучшей топологии")
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, output_path)


def plot_hyperedge_size_histogram_for_genome(
    genome: str,
    candidate_df: pd.DataFrame,
    output_path: str | Path,
) -> str | None:
    """Строит распределение размеров гиперребер для выбранной топологии."""

    selected_edges = selected_edges_from_genome(genome, candidate_df)
    if not selected_edges:
        return None
    sizes = [len(edge) for edge in selected_edges]
    values, counts = np.unique(sizes, return_counts=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(values, counts, color="#b66a3c")
    ax.set_xlabel("Размер гиперребра")
    ax.set_ylabel("Количество гиперребер")
    ax.set_title("Размеры гиперребер лучшей топологии")
    ax.set_xticks(values)
    ax.grid(axis="y", alpha=0.25)
    return _save_figure(fig, output_path)


def plot_noise_curves(
    pareto_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    output_path: str | Path,
) -> str | None:
    """Строит кривые зависимости точности от уровня шума."""

    if "accuracy_by_noise" not in pareto_df or "accuracy_by_noise" not in baseline_df:
        return None
    best_row = pareto_df.sort_values("average_recovery_accuracy", ascending=False).head(1)
    if best_row.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_noise_dict(ax, "NSGA-II", best_row.iloc[0]["accuracy_by_noise"], "#2f6f9f")
    palette = ["#3f7d44", "#b66a3c", "#7a4e9f", "#8c6d31", "#5f7f8f"]
    for color, (name, group) in zip(palette, baseline_df.groupby("baseline", dropna=False)):
        best_baseline = group.sort_values("average_recovery_accuracy", ascending=False).iloc[0]
        _plot_noise_dict(ax, str(name), best_baseline["accuracy_by_noise"], color)

    ax.set_xlabel("Уровень шума")
    ax.set_ylabel("Точность восстановления")
    ax.set_title("Точность при разных уровнях шума")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save_figure(fig, output_path)


def create_all_figures(
    pareto_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    summary: dict[str, Any],
    figures_dir: str | Path,
) -> dict[str, str]:
    """Создает все доступные графики и возвращает пути к ним."""

    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    created: dict[str, str] = {}

    figure_specs = {
        "pareto_accuracy_complexity": plot_accuracy_complexity(
            pareto_df, figures_path / "pareto_accuracy_complexity.png"
        ),
        "pareto_robustness_discrimination": plot_robustness_discrimination(
            pareto_df, figures_path / "pareto_robustness_discrimination.png"
        ),
        "baseline_comparison_boxplot": plot_baseline_comparison(
            pareto_df, baseline_df, figures_path / "baseline_comparison_boxplot.png"
        ),
        "noise_curve_best_vs_baselines": plot_noise_curves(
            pareto_df, baseline_df, figures_path / "noise_curve_best_vs_baselines.png"
        ),
    }

    if not pareto_df.empty and "genome" in pareto_df:
        best = pareto_df.sort_values("average_recovery_accuracy", ascending=False).iloc[0]
        n_vertices = int(summary.get("generation", {}).get("n_vertices", 0))
        figure_specs["degree_distribution_best_accuracy"] = plot_degree_distribution_for_genome(
            str(best["genome"]),
            candidate_df,
            n_vertices,
            figures_path / "degree_distribution_best_accuracy.png",
        )
        figure_specs["hyperedge_size_histogram_best_accuracy"] = plot_hyperedge_size_histogram_for_genome(
            str(best["genome"]),
            candidate_df,
            figures_path / "hyperedge_size_histogram_best_accuracy.png",
        )

    for name, path in figure_specs.items():
        if path is not None:
            created[name] = path
    return created


def selected_edges_from_genome(genome: str, candidate_df: pd.DataFrame) -> list[tuple[int, ...]]:
    """Восстанавливает выбранные гиперребра по строке генома."""

    edges = []
    for bit, symbols in zip(str(genome), candidate_df["symbols"]):
        if bit != "1":
            continue
        edges.append(tuple(int(part) for part in str(symbols).split()))
    return edges


def parse_metric_dict(value: Any) -> dict[float, float]:
    """Преобразует сохраненный словарь метрик в числовой словарь."""

    if isinstance(value, dict):
        raw = value
    else:
        raw = ast.literal_eval(str(value))
    return {float(key): float(item) for key, item in raw.items()}


def _plot_noise_dict(ax: Any, label: str, value: Any, color: str) -> None:
    data = parse_metric_dict(value)
    xs = sorted(data)
    ys = [data[x] for x in xs]
    ax.plot(xs, ys, marker="o", label=label, color=color)


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return not df.empty and all(column in df for column in columns)


def _save_figure(fig: Any, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.as_posix()
