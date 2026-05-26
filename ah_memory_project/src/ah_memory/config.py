"""Конфигурация экспериментов АГ-памяти."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PatternGenerationConfig:
    """Хранит параметры генерации обучающих паттернов и тестовых входов."""

    pattern_count: int = 60
    dimension: int = 32
    active_fraction: float = 0.25
    cluster_count: int = 3
    intra_cluster_flips: int = 4
    noise_levels: tuple[float, ...] = field(default_factory=lambda: (0.0, 0.05, 0.10, 0.20))
    missing_levels: tuple[float, ...] = field(default_factory=lambda: (0.0, 0.10, 0.20))
    trials_per_pattern: int = 2
    close_pair_distance: int = 6
    seed: int = 42

    def __post_init__(self) -> None:
        if self.pattern_count <= 0:
            raise ValueError("Количество паттернов должно быть положительным.")
        if self.dimension <= 0:
            raise ValueError("Размерность паттерна должна быть положительной.")
        if not 0 < self.active_fraction < 1:
            raise ValueError("Доля активных битов должна быть между нулем и единицей.")
        if self.cluster_count <= 0:
            raise ValueError("Количество кластеров должно быть положительным.")
        if self.intra_cluster_flips < 0:
            raise ValueError("Степень сходства внутри кластера не может быть отрицательной.")
        if self.trials_per_pattern <= 0:
            raise ValueError("Количество испытаний должно быть положительным.")
        if self.close_pair_distance < 0:
            raise ValueError("Порог близости не может быть отрицательным.")
        self._validate_levels(self.noise_levels, "Уровни шума")
        self._validate_levels(self.missing_levels, "Уровни пропусков")

    @staticmethod
    def _validate_levels(levels: tuple[float, ...], label: str) -> None:
        if not levels:
            raise ValueError(f"{label} должны содержать хотя бы одно значение.")
        if any(level < 0 or level > 1 for level in levels):
            raise ValueError(f"{label} должны быть в диапазоне от нуля до единицы.")


@dataclass(frozen=True)
class RetrievalConfig:
    """Хранит коэффициенты для восстановления паттернов из памяти."""

    bit_match_weight: float = 1.0
    hyperedge_weight: float = 1.0
    complexity_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.bit_match_weight < 0:
            raise ValueError("Коэффициент совпадения битов не может быть отрицательным.")
        if self.hyperedge_weight < 0:
            raise ValueError("Коэффициент активности гиперребер не может быть отрицательным.")
        if self.complexity_weight < 0:
            raise ValueError("Коэффициент штрафа за сложность не может быть отрицательным.")


@dataclass(frozen=True)
class OptimizationConfig:
    """Хранит параметры эволюционной оптимизации топологии."""

    population_size: int = 80
    n_generations: int = 50
    crossover_probability: float = 0.9
    mutation_probability: float | None = None
    eliminate_duplicates: bool = True
    min_selected_hyperedges: int = 1
    max_selected_hyperedges: int | None = None
    parallel_workers: int = 1
    complexity_edge_count_weight: float = 0.35
    complexity_density_weight: float = 0.25
    complexity_hyperedge_size_weight: float = 0.25
    complexity_degree_variance_weight: float = 0.15
    seed: int = 42

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("Размер популяции должен быть положительным.")
        if self.n_generations <= 0:
            raise ValueError("Число поколений должно быть положительным.")
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("Вероятность кроссовера должна быть в диапазоне от нуля до единицы.")
        if self.mutation_probability is not None and not 0 <= self.mutation_probability <= 1:
            raise ValueError("Вероятность мутации должна быть в диапазоне от нуля до единицы.")
        if self.min_selected_hyperedges < 0:
            raise ValueError("Минимальное число гиперребер не может быть отрицательным.")
        if self.max_selected_hyperedges is not None and self.max_selected_hyperedges < self.min_selected_hyperedges:
            raise ValueError("Максимальное число гиперребер не может быть меньше минимального.")
        if self.parallel_workers <= 0:
            raise ValueError("Число параллельных работников должно быть положительным.")
        complexity_weight_sum = (
            self.complexity_edge_count_weight
            + self.complexity_density_weight
            + self.complexity_hyperedge_size_weight
            + self.complexity_degree_variance_weight
        )
        if any(
            weight < 0
            for weight in (
                self.complexity_edge_count_weight,
                self.complexity_density_weight,
                self.complexity_hyperedge_size_weight,
                self.complexity_degree_variance_weight,
            )
        ):
            raise ValueError("Веса структурной сложности не могут быть отрицательными.")
        if abs(complexity_weight_sum - 1.0) > 1e-9:
            raise ValueError("Веса структурной сложности должны суммарно давать единицу.")
