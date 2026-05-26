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
