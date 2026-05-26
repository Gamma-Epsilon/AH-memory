"""Множество абстрактных символов АГ-памяти."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SymbolSet:
    """Хранит размерность пространства бинарных признаков."""

    dimension: int

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("Размерность множества символов должна быть положительной.")

    def validate_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Проверяет, что паттерн согласован с пространством символов."""

        checked = np.asarray(pattern)
        if checked.shape != (self.dimension,):
            raise ValueError("Паттерн должен иметь длину, равную размерности множества символов.")
        return checked

    def random_pattern(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Возвращает один случайный бинарный паттерн для первичной проверки."""

        generator = rng or np.random.default_rng()
        return generator.integers(0, 2, size=self.dimension, dtype=int)
