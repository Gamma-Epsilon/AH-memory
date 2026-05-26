"""Единый слой знаний и ассоциативные гиперсвязи."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Hyperedge:
    """Описывает ассоциацию между несколькими символами."""

    symbols: tuple[int, ...]
    weight: float = 0.0

    def __post_init__(self) -> None:
        unique_symbols = tuple(sorted(set(self.symbols)))
        if not unique_symbols:
            raise ValueError("Гиперребро должно связывать хотя бы один символ.")
        self.symbols = unique_symbols

    @property
    def size(self) -> int:
        """Возвращает кратность гиперребра."""

        return len(self.symbols)


class KnowledgeHypergraph:
    """Хранит ассоциативные гиперребра единого слоя знаний."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("Размерность слоя знаний должна быть положительной.")
        self.dimension = dimension
        self.hyperedges: list[Hyperedge] = []

    def add_hyperedge(self, symbols: tuple[int, ...] | list[int], weight: float = 0.0) -> Hyperedge:
        """Добавляет гиперребро после проверки индексов символов."""

        edge = Hyperedge(tuple(symbols), float(weight))
        if edge.symbols[-1] >= self.dimension or edge.symbols[0] < 0:
            raise ValueError("Гиперребро содержит индекс вне пространства символов.")
        self.hyperedges.append(edge)
        return edge

    def update_weights(self, patterns: np.ndarray) -> None:
        """Обновляет веса по частоте совместной активации символов."""

        prepared = np.asarray(patterns)
        if prepared.ndim != 2 or prepared.shape[1] != self.dimension:
            raise ValueError("Обучающие паттерны должны образовывать двумерную матрицу нужной ширины.")
        if prepared.shape[0] == 0:
            for edge in self.hyperedges:
                edge.weight = 0.0
            return

        binary = prepared == 1
        for edge in self.hyperedges:
            active_rows = np.all(binary[:, edge.symbols], axis=1)
            edge.weight = float(np.mean(active_rows))

    def edge_sizes(self) -> list[int]:
        """Возвращает размеры всех гиперребер."""

        return [edge.size for edge in self.hyperedges]

    def as_tuples(self) -> list[tuple[tuple[int, ...], float]]:
        """Возвращает простое представление гиперребер и их весов."""

        return [(edge.symbols, edge.weight) for edge in self.hyperedges]

    def complexity(self) -> float:
        """Оценивает структурную сложность слоя знаний простым средним размером."""

        if not self.hyperedges:
            return 0.0
        return float(sum(self.edge_sizes()) / self.dimension)
