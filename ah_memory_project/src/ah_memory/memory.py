"""Операции записи, извлечения и истории АГ-памяти."""

from __future__ import annotations

import numpy as np

from ah_memory.focus import ActivationFocus
from ah_memory.history import Episode, EpisodeMemory
from ah_memory.knowledge import KnowledgeHypergraph
from ah_memory.symbols import SymbolSet


class AHMemory:
    """Объединяет символы, знания, историю и фокус активации."""

    def __init__(
        self,
        symbol_set: SymbolSet,
        knowledge: KnowledgeHypergraph | None = None,
        history: EpisodeMemory | None = None,
        focus: ActivationFocus | None = None,
    ) -> None:
        self.symbol_set = symbol_set
        self.knowledge = knowledge or KnowledgeHypergraph(symbol_set.dimension)
        self.history = history or EpisodeMemory()
        self.focus = focus or ActivationFocus()
        self.patterns = np.empty((0, symbol_set.dimension), dtype=int)
        self.last_score = 0.0

    def fit(self, patterns: np.ndarray) -> "AHMemory":
        """Сохраняет обучающие паттерны и пересчитывает веса ассоциаций."""

        prepared = self._prepare_patterns(patterns)
        self.patterns = prepared.copy()
        self.knowledge.update_weights(self.patterns)
        return self

    def record(self, pattern: np.ndarray) -> None:
        """Добавляет один паттерн в память и обновляет веса гиперребер."""

        prepared = self.symbol_set.validate_pattern(pattern).astype(int)
        self.patterns = np.vstack([self.patterns, prepared])
        self.knowledge.update_weights(self.patterns)

    def retrieve(self, query: np.ndarray) -> tuple[int, np.ndarray]:
        """Восстанавливает наиболее подходящий сохраненный паттерн."""

        prepared_query = self.symbol_set.validate_pattern(query)
        best_index, score = self.focus.best_match(prepared_query, self.patterns, self.knowledge)
        restored = self.patterns[best_index].copy()
        self.last_score = score
        self.add_episode(restored, prepared_query, restored, score)
        return best_index, restored

    def retrieve_batch(self, tests: list[np.ndarray] | np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Последовательно восстанавливает набор искаженных входов."""

        return [self.retrieve(test) for test in tests]

    def add_episode(
        self,
        original: np.ndarray,
        distorted: np.ndarray,
        result: np.ndarray,
        score: float,
    ) -> Episode:
        """Записывает эпизод восстановления в историческую сеть."""

        return self.history.add_episode(original, distorted, result, score)

    def _prepare_patterns(self, patterns: np.ndarray) -> np.ndarray:
        prepared = np.asarray(patterns)
        if prepared.ndim != 2 or prepared.shape[1] != self.symbol_set.dimension:
            raise ValueError("Набор паттернов должен быть двумерным и согласованным с множеством символов.")
        return prepared.astype(int)
