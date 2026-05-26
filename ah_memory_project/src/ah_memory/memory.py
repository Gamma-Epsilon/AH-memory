"""Операции записи, извлечения и истории АГ-памяти."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ah_memory.config import RetrievalConfig
from ah_memory.focus import ActivationFocus
from ah_memory.history import Episode, EpisodeMemory
from ah_memory.knowledge import KnowledgeHypergraph
from ah_memory.symbols import SymbolSet


@dataclass(frozen=True)
class RetrievalResult:
    """Хранит результат восстановления одного тестового входа."""

    original_index: int | None
    restored_index: int
    restored_pattern: np.ndarray
    accuracy: float | None
    exact_match: bool | None
    score: float


class AHMemory:
    """Объединяет символы, знания, историю и фокус активации."""

    def __init__(
        self,
        symbol_set: SymbolSet,
        knowledge: KnowledgeHypergraph | None = None,
        history: EpisodeMemory | None = None,
        focus: ActivationFocus | None = None,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.symbol_set = symbol_set
        self.knowledge = knowledge or KnowledgeHypergraph(symbol_set.dimension)
        self.history = history or EpisodeMemory()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.focus = focus or ActivationFocus(self.retrieval_config)
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

    def score_pattern(self, query: np.ndarray, candidate: np.ndarray) -> float:
        """Считает оценку сохраненного паттерна для искаженного входа."""

        prepared_query = self.symbol_set.validate_pattern(query)
        prepared_candidate = self.symbol_set.validate_pattern(candidate)
        return self.focus.score(prepared_query, prepared_candidate, self.knowledge)

    def retrieve(self, query: np.ndarray, return_score: bool = False) -> tuple[int, np.ndarray] | tuple[int, np.ndarray, float]:
        """Восстанавливает наиболее подходящий сохраненный паттерн."""

        prepared_query = self.symbol_set.validate_pattern(query)
        best_index, score = self._best_match(prepared_query)
        restored = self.patterns[best_index].copy()
        self.last_score = score
        self.add_episode(restored, prepared_query, restored, score)
        if return_score:
            return best_index, restored, score
        return best_index, restored

    def retrieve_batch(self, tests: list[Any] | np.ndarray) -> list[RetrievalResult]:
        """Восстанавливает набор тестов и считает качество каждого ответа."""

        return [self._retrieve_single_test(test) for test in tests]

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

    def _best_match(self, query: np.ndarray) -> tuple[int, float]:
        if len(self.patterns) == 0:
            raise ValueError("Память не содержит сохраненных паттернов.")

        scores = [self.score_pattern(query, candidate) for candidate in self.patterns]
        best_index = int(np.argmax(scores))
        return best_index, float(scores[best_index])

    def _retrieve_single_test(self, test: Any) -> RetrievalResult:
        original_index = getattr(test, "pattern_index", None)
        original = getattr(test, "original", None)
        distorted = getattr(test, "distorted", test)

        restored_index, restored, score = self.retrieve(distorted, return_score=True)
        accuracy = None
        exact_match = None

        if original is not None:
            prepared_original = self.symbol_set.validate_pattern(original)
            accuracy = float(np.mean(restored == prepared_original))
            exact_match = bool(np.array_equal(restored, prepared_original))

        return RetrievalResult(
            original_index=original_index,
            restored_index=restored_index,
            restored_pattern=restored,
            accuracy=accuracy,
            exact_match=exact_match,
            score=score,
        )
