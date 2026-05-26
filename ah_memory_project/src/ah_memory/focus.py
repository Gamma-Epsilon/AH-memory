"""Фокус активации для оценки сохраненных паттернов."""

from __future__ import annotations

import numpy as np

from ah_memory.config import RetrievalConfig
from ah_memory.knowledge import KnowledgeHypergraph


class ActivationFocus:
    """Считает оценку соответствия запроса сохраненному паттерну."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        complexity_penalty: float | None = None,
    ) -> None:
        if complexity_penalty is not None:
            config = RetrievalConfig(complexity_weight=complexity_penalty)
        self.config = config or RetrievalConfig()

    def score(
        self,
        query: np.ndarray,
        candidate: np.ndarray,
        knowledge: KnowledgeHypergraph,
    ) -> float:
        """Объединяет совпадение битов, вклад гиперребер и штраф сложности."""

        prepared_query = np.asarray(query)
        prepared_candidate = np.asarray(candidate)
        known_mask = self._known_mask(prepared_query)
        if not np.any(known_mask):
            bit_score = 0.0
        else:
            bit_score = float(np.mean(prepared_query[known_mask] == prepared_candidate[known_mask]))

        edge_score = self.edge_activity(prepared_query, prepared_candidate, knowledge)
        complexity_penalty = len(knowledge.hyperedges)
        return (
            self.config.bit_match_weight * bit_score
            + self.config.hyperedge_weight * edge_score
            - self.config.complexity_weight * complexity_penalty
        )

    def best_match(
        self,
        query: np.ndarray,
        patterns: np.ndarray,
        knowledge: KnowledgeHypergraph,
    ) -> tuple[int, float]:
        """Выбирает сохраненный паттерн с максимальной оценкой."""

        if len(patterns) == 0:
            raise ValueError("Память не содержит сохраненных паттернов.")

        scores = [self.score(query, candidate, knowledge) for candidate in patterns]
        best_index = int(np.argmax(scores))
        return best_index, float(scores[best_index])

    def edge_activity(
        self,
        query: np.ndarray,
        candidate: np.ndarray,
        knowledge: KnowledgeHypergraph,
    ) -> float:
        """Суммирует веса гиперребер, активных и во входе, и в паттерне."""

        active_weight = 0.0

        for edge in knowledge.hyperedges:
            edge_query = query[list(edge.symbols)]
            if not np.all(self._known_mask(edge_query)):
                continue
            if not np.all(edge_query == 1):
                continue
            if np.all(candidate[list(edge.symbols)] == 1):
                active_weight += edge.weight

        return float(active_weight)

    @staticmethod
    def _known_mask(values: np.ndarray) -> np.ndarray:
        """Считает пропусками значения, которые нельзя сравнивать как известные биты."""

        prepared = np.asarray(values)
        if np.issubdtype(prepared.dtype, np.floating):
            return ~np.isnan(prepared)
        return prepared != -1
