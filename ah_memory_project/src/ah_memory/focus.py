"""Фокус активации для оценки сохраненных паттернов."""

from __future__ import annotations

import numpy as np

from ah_memory.knowledge import KnowledgeHypergraph


class ActivationFocus:
    """Считает оценку соответствия запроса сохраненному паттерну."""

    def __init__(self, complexity_penalty: float = 0.01) -> None:
        if complexity_penalty < 0:
            raise ValueError("Штраф за сложность не может быть отрицательным.")
        self.complexity_penalty = float(complexity_penalty)

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

        edge_score = self._edge_score(prepared_query, prepared_candidate, knowledge)
        penalty = self.complexity_penalty * knowledge.complexity()
        return bit_score + edge_score - penalty

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

    def _edge_score(
        self,
        query: np.ndarray,
        candidate: np.ndarray,
        knowledge: KnowledgeHypergraph,
    ) -> float:
        active_weight = 0.0
        possible_weight = 0.0

        for edge in knowledge.hyperedges:
            edge_query = query[list(edge.symbols)]
            if not np.all(self._known_mask(edge_query)):
                continue
            if not np.all(edge_query == 1):
                continue
            possible_weight += edge.weight
            if np.all(candidate[list(edge.symbols)] == 1):
                active_weight += edge.weight

        if possible_weight == 0.0:
            return 0.0
        return float(active_weight / possible_weight)

    @staticmethod
    def _known_mask(values: np.ndarray) -> np.ndarray:
        """Считает пропусками значения, которые нельзя сравнивать как известные биты."""

        prepared = np.asarray(values)
        if np.issubdtype(prepared.dtype, np.floating):
            return ~np.isnan(prepared)
        return prepared != -1
