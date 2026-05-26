"""Историческая сеть эпизодов АГ-памяти."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    """Фиксирует один акт восстановления паттерна."""

    original: np.ndarray
    distorted: np.ndarray
    restored: np.ndarray
    score: float


class EpisodeMemory:
    """Хранит последовательность эпизодов для последующей статистики."""

    def __init__(self) -> None:
        self.episodes: list[Episode] = []

    def add_episode(
        self,
        original: np.ndarray,
        distorted: np.ndarray,
        restored: np.ndarray,
        score: float,
    ) -> Episode:
        """Добавляет эпизод в историческую сеть."""

        episode = Episode(
            original=np.asarray(original).copy(),
            distorted=np.asarray(distorted).copy(),
            restored=np.asarray(restored).copy(),
            score=float(score),
        )
        self.episodes.append(episode)
        return episode

    def all(self) -> list[Episode]:
        """Возвращает копию списка эпизодов."""

        return list(self.episodes)

    def __len__(self) -> int:
        return len(self.episodes)
