"""Пакет для упрощенной модели ассоциативно-гетероархической памяти."""

from ah_memory.baselines import (
    dense_topology,
    evaluate_baselines,
    frequency_topology,
    pairwise_topology,
    random_topology,
)
from ah_memory.config import BaselineConfig, RetrievalConfig
from ah_memory.focus import ActivationFocus
from ah_memory.history import Episode, EpisodeMemory
from ah_memory.knowledge import Hyperedge, KnowledgeHypergraph
from ah_memory.memory import AHMemory, RetrievalResult
from ah_memory.metrics import (
    average_recovery_accuracy,
    close_pattern_confusion_rate,
    close_pattern_discrimination,
    evaluate_memory,
    exact_match_accuracy,
    structural_metrics,
)
from ah_memory.optimization import AHMemoryTopologyProblem, run_nsga2
from ah_memory.patterns import (
    ClusteredPatternData,
    PatternTestCase,
    corrupt_pattern,
    find_close_pairs,
    generate_clustered_pattern_data,
    generate_clustered_patterns,
    prepare_pattern_tests,
)
from ah_memory.symbols import SymbolSet

__all__ = [
    "AHMemory",
    "ActivationFocus",
    "Episode",
    "EpisodeMemory",
    "Hyperedge",
    "KnowledgeHypergraph",
    "ClusteredPatternData",
    "PatternTestCase",
    "RetrievalConfig",
    "RetrievalResult",
    "SymbolSet",
    "AHMemoryTopologyProblem",
    "BaselineConfig",
    "average_recovery_accuracy",
    "close_pattern_confusion_rate",
    "close_pattern_discrimination",
    "corrupt_pattern",
    "dense_topology",
    "evaluate_baselines",
    "evaluate_memory",
    "exact_match_accuracy",
    "find_close_pairs",
    "frequency_topology",
    "generate_clustered_pattern_data",
    "generate_clustered_patterns",
    "pairwise_topology",
    "prepare_pattern_tests",
    "random_topology",
    "run_nsga2",
    "structural_metrics",
]
