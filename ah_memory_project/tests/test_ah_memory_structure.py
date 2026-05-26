import numpy as np

from ah_memory import AHMemory, KnowledgeHypergraph, SymbolSet
from ah_memory.config import RetrievalConfig
from ah_memory.patterns import PatternTestCase


def test_memory_retrieves_best_pattern_and_records_episode():
    symbols = SymbolSet(4)
    knowledge = KnowledgeHypergraph(4)
    knowledge.add_hyperedge((0, 1))
    knowledge.add_hyperedge((2, 3))
    memory = AHMemory(symbols, knowledge=knowledge)

    memory.fit(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]))
    index, restored = memory.retrieve(np.array([1, 1, -1, -1]))

    assert index == 0
    assert restored.tolist() == [1, 1, 0, 0]
    assert len(memory.history) == 1


def test_score_uses_bit_match_hyperedges_and_complexity_penalty():
    symbols = SymbolSet(4)
    knowledge = KnowledgeHypergraph(4)
    knowledge.add_hyperedge((0, 1), weight=0.5)
    knowledge.add_hyperedge((2, 3), weight=0.25)
    memory = AHMemory(
        symbols,
        knowledge=knowledge,
        retrieval_config=RetrievalConfig(
            bit_match_weight=2.0,
            hyperedge_weight=3.0,
            complexity_weight=0.1,
        ),
    )

    score = memory.score_pattern(np.array([1, 1, -1, -1]), np.array([1, 1, 0, 0]))

    assert score == 2.0 * 1.0 + 3.0 * 0.5 - 0.1 * 2


def test_retrieve_can_return_score():
    symbols = SymbolSet(4)
    knowledge = KnowledgeHypergraph(4)
    knowledge.add_hyperedge((0, 1))
    memory = AHMemory(symbols, knowledge=knowledge)
    memory.fit(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]))

    index, restored, score = memory.retrieve(np.array([1, 1, -1, -1]), return_score=True)

    assert index == 0
    assert restored.tolist() == [1, 1, 0, 0]
    assert isinstance(score, float)


def test_knowledge_weights_follow_joint_activation_frequency():
    knowledge = KnowledgeHypergraph(3)
    knowledge.add_hyperedge((0, 1))
    knowledge.add_hyperedge((1, 2))

    knowledge.update_weights(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]]))

    weights = [weight for _, weight in knowledge.as_tuples()]
    assert weights == [2 / 3, 2 / 3]


def test_retrieve_batch_returns_accuracy_and_exact_match():
    symbols = SymbolSet(4)
    knowledge = KnowledgeHypergraph(4)
    knowledge.add_hyperedge((0, 1))
    memory = AHMemory(symbols, knowledge=knowledge)
    memory.fit(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]))
    tests = [
        PatternTestCase(
            pattern_index=0,
            original=np.array([1, 1, 0, 0]),
            distorted=np.array([1, 1, -1, -1]),
            noise_level=0.0,
            missing_level=0.5,
        )
    ]

    results = memory.retrieve_batch(tests)

    assert len(results) == 1
    assert results[0].original_index == 0
    assert results[0].restored_index == 0
    assert results[0].accuracy == 1.0
    assert results[0].exact_match is True
