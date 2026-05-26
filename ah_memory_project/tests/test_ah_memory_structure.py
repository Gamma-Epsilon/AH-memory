import numpy as np

from ah_memory import AHMemory, KnowledgeHypergraph, SymbolSet


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


def test_knowledge_weights_follow_joint_activation_frequency():
    knowledge = KnowledgeHypergraph(3)
    knowledge.add_hyperedge((0, 1))
    knowledge.add_hyperedge((1, 2))

    knowledge.update_weights(np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]]))

    weights = [weight for _, weight in knowledge.as_tuples()]
    assert weights == [2 / 3, 2 / 3]
